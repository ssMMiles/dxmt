#include "d3d11_pipeline_cache.hpp"
#include "airconv_public.h"
#include "d3d11_device.hpp"
#include "d3d11_shader.hpp"
#include "d3d11_pipeline.hpp"
#include "d3d11_pso_config_cache.hpp"
#include "log/log.hpp"
#include "sha1/sha1_util.hpp"
#include "../d3d10/d3d10_shader.hpp"
#include "../d3d10/d3d10_input_layout.hpp"
#include <cstring>
#include <shared_mutex>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <dirent.h>
#endif

namespace dxmt {

// Cache directory and utility functions
static std::string getCacheDir() {
  const char *custom_path = std::getenv("DXMT_CACHE_PATH");
  if (custom_path && custom_path[0] != '\0') {
    return std::string(custom_path);
  }

#ifdef _WIN32
  const char *tmp_dir = std::getenv("TEMP");
  if (!tmp_dir)
    tmp_dir = std::getenv("TMP");
  if (!tmp_dir)
    tmp_dir = "C:\\temp";
  return std::string(tmp_dir) + "\\dxmt_shader_cache";
#else
  const char *tmp_dir = std::getenv("TMPDIR");
  if (!tmp_dir)
    tmp_dir = "/tmp";
  return std::string(tmp_dir) + "/dxmt_shader_cache";
#endif
}

static void ensureCacheDir() {
  std::string cache_dir = getCacheDir();
#ifdef _WIN32
  _mkdir(cache_dir.c_str());
#else
  mkdir(cache_dir.c_str(), 0755);
#endif
}

static void logToCache(const std::string &message) {
  std::string cache_dir = getCacheDir();
  std::string log_path = cache_dir + "/../d3d11_cache.log";
  FILE *log_file = fopen(log_path.c_str(), "a");
  if (log_file) {
    fprintf(log_file, "%s\n", message.c_str());
    fclose(log_file);
  }
}

static std::string generateCacheFileName(const Sha1Hash &hash) {
  return getCacheDir() + "/" + hash.toString() + ".shader_cache";
}

struct CacheEntry {
  uint32_t magic;
  uint32_t version;
  uint64_t timestamp;
  uint32_t dxbc_size; // Size of original DXBC bytecode
  uint32_t hash_size; // Size of SHA1 hash (should be 20)
  // Followed by: SHA1 hash (20 bytes), then DXBC bytecode
};

static const uint32_t CACHE_MAGIC = 0xD3D11C4C; // "D3D11CAC"
static const uint32_t CACHE_VERSION = 1;

class CachedSM50Shader final : public Shader {
  MTLD3D11Device *device;
  sm50_shader_t shader = nullptr;
  Sha1Hash hash_;
  MTL_SHADER_REFLECTION reflection_;
  MTL_SM50_SHADER_ARGUMENT *arguments_info_buffer;
  std::unordered_map<ShaderVariant, std::unique_ptr<CompiledShader>> variants;

public:
  CachedSM50Shader(MTLD3D11Device *device, sm50_shader_t shader_transfered,
                   const Sha1Hash &hash, MTL_SHADER_REFLECTION &reflection)
      : device(device), shader(shader_transfered), hash_(hash),
        reflection_(reflection) {
    if (reflection_.NumConstantBuffers + reflection_.NumArguments) {
      arguments_info_buffer = (MTL_SM50_SHADER_ARGUMENT *)malloc(
          sizeof(MTL_SM50_SHADER_ARGUMENT) *
          (reflection_.NumConstantBuffers + reflection_.NumArguments));
      SM50GetArgumentsInfo(shader, arguments_info_buffer,
                           arguments_info_buffer +
                               reflection_.NumConstantBuffers);
    } else {
      arguments_info_buffer = nullptr;
    }
  }

  ~CachedSM50Shader() {
    if (shader) {
      SM50Destroy(shader);
      if (arguments_info_buffer)
        free(arguments_info_buffer);
      shader = nullptr;
    }
  };

  CachedSM50Shader(CachedSM50Shader &&moved) = delete;
  CachedSM50Shader(const CachedSM50Shader &copy) = delete;

  virtual sm50_shader_t handle() { return shader; };
  virtual MTL_SHADER_REFLECTION &reflection() { return reflection_; }
  virtual MTL_SM50_SHADER_ARGUMENT *constant_buffers_info() {
    return arguments_info_buffer;
  };
  virtual MTL_SM50_SHADER_ARGUMENT *arguments_info() {
    return arguments_info_buffer + reflection_.NumConstantBuffers;
  };
  virtual Com<CompiledShader> get_shader(ShaderVariant variant) {
    auto c = variants.insert({variant, nullptr});
    if (c.second) {
      c.first->second = std::visit(
          [=, this](auto var) {
            return CreateVariantShader(device, this, var);
          },
          variant);
      device->SubmitThreadgroupWork(c.first->second.get());
    }
    return c.first->second.get();
  }
  virtual const Sha1Hash &hash() { return hash_; };

#ifdef DXMT_DEBUG
  void *bytecode;
  size_t bytecode_length;

  virtual void dump() {
    std::fstream dump_out;
    dump_out.open("shader_dump_" + hash_.toString() + ".cso",
                  std::ios::out | std::ios::binary);
    if (dump_out) {
      dump_out.write((char *)bytecode, bytecode_length);
    }
    dump_out.close();
    WARN("shader dumped to ./shader_dump_" + hash_.toString() + ".cso");
  }
#else
  virtual void dump() {}
#endif
};

class CachedInputLayout final : public InputLayout {
private:
public:
  CachedInputLayout(
      std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> &&attributes,
      uint32_t input_slot_mask)
      : attributes_(attributes), input_slot_mask_(input_slot_mask) {}

  virtual uint32_t input_slot_mask() final { return input_slot_mask_; }

  virtual uint32_t input_layout_element(
      MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC **ppElements) final {
    *ppElements = attributes_.data();
    return attributes_.size();
  }

  std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> attributes_;
  uint32_t input_slot_mask_;
};

// Global shader cache shared across all devices
// static std::unordered_map<Sha1Hash, std::unique_ptr<CachedSM50Shader>>
// shaders_; static std::shared_mutex mutex_shares;

class MTLD3D11InputLayout final
    : public MTLD3D11DeviceChild<IMTLD3D11InputLayout> {
public:
  MTLD3D11InputLayout(MTLD3D11Device *device, ManagedInputLayout input_layout)
      : MTLD3D11DeviceChild<IMTLD3D11InputLayout>(device),
        input_layout(input_layout), d3d10(this) {}

  ~MTLD3D11InputLayout() {}

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid,
                                           void **ppvObject) final {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D11DeviceChild) ||
        riid == __uuidof(ID3D11InputLayout) ||
        riid == __uuidof(IMTLD3D11InputLayout)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (riid == __uuidof(ID3D10DeviceChild) ||
        riid == __uuidof(ID3D10InputLayout)) {
      *ppvObject = ref(&d3d10);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D11InputLayout), riid)) {
      WARN("D3D311InputLayout: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  };

  ManagedInputLayout GetManagedInputLayout() override { return input_layout; }

private:
  ManagedInputLayout input_layout;
  MTLD3D10InputLayout d3d10;
};

class PipelineCache : public MTLD3D11PipelineCacheBase {
  MTLD3D11Device *device;
  StateObjectCache<D3D11_BLEND_DESC1, IMTLD3D11BlendState> blend_states;

  std::unordered_map<MTL_INPUT_LAYOUT_DESC, std::unique_ptr<CachedInputLayout>>
      input_layouts;
  dxmt::mutex mutex_ia_;

  StateObjectCache<MTL_STREAM_OUTPUT_DESC, IMTLD3D11StreamOutputLayout>
      so_layouts;
  dxmt::mutex mutex_so_;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC,
                     Com<IMTLCompiledGraphicsPipeline>>
      pipelines_;
  dxmt::mutex mutex_;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC,
                     Com<IMTLCompiledTessellationPipeline>>
      pipelines_ts_;
  dxmt::mutex mutex_ts_;

  std::unordered_map<MTL_GRAPHICS_PIPELINE_DESC,
                     Com<IMTLCompiledGeometryPipeline>>
      pipelines_gs_;
  dxmt::mutex mutex_gs_;

  // Shader cache is now global - see g_shaders and g_shader_mutex above
  std::unordered_map<Sha1Hash, std::unique_ptr<CachedSM50Shader>> shaders_;
  mutable std::shared_mutex mutex_shares;

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid,
                                           void **ppvObject) final {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D11DeviceChild) ||
        riid == __uuidof(IMTLD3D11PipelineCache)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    return E_NOINTERFACE;
  }

  CachedSM50Shader *CreateShader(const void *pBytecode,
                                 uint32_t BytecodeLength) {
    auto sha1 = Sha1Hash::compute(pBytecode, BytecodeLength);
    {
      std::shared_lock<std::shared_mutex> lock(mutex_shares);
      auto result = shaders_.find(sha1);
      if (result != shaders_.end()) {
        return shaders_.at(sha1).get();
      }
    }
    sm50_error_t err;
    sm50_shader_t sm50;
    MTL_SHADER_REFLECTION reflection;
    if (SM50Initialize(pBytecode, BytecodeLength, &sm50, &reflection, &err)) {
      ERR("Failed to initialize shader: ", SM50GetErrorMessageString(err));
      SM50FreeError(err);
      return nullptr;
    }

    auto shader =
        std::make_unique<CachedSM50Shader>(device, sm50, sha1, reflection);

    bool acquired_lock = false;
    CachedSM50Shader *shader_ref = nullptr;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_shares);
      auto result = shaders_.find(sha1);
      if (result != shaders_.end()) {
        return shaders_.at(sha1).get();
      }

#ifdef DXMT_DEBUG
      shader->bytecode = malloc(BytecodeLength);
      shader->bytecode_length = BytecodeLength;
      memcpy(shader->bytecode, pBytecode, BytecodeLength);
#endif
      acquired_lock = true;
      shader_ref =
          shaders_.emplace(sha1, std::move(shader)).first->second.get();
    }

    if (acquired_lock) {
      ensureCacheDir();
      std::string cache_file = generateCacheFileName(sha1);
      logToCache("Saving DXBC bytecode to disk cache: " + cache_file);

      FILE *cache_out = fopen(cache_file.c_str(), "wb");
      if (cache_out) {
        CacheEntry header;
        header.magic = CACHE_MAGIC;
        header.version = CACHE_VERSION;
        header.timestamp = time(nullptr);
        header.dxbc_size = BytecodeLength;
        header.hash_size = 20; // SHA1 hash size

        bool write_success = true;
        write_success &= (fwrite(&header, sizeof(header), 1, cache_out) == 1);
        write_success &= (fwrite(&sha1, 20, 1, cache_out) == 1);
        write_success &= (fwrite(pBytecode, BytecodeLength, 1, cache_out) == 1);
        fclose(cache_out);

        if (!write_success) {
          WARN("Failed to write complete cache data to: " + cache_file);
        }

      } else {
        WARN("Failed to create disk cache file: " + cache_file);
      }
    }

    return shader_ref;
  }

  virtual HRESULT AddVertexShader(const void *pBytecode,
                                  uint32_t BytecodeLength,
                                  ID3D11VertexShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader = ref(new TShaderBase<ID3D11VertexShader, MTLD3D10VertexShader>(
        device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddPixelShader(const void *pBytecode, uint32_t BytecodeLength,
                                 ID3D11PixelShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader = ref(new TShaderBase<ID3D11PixelShader, MTLD3D10PixelShader>(
        device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddHullShader(const void *pBytecode, uint32_t BytecodeLength,
                                ID3D11HullShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader = ref(new TShaderBase<ID3D11HullShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddDomainShader(const void *pBytecode,
                                  uint32_t BytecodeLength,
                                  ID3D11DomainShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader =
        ref(new TShaderBase<ID3D11DomainShader>(device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddGeometryShader(const void *pBytecode,
                                    uint32_t BytecodeLength,
                                    ID3D11GeometryShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader =
        ref(new TShaderBase<ID3D11GeometryShader, MTLD3D10GeometryShader>(
            device, managed_shader));
    return S_OK;
  }

  virtual HRESULT AddComputeShader(const void *pBytecode,
                                   uint32_t BytecodeLength,
                                   ID3D11ComputeShader **ppShader) override {
    auto managed_shader = CreateShader(pBytecode, BytecodeLength);
    if (!managed_shader) {
      return E_FAIL;
    }
    *ppShader =
        ref(new TShaderBase<ID3D11ComputeShader>(device, managed_shader));
    return S_OK;
  }

  HRESULT AddInputLayout(const void *pShaderBytecodeWithInputSignature,
                         const D3D11_INPUT_ELEMENT_DESC *pInputElementDesc,
                         UINT NumElements,
                         IMTLD3D11InputLayout **ppInputLayout) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ia_);
    std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC> buffer(NumElements);
    uint32_t num_metal_ia_elements;
    HRESULT hr;
    if (FAILED(hr = ExtractMTLInputLayoutElements(
                   device, pShaderBytecodeWithInputSignature, pInputElementDesc,
                   NumElements, buffer.data(), &num_metal_ia_elements))) {
      return hr;
    }
    buffer.resize(num_metal_ia_elements);
    if (!input_layouts.contains(buffer)) {
      uint32_t input_slot_mask = 0;
      for (auto &element : buffer) {
        input_slot_mask |= (1 << element.Slot);
      }
      input_layouts.emplace(buffer, std::make_unique<CachedInputLayout>(
                                        std::move(buffer), input_slot_mask));
    }
    *ppInputLayout =
        ref(new MTLD3D11InputLayout(device, input_layouts.at(buffer).get()));
    return hr;
  }

  HRESULT
  AddStreamOutputLayout(const void *pShaderBytecode, UINT NumEntries,
                        const D3D11_SO_DECLARATION_ENTRY *pEntries,
                        UINT NumStrides, const UINT *pStrides,
                        IMTLD3D11StreamOutputLayout **ppSOLayout) override {
    std::lock_guard<dxmt::mutex> lock(mutex_so_);
    std::vector<MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC> buffer(NumEntries * 4);
    std::array<uint32_t, 4> strides = {{}};
    uint32_t num_metal_so_elements;
    if (FAILED(ExtractMTLStreamOutputElements(
            device, pShaderBytecode, NumEntries, pEntries, buffer.data(),
            &num_metal_so_elements))) {
      return E_FAIL;
    }
    buffer.resize(num_metal_so_elements);
    for (unsigned i = 0; i < NumStrides; i++) {
      strides[i] = pStrides[i];
    }
    MTL_STREAM_OUTPUT_DESC desc;
    memcpy(desc.Strides, strides.data(), sizeof(strides));
    desc.Elements = std::move(buffer);
    return so_layouts.CreateStateObject(&desc, ppSOLayout);
  };

  HRESULT AddBlendState(const D3D11_BLEND_DESC1 *pBlendDesc,
                        IMTLD3D11BlendState **ppBlendState) override {
    return blend_states.CreateStateObject(pBlendDesc, ppBlendState);
  }

  HRESULT AddInputLayoutDirect(const MTL_INPUT_LAYOUT_DESC &layout_desc,
                               IMTLD3D11InputLayout **ppInputLayout) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ia_);

    if (!input_layouts.contains(layout_desc)) {
      uint32_t input_slot_mask = 0;
      for (auto &element : layout_desc) {
        input_slot_mask |= (1 << element.Slot);
      }
      auto layout_desc_copy = layout_desc;
      input_layouts.emplace(layout_desc,
                            std::make_unique<CachedInputLayout>(
                                std::move(layout_desc_copy), input_slot_mask));
    }

    auto &cached_layout = input_layouts.at(layout_desc);
    *ppInputLayout = ref(new MTLD3D11InputLayout(device, cached_layout.get()));
    return S_OK;
  }

  HRESULT AddStreamOutputLayoutDirect(
      const MTL_STREAM_OUTPUT_DESC &so_desc,
      IMTLD3D11StreamOutputLayout **ppSOLayout) override {
    return so_layouts.CreateStateObject(&so_desc, ppSOLayout);
  }

  void GetGraphicsPipeline(MTL_GRAPHICS_PIPELINE_DESC *pDesc,
                           IMTLCompiledGraphicsPipeline **ppPipeline) override {
    std::lock_guard<dxmt::mutex> lock(mutex_);

    // std::hash<MTL_GRAPHICS_PIPELINE_DESC> pipeline_hash;
    // auto hash = pipeline_hash(*pDesc);

    auto iter = pipelines_.find(*pDesc);
    if (iter != pipelines_.end()) {
      // WARN("Graphics pipeline cache hit", " hash: ", std::to_string(hash));
      *ppPipeline = iter->second.ref();
      return;
    }
    // WARN("Graphics pipeline cache miss", " hash: ", std::to_string(hash));
    auto temp = dxmt::CreateGraphicsPipeline(device, pDesc);
    if (!pipelines_.insert({*pDesc, temp}).second) // copy
    {
      D3D11_ASSERT(0 && "duplicated graphics pipeline");
    }
    *ppPipeline = std::move(temp); // move
  }

  void GetTessellationPipeline(
      MTL_GRAPHICS_PIPELINE_DESC *pDesc,
      IMTLCompiledTessellationPipeline **ppPipeline) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ts_);

    auto iter = pipelines_ts_.find(*pDesc);
    if (iter != pipelines_ts_.end()) {
      *ppPipeline = iter->second.ref();
      return;
    }
    auto temp = dxmt::CreateTessellationPipeline(device, pDesc);
    if (!pipelines_ts_.insert({*pDesc, temp}).second) // copy
    {
      D3D11_ASSERT(0 && "duplicated tessellation pipeline");
    }
    *ppPipeline = std::move(temp);
  }

  void GetGeometryPipeline(MTL_GRAPHICS_PIPELINE_DESC *pDesc,
                           IMTLCompiledGeometryPipeline **ppPipeline) override {
    std::lock_guard<dxmt::mutex> lock(mutex_ts_);

    auto iter = pipelines_gs_.find(*pDesc);
    if (iter != pipelines_gs_.end()) {
      *ppPipeline = iter->second.ref();
      return;
    }
    auto temp = dxmt::CreateGeometryPipeline(device, pDesc);
    if (!pipelines_gs_.insert({*pDesc, temp}).second) // copy
    {
      D3D11_ASSERT(0 && "duplicated geometry pipeline");
    }
    *ppPipeline = std::move(temp);
  }

  // Pre-warming and cache management methods
  void PrewarmShaders() override {
    std::string cache_dir = getCacheDir();
    ensureCacheDir();

    Logger::info("[DXMT] Starting shader pre-warming from cache directory: " +
                 cache_dir);
    logToCache("Starting shader pre-warming from cache directory: " +
               cache_dir);

    int total_files = 0;
    int successful_prewarmed = 0;

    // Scan cache directory for .shader_cache files
#ifdef _WIN32
    WIN32_FIND_DATAA find_data;
    HANDLE find_handle =
        FindFirstFileA((cache_dir + "\\*.shader_cache").c_str(), &find_data);
    if (find_handle == INVALID_HANDLE_VALUE) {
      Logger::info("[DXMT] No cache files found for pre-warming");
      return;
    }

    do {
      std::string cache_file = cache_dir + "\\" + find_data.cFileName;
#else
    DIR *dir = opendir(cache_dir.c_str());
    if (!dir) {
      Logger::info("[DXMT] Failed to open cache directory for pre-warming");
      return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string filename = entry->d_name;
      if (filename.find(".shader_cache") == std::string::npos) {
        continue;
      }

      std::string cache_file = cache_dir + "/" + filename;
#endif

      total_files++;
      if (PrewarmShaderFromFile(cache_file)) {
        successful_prewarmed++;
      }

#ifdef _WIN32
    } while (FindNextFileA(find_handle, &find_data));
    FindClose(find_handle);
#else
    }
    closedir(dir);
#endif

    Logger::info(str::format(
        "[DXMT] Shader pre-warming completed. Successfully pre-warmed ",
        successful_prewarmed, "/", total_files,
        " shaders. Total in memory: ", shaders_.size()));
    logToCache("Shader pre-warming completed. Total shaders in memory: " +
               std::to_string(shaders_.size()));
  }

  size_t GetCachedShaderCount() const override {
    std::shared_lock<std::shared_mutex> lock(mutex_shares);
    return shaders_.size();
  }

public:
  PipelineCache(MTLD3D11Device *pDevice)
      : MTLD3D11PipelineCacheBase(pDevice), device(pDevice),
        blend_states(pDevice), so_layouts(pDevice) {
    logToCache("PipelineCache constructor called - device ptr: " +
               std::to_string(reinterpret_cast<uintptr_t>(pDevice)) +
               " - starting prewarming");

    // Check if pre-warming is disabled
    const char *disable_prewarm = std::getenv("DXMT_DISABLE_PREWARM");
    if (disable_prewarm && strcmp(disable_prewarm, "1") == 0) {
      logToCache("Shader pre-warming disabled via DXMT_DISABLE_PREWARM=1");
    } else {
      bool shaders_prewarmed = false;
      if (!shaders_prewarmed) {
        logToCache("Starting shader prewarming...");
        PrewarmShaders();
        logToCache("Shader prewarming completed");
        shaders_prewarmed = true;
      } else {
        logToCache(
            "Shader prewarming skipped - already done by previous device");
      }

      // Pre-warm PSO configurations after shaders are loaded
      extern PSO_ConfigCache g_pso_cache;
      if (g_pso_cache.Initialize()) {
        logToCache("Starting PSO configuration prewarming...");
        g_pso_cache.PrewarmPSOConfigurations(this);
        logToCache("PSO configuration prewarming completed");
      }
    }

    logToCache("PipelineCache constructor finished - device ptr: " +
               std::to_string(reinterpret_cast<uintptr_t>(pDevice)));
  };

  // Shader lookup helper for PSO pre-warming
  ManagedShader LookupShaderByHash(const Sha1Hash &hash) const override {
    std::shared_lock<std::shared_mutex> lock(mutex_shares);
    auto iter = shaders_.find(hash);
    return (iter != shaders_.end()) ? iter->second.get() : nullptr;
  }

private:
  bool PrewarmShaderFromFile(const std::string &cache_file) {
    logToCache("Pre-warming shader from: " + cache_file);

    FILE *cache = fopen(cache_file.c_str(), "rb");
    if (!cache) {
      logToCache("Failed to open cache file: " + cache_file);
      return false;
    }

    CacheEntry header;
    if (fread(&header, sizeof(header), 1, cache) != 1) {
      logToCache("Failed to read cache header from: " + cache_file);
      fclose(cache);
      return false;
    }

    if (header.magic != CACHE_MAGIC || header.version != CACHE_VERSION) {
      logToCache("Invalid cache file (wrong magic/version): " + cache_file);
      fclose(cache);
      return false;
    }

    // Read SHA1 hash
    Sha1Digest hash_data;
    if (fread(hash_data.data(), 20, 1, cache) != 1) {
      logToCache("Failed to read hash from: " + cache_file);
      fclose(cache);
      return false;
    }

    Sha1Hash sha1(hash_data);

    // Check if shader is already in memory
    {
      std::shared_lock<std::shared_mutex> lock(mutex_shares);
      if (shaders_.find(sha1) != shaders_.end()) {
        logToCache("Shader already in memory, skipping: " + sha1.toString());
        fclose(cache);
        return true; // Consider this a success since shader is already loaded
      }
    }

    // Read DXBC bytecode
    std::vector<uint8_t> bytecode(header.dxbc_size);
    if (fread(bytecode.data(), header.dxbc_size, 1, cache) != 1) {
      logToCache("Failed to read bytecode from: " + cache_file);
      fclose(cache);
      return false;
    }

    fclose(cache);

    // Verify hash matches
    auto computed_hash = Sha1Hash::compute(bytecode.data(), bytecode.size());
    if (computed_hash != sha1) {
      logToCache("Hash mismatch in cache file: " + cache_file);
      return false;
    }

    // Compile shader
    logToCache("Compiling pre-warmed shader: " + sha1.toString());

    sm50_error_t err;
    sm50_shader_t sm50;
    MTL_SHADER_REFLECTION reflection;
    if (SM50Initialize(bytecode.data(), bytecode.size(), &sm50, &reflection,
                       &err)) {
      logToCache("Failed to compile pre-warmed shader: " +
                 SM50GetErrorMessageString(err));
      SM50FreeError(err);
      return false;
    }

    auto shader =
        std::make_unique<CachedSM50Shader>(device, sm50, sha1, reflection);

    {
      std::unique_lock<std::shared_mutex> lock(mutex_shares);
      shaders_.emplace(sha1, std::move(shader));
    }

    logToCache("Successfully pre-warmed shader: " + sha1.toString());
    return true;
  }
};

std::unique_ptr<MTLD3D11PipelineCacheBase>
InitializePipelineCache(MTLD3D11Device *device) {
  return std::make_unique<PipelineCache>(device);
}

// Helper function for shader lookups that handles null hashes
ManagedShader LookupShaderIfValid(PipelineCache *cache, const Sha1Hash &hash) {
  if (hash == Sha1Hash()) { // Null/empty hash
    return nullptr;
  }
  return cache->LookupShaderByHash(hash);
}

}; // namespace dxmt