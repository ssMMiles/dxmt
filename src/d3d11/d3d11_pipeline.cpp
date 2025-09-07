#include "Metal.hpp"
#include "d3d11_private.h"
#include "d3d11_pipeline.hpp"
#include "d3d11_pso_config_cache.hpp"
#include "com/com_object.hpp"
#include "d3d11_device.hpp"
#include "d3d11_shader.hpp"
#include "log/log.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>

namespace dxmt {

// Global PSO configuration cache
PSO_ConfigCache g_pso_cache;

// Initialize PSO cache (called once)
static void InitializePSOCache() {
  static bool initialized = false;
  if (!initialized) {
    g_pso_cache.Initialize();
    initialized = true;
  }
}

// Helper function to compute deterministic input layout hash
static uint64_t ComputeInputLayoutHash(ManagedInputLayout input_layout) {
  if (!input_layout) {
    TRACE("DEBUG: input_layout is null, returning 0");
    return 0;
  }

  MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC *elements = nullptr;
  uint32_t element_count = input_layout->input_layout_element(&elements);

  TRACE("DEBUG: input_layout has ", element_count, " elements");

  if (element_count == 0 || !elements) {
    TRACE("DEBUG: no elements, returning 0");
    return 0;
  }

  // Create MTL_INPUT_LAYOUT_DESC and use existing hash function
  MTL_INPUT_LAYOUT_DESC desc(elements, elements + element_count);
  std::hash<MTL_INPUT_LAYOUT_DESC> hasher;
  uint64_t hash = static_cast<uint64_t>(hasher(desc));

  TRACE("DEBUG: computed input layout hash: ", std::hex, hash);
  return hash;
}

// Helper functions to serialize shader variants deterministically
static SerializedVertexVariant
SerializeVertexVariant(const ShaderVariant &variant) {
  SerializedVertexVariant serialized = {}; // Zero-initialize

  if (std::holds_alternative<ShaderVariantVertex>(variant)) {
    const auto &v = std::get<ShaderVariantVertex>(variant);
    serialized.variant_type = 1;
    serialized.input_layout_handle = v.input_layout_handle;
    serialized.gs_passthrough = v.gs_passthrough;
    serialized.rasterization_disabled = v.rasterization_disabled;
  } else if (std::holds_alternative<ShaderVariantVertexStreamOutput>(variant)) {
    const auto &v = std::get<ShaderVariantVertexStreamOutput>(variant);
    serialized.variant_type = 2;
    serialized.input_layout_handle = v.input_layout_handle;
    serialized.stream_output_layout_handle = v.stream_output_layout_handle;
  } else {
    // ShaderVariantDefault or unknown
    serialized.variant_type = 0;
  }

  return serialized;
}

static SerializedPixelVariant
SerializePixelVariant(const ShaderVariant &variant) {
  SerializedPixelVariant serialized = {}; // Zero-initialize

  if (std::holds_alternative<ShaderVariantPixel>(variant)) {
    const auto &v = std::get<ShaderVariantPixel>(variant);
    serialized.variant_type = 1;
    serialized.sample_mask = v.sample_mask;
    serialized.dual_source_blending = v.dual_source_blending;
    serialized.disable_depth_output = v.disable_depth_output;
    serialized.unorm_output_reg_mask = v.unorm_output_reg_mask;
  } else {
    // ShaderVariantDefault or unknown
    serialized.variant_type = 0;
  }

  return serialized;
}

class MTLCompiledGraphicsPipeline
    : public ComObject<IMTLCompiledGraphicsPipeline> {
public:
  MTLCompiledGraphicsPipeline(MTLD3D11Device *pDevice,
                              MTL_GRAPHICS_PIPELINE_DESC *pDesc)
      : ComObject<IMTLCompiledGraphicsPipeline>(),
        num_rtvs(pDesc->NumColorAttachments),
        depth_stencil_format(pDesc->DepthStencilFormat),
        topology_class(pDesc->TopologyClass), device_(pDevice),
        pBlendState(pDesc->BlendState),
        RasterizationEnabled(pDesc->RasterizationEnabled),
        SampleCount(pDesc->SampleCount),
        vertex_variant_(ShaderVariantDefault{}),
        pixel_variant_(ShaderVariantDefault{}),
        vertex_shader_managed_(pDesc->VertexShader),
        pixel_shader_managed_(pDesc->PixelShader),
        hull_shader_managed_(pDesc->HullShader), // Store all shader stages
        domain_shader_managed_(pDesc->DomainShader),
        geometry_shader_managed_(pDesc->GeometryShader),
        sample_mask_(pDesc->SampleMask), // Store additional pipeline data
        gs_strip_topology_(pDesc->GSStripTopology),
        gs_passthrough_(pDesc->GSPassthrough),
        index_buffer_format_(pDesc->IndexBufferFormat),
        so_layout_(pDesc->SOLayout) {

    uint32_t unorm_output_reg_mask = 0;
    for (unsigned i = 0; i < num_rtvs; i++) {
      rtv_formats[i] = pDesc->ColorAttachmentFormats[i];
      unorm_output_reg_mask |= (uint32_t(IsUnorm8RenderTargetFormat(pDesc->ColorAttachmentFormats[i])) << i);
    }

    uint64_t input_layout_hash = ComputeInputLayoutHash(pDesc->InputLayout);
    uint64_t input_layout_pointer = (uint64_t)pDesc->InputLayout;

    if (pDesc->InputLayout) {
      MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC *elements = nullptr;
      uint32_t element_count =
          pDesc->InputLayout->input_layout_element(&elements);
      if (elements && element_count > 0) {
        stored_input_layout_elements_.assign(elements,
                                             elements + element_count);
      } else {
        WARN("DEBUG: InputLayout exists but has no elements");
      }
    } else {
      WARN("DEBUG: No InputLayout provided to pipeline");
    }

    if (pDesc->SOLayout) {
      ShaderVariantVertexStreamOutput shader_variant{input_layout_pointer,
                                                     (uint64_t)pDesc->SOLayout};
      vertex_variant_ = ShaderVariantVertexStreamOutput{
          input_layout_hash, (uint64_t)pDesc->SOLayout};
      VertexShader = pDesc->VertexShader->get_shader(shader_variant);
    } else {
      ShaderVariantVertex shader_variant{input_layout_pointer,
                                         pDesc->GSPassthrough,
                                         !pDesc->RasterizationEnabled};
      vertex_variant_ =
          ShaderVariantVertex{input_layout_hash, pDesc->GSPassthrough,
                              !pDesc->RasterizationEnabled};
      VertexShader = pDesc->VertexShader->get_shader(shader_variant);
    }

    if (pDesc->PixelShader) {
      pixel_variant_ = ShaderVariantPixel{
          pDesc->SampleMask, pDesc->BlendState->IsDualSourceBlending(),
          depth_stencil_format == WMTPixelFormatInvalid, unorm_output_reg_mask};
      PixelShader = pDesc->PixelShader->get_shader(
          std::get<ShaderVariantPixel>(pixel_variant_));
    }
  }

  void SubmitWork() { device_->SubmitThreadgroupWork(this); }

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(IMTLThreadpoolWork) ||
        riid == __uuidof(IMTLCompiledGraphicsPipeline)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    return E_NOINTERFACE;
  }

  bool IsReady() final { return ready_.load(std::memory_order_relaxed); }

  void GetPipeline(MTL_COMPILED_GRAPHICS_PIPELINE *pPipeline) final {
    ready_.wait(false, std::memory_order_acquire);
    *pPipeline = {state_};
  }

  PSO_Configuration ExtractPSOConfiguration(const WMTRenderPipelineInfo &info) {
    PSO_Configuration config = {};

    // Zero-initialize the entire structure to avoid padding bytes affecting
    // hash
    memset(&config, 0, sizeof(config));

    // Extract deterministic shader hashes from ALL managed shaders
    if (vertex_shader_managed_) {
      config.vertex_shader_hash = vertex_shader_managed_->hash();
      config.vertex_variant = SerializeVertexVariant(vertex_variant_);
      TRACE("DEBUG: Vertex shader hash: ",
            config.vertex_shader_hash.toString());
    }

    if (pixel_shader_managed_) {
      config.pixel_shader_hash = pixel_shader_managed_->hash();
      config.pixel_variant = SerializePixelVariant(pixel_variant_);
    }

    if (hull_shader_managed_) {
      config.hull_shader_hash = hull_shader_managed_->hash();
    }

    if (domain_shader_managed_) {
      config.domain_shader_hash = domain_shader_managed_->hash();
    }

    if (geometry_shader_managed_) {
      config.geometry_shader_hash = geometry_shader_managed_->hash();
    }

    config.num_rtvs = static_cast<uint8_t>(num_rtvs);
    for (UINT i = 0; i < num_rtvs && i < 8; i++) {
      config.rtv_formats[i] = rtv_formats[i];
    }
    for (UINT i = num_rtvs; i < 8; i++) {
      config.rtv_formats[i] = WMTPixelFormatInvalid;
    }

    config.depth_format = depth_stencil_format;
    config.stencil_format =
        depth_stencil_format; // FIXME: extract stencil separately

    config.topology = topology_class;
    config.sample_count = SampleCount;
    config.sample_mask = sample_mask_;
    config.rasterization_enabled = RasterizationEnabled;
    config.gs_strip_topology = gs_strip_topology_;
    config.gs_passthrough = gs_passthrough_;
    config.index_buffer_format = index_buffer_format_;

    if (!stored_input_layout_elements_.empty()) {
      size_t total_size = stored_input_layout_elements_.size() *
                          sizeof(MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC);
      if (total_size <= sizeof(config.input_layout_data)) {
        memcpy(config.input_layout_data, stored_input_layout_elements_.data(),
               total_size);
        config.input_layout_size = static_cast<uint32_t>(total_size);
      } else {
        WARN("Input layout data too large for cache: ", total_size, " > ",
             sizeof(config.input_layout_data));
      }
    }

    if (so_layout_) {
      MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC *elements = nullptr;
      uint32_t strides[4] = {0};
      uint32_t element_count =
          so_layout_->GetStreamOutputElements(&elements, strides);

      struct SOLayoutData {
        uint32_t strides[4];
        uint32_t element_count;
      } so_data = {};

      memcpy(so_data.strides, strides, sizeof(strides));
      so_data.element_count = element_count;

      size_t total_size =
          sizeof(so_data) +
          (element_count * sizeof(MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC));
      if (total_size <= sizeof(config.stream_output_data)) {
        memcpy(config.stream_output_data, &so_data, sizeof(so_data));
        if (elements && element_count > 0) {
          memcpy(config.stream_output_data + sizeof(so_data), elements,
                 element_count * sizeof(MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC));
        }
        config.stream_output_size = static_cast<uint32_t>(total_size);
      }
    }

    return config;
  }

  IMTLThreadpoolWork *RunThreadpoolWork() {
    auto pso_start_time = std::chrono::high_resolution_clock::now();

    TRACE("Start compiling 1 graphics PSO");

    WMT::Reference<WMT::Error> err;
    MTL_COMPILED_SHADER vs, ps;
    if (!VertexShader->GetShader(&vs)) {
      return VertexShader.ptr();
    }
    if (PixelShader && !PixelShader->GetShader(&ps)) {
      return PixelShader.ptr();
    }

    WMTRenderPipelineInfo info;
    WMT::InitializeRenderPipelineInfo(info);

    info.vertex_function = vs.Function;

    if (PixelShader) {
      info.fragment_function = ps.Function;
    }
    info.rasterization_enabled = RasterizationEnabled;

    for (unsigned i = 0; i < num_rtvs; i++) {
      if (rtv_formats[i] == WMTPixelFormatInvalid)
        continue;
      info.colors[i].pixel_format = rtv_formats[i];
    }

    if (depth_stencil_format != WMTPixelFormatInvalid) {
      info.depth_pixel_format = depth_stencil_format;
    }
    // FIXME: don't hardcoding!
    if (depth_stencil_format == WMTPixelFormatDepth32Float_Stencil8 ||
        depth_stencil_format == WMTPixelFormatDepth24Unorm_Stencil8 ||
        depth_stencil_format == WMTPixelFormatStencil8) {
      info.stencil_pixel_format = depth_stencil_format;
    }

    if (pBlendState) {
      pBlendState->SetupMetalPipelineDescriptor((WMTRenderPipelineBlendInfo *)&info, num_rtvs);
    }

    info.input_primitive_topology = topology_class;
    info.raster_sample_count = SampleCount;
    info.immutable_vertex_buffers = (1 << 16) | (1 << 29) | (1 << 30);
    info.immutable_fragment_buffers = (1 << 29) | (1 << 30);

    // Check PSO configuration cache
    PSO_Configuration pso_config = ExtractPSOConfiguration(info);

    // Debug: Log comprehensive PSO configuration in one line for easy
    // comparison
    std::string vs_hash =
        vertex_shader_managed_
            ? pso_config.vertex_shader_hash.toString().substr(0, 8)
            : "none";
    std::string ps_hash =
        pixel_shader_managed_
            ? pso_config.pixel_shader_hash.toString().substr(0, 8)
            : "none";
    std::string vs_variant_info = "default";
    std::string ps_variant_info = "default";

    // Extract variant info from serialized format
    if (pso_config.vertex_variant.variant_type == 1) {
      // ShaderVariantVertex
      vs_variant_info =
          "V:" + std::to_string(pso_config.vertex_variant.input_layout_handle) +
          ":" + std::to_string(pso_config.vertex_variant.gs_passthrough) + ":" +
          std::to_string(pso_config.vertex_variant.rasterization_disabled);
    } else if (pso_config.vertex_variant.variant_type == 2) {
      // ShaderVariantVertexStreamOutput
      vs_variant_info =
          "SO:" +
          std::to_string(pso_config.vertex_variant.input_layout_handle) + ":" +
          std::to_string(pso_config.vertex_variant.stream_output_layout_handle);
    }

    if (pso_config.pixel_variant.variant_type == 1) {
      // ShaderVariantPixel
      ps_variant_info =
          "P:" + std::to_string(pso_config.pixel_variant.sample_mask) + ":" +
          std::to_string(pso_config.pixel_variant.dual_source_blending) + ":" +
          std::to_string(pso_config.pixel_variant.disable_depth_output) + ":" +
          std::to_string(pso_config.pixel_variant.unorm_output_reg_mask);
    }

    auto config_hash = g_pso_cache.HashConfiguration(pso_config);

    // Debug: Hash individual components to identify differences
    auto vs_hash_only =
        Sha1Hash::compute((uint8_t *)&pso_config.vertex_shader_hash,
                          sizeof(pso_config.vertex_shader_hash));
    auto ps_hash_only =
        Sha1Hash::compute((uint8_t *)&pso_config.pixel_shader_hash,
                          sizeof(pso_config.pixel_shader_hash));
    auto variant_hash = Sha1Hash::compute((uint8_t *)&pso_config.vertex_variant,
                                          sizeof(pso_config.vertex_variant) +
                                              sizeof(pso_config.pixel_variant));
    auto state_hash = Sha1Hash::compute(
        (uint8_t *)&pso_config.rtv_formats,
        sizeof(pso_config.rtv_formats) + sizeof(pso_config.num_rtvs) +
            sizeof(pso_config.depth_format) +
            sizeof(pso_config.stencil_format) + sizeof(pso_config.topology) +
            sizeof(pso_config.sample_count) +
            sizeof(pso_config.rasterization_enabled));

    // Compute deterministic blend hash directly from blend state
    Sha1Hash blend_hash;
    if (pBlendState) {
      // Get the actual blend descriptor content for deterministic hashing
      D3D11_BLEND_DESC1 blend_desc;
      memset(&blend_desc, 0,
             sizeof(blend_desc)); // Zero-initialize to avoid padding bytes
      pBlendState->GetDesc1(&blend_desc);

      // Create the same deterministic structure as in ExtractPSOConfiguration
      struct DeterministicBlendState {
        BOOL IndependentBlendEnable;
        BOOL AlphaToCoverageEnable;
        struct {
          BOOL BlendEnable;
          BOOL LogicOpEnable;
          D3D11_BLEND SrcBlend;
          D3D11_BLEND DestBlend;
          D3D11_BLEND_OP BlendOp;
          D3D11_BLEND SrcBlendAlpha;
          D3D11_BLEND DestBlendAlpha;
          D3D11_BLEND_OP BlendOpAlpha;
          D3D11_LOGIC_OP LogicOp;
          UINT8 RenderTargetWriteMask;
        } RenderTarget[8];
      } deterministic_blend = {};

      // Copy only the meaningful fields
      deterministic_blend.IndependentBlendEnable =
          blend_desc.IndependentBlendEnable;
      deterministic_blend.AlphaToCoverageEnable =
          blend_desc.AlphaToCoverageEnable;

      for (int i = 0; i < 8; i++) {
        deterministic_blend.RenderTarget[i].BlendEnable =
            blend_desc.RenderTarget[i].BlendEnable;
        deterministic_blend.RenderTarget[i].SrcBlend =
            blend_desc.RenderTarget[i].SrcBlend;
        deterministic_blend.RenderTarget[i].DestBlend =
            blend_desc.RenderTarget[i].DestBlend;
        deterministic_blend.RenderTarget[i].BlendOp =
            blend_desc.RenderTarget[i].BlendOp;
        deterministic_blend.RenderTarget[i].SrcBlendAlpha =
            blend_desc.RenderTarget[i].SrcBlendAlpha;
        deterministic_blend.RenderTarget[i].DestBlendAlpha =
            blend_desc.RenderTarget[i].DestBlendAlpha;
        deterministic_blend.RenderTarget[i].BlendOpAlpha =
            blend_desc.RenderTarget[i].BlendOpAlpha;
        deterministic_blend.RenderTarget[i].LogicOpEnable =
            blend_desc.RenderTarget[i].LogicOpEnable;
        deterministic_blend.RenderTarget[i].LogicOp =
            blend_desc.RenderTarget[i].LogicOp;
        deterministic_blend.RenderTarget[i].RenderTargetWriteMask =
            blend_desc.RenderTarget[i].RenderTargetWriteMask;
      }

      // Hash the deterministic structure directly
      blend_hash = Sha1Hash::compute((uint8_t *)&deterministic_blend,
                                     sizeof(deterministic_blend));
    } else {
      // No blend state, hash empty data
      blend_hash = Sha1Hash::compute(nullptr, 0);
    }

    auto input_hash =
        Sha1Hash::compute((uint8_t *)&pso_config.input_layout_data,
                          sizeof(pso_config.input_layout_data) +
                              sizeof(pso_config.input_layout_size));
    auto padding_hash = Sha1Hash::compute((uint8_t *)&pso_config.padding,
                                          sizeof(pso_config.padding));

    WARN("PSO_CONFIG: ", config_hash.toString().substr(0, 12), " VS=", vs_hash,
         "(", vs_variant_info, ")", " PS=", ps_hash, "(", ps_variant_info, ")",
         " HS=",
         (hull_shader_managed_
              ? pso_config.hull_shader_hash.toString().substr(0, 8)
              : "none"),
         " DS=",
         (domain_shader_managed_
              ? pso_config.domain_shader_hash.toString().substr(0, 8)
              : "none"),
         " GS=",
         (geometry_shader_managed_
              ? pso_config.geometry_shader_hash.toString().substr(0, 8)
              : "none"),
         " RTVs=", (int)pso_config.num_rtvs, "[",
         (int)pso_config.rtv_formats[0], ",", (int)pso_config.rtv_formats[1],
         ",", (int)pso_config.rtv_formats[2], ",",
         (int)pso_config.rtv_formats[3], ",", (int)pso_config.rtv_formats[4],
         ",", (int)pso_config.rtv_formats[5], ",",
         (int)pso_config.rtv_formats[6], ",", (int)pso_config.rtv_formats[7],
         "]", " Depth=", (int)pso_config.depth_format,
         " Topo=", (int)pso_config.topology,
         " Samples=", pso_config.sample_count, "/", pso_config.sample_mask,
         " Raster=", pso_config.rasterization_enabled,
         " GSStrip=", pso_config.gs_strip_topology,
         " GSPass=", pso_config.gs_passthrough,
         " IdxFmt=", pso_config.index_buffer_format,
         " InputSize=", pso_config.input_layout_size,
         " SOSize=", pso_config.stream_output_size,
         " VariantIdx=", (int)pso_config.vertex_variant.variant_type, "/",
         (int)pso_config.pixel_variant.variant_type,
         " AllHashes=", vs_hash_only.toString().substr(0, 4), "/",
         ps_hash_only.toString().substr(0, 4), "/",
         variant_hash.toString().substr(0, 4), "/",
         state_hash.toString().substr(0, 4), "/",
         blend_hash.toString().substr(0, 4), "/",
         input_hash.toString().substr(0, 4), "/",
         padding_hash.toString().substr(0, 4));

    std::string cache_file = g_pso_cache.CheckCacheHit(pso_config);
    if (!cache_file.empty()) {
      WARN(" PERF: PSO config Cache HIT - ",
           std::filesystem::path(cache_file).filename().string());
    } else {
      WARN(" PERF: PSO config Cache MISS - ",
           config_hash.toString().substr(0, 12), " VS=", vs_hash, "(",
           vs_variant_info, ")", " PS=", ps_hash, "(", ps_variant_info, ")",
           " HS=",
           (hull_shader_managed_
                ? pso_config.hull_shader_hash.toString().substr(0, 8)
                : "none"),
           " DS=",
           (domain_shader_managed_
                ? pso_config.domain_shader_hash.toString().substr(0, 8)
                : "none"),
           " GS=",
           (geometry_shader_managed_
                ? pso_config.geometry_shader_hash.toString().substr(0, 8)
                : "none"),
           " RTVs=", (int)pso_config.num_rtvs, "[",
           (int)pso_config.rtv_formats[0], ",", (int)pso_config.rtv_formats[1],
           ",", (int)pso_config.rtv_formats[2], ",",
           (int)pso_config.rtv_formats[3], ",", (int)pso_config.rtv_formats[4],
           ",", (int)pso_config.rtv_formats[5], ",",
           (int)pso_config.rtv_formats[6], ",", (int)pso_config.rtv_formats[7],
           "]", " Depth=", (int)pso_config.depth_format,
           " Topo=", (int)pso_config.topology,
           " Samples=", pso_config.sample_count, "/", pso_config.sample_mask,
           " Raster=", pso_config.rasterization_enabled,
           " GSStrip=", pso_config.gs_strip_topology,
           " GSPass=", pso_config.gs_passthrough,
           " IdxFmt=", pso_config.index_buffer_format,
           " InputSize=", pso_config.input_layout_size,
           " SOSize=", pso_config.stream_output_size,
           " VariantIdx=", (int)pso_config.vertex_variant.variant_type, "/",
           (int)pso_config.pixel_variant.variant_type,
           " AllHashes=", vs_hash_only.toString().substr(0, 4), "/",
           ps_hash_only.toString().substr(0, 4), "/",
           variant_hash.toString().substr(0, 4), "/",
           state_hash.toString().substr(0, 4), "/",
           blend_hash.toString().substr(0, 4), "/",
           input_hash.toString().substr(0, 4), "/",
           padding_hash.toString().substr(0, 4));
    }

    auto metal_compile_start = std::chrono::high_resolution_clock::now();
    state_ = device_->GetMTLDevice().newRenderPipelineState(info, err);
    auto metal_compile_end = std::chrono::high_resolution_clock::now();

    if (state_ == nullptr) {
      ERR("Failed to create PSO: ", err.description().getUTF8String());
      return this;
    }

    // Save PSO configuration to cache (only if cache was miss)
    if (cache_file.empty()) {
      if (g_pso_cache.SaveConfiguration(pso_config)) {
        WARN("PSO_CACHE_SAVE: ", config_hash.toString().substr(0, 12),
             ".pso_config - ", "VS=", vs_hash, "(", vs_variant_info,
             ") PS=", ps_hash, "(", ps_variant_info, ")");
      } else {
        WARN("Failed to save PSO configuration to cache");
      }
    }

    auto pso_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        pso_end_time - pso_start_time);
    auto metal_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        metal_compile_end - metal_compile_start);

    if (total_duration.count() > 1000) { // Log if PSO creation takes > 1ms
      WARN(" PERF: Graphics PSO creation took ", total_duration.count(),
           "μs (Metal compile: ", metal_duration.count(),
           "μs) - potential stutter");
    } else {
      TRACE("Compiled 1 graphics PSO in ", total_duration.count(),
            "μs (Metal: ", metal_duration.count(), "μs)");
    }

    return this;
  }

  bool GetIsDone() { return ready_; }

  void SetIsDone(bool state) {
    ready_.store(state);
    ready_.notify_all();
  }

private:
  UINT num_rtvs;
  WMTPixelFormat rtv_formats[8];
  WMTPixelFormat depth_stencil_format;
  WMTPrimitiveTopologyClass topology_class;
  MTLD3D11Device *device_;
  std::atomic_bool ready_;
  Com<CompiledShader> VertexShader;
  Com<CompiledShader> PixelShader;
  IMTLD3D11BlendState *pBlendState;
  WMT::Reference<WMT::RenderPipelineState> state_;
  bool RasterizationEnabled;
  UINT SampleCount;

  // Store shader variants and managed shaders for deterministic PSO config
  // extraction
  ShaderVariant vertex_variant_;
  ShaderVariant pixel_variant_;
  ManagedShader vertex_shader_managed_;
  ManagedShader pixel_shader_managed_;
  ManagedShader hull_shader_managed_; // NEW: Store all shader stages
  ManagedShader domain_shader_managed_;
  ManagedShader geometry_shader_managed_;

  // Store additional pipeline descriptor data for complete PSO config
  uint32_t sample_mask_;
  bool gs_strip_topology_;
  uint32_t gs_passthrough_;
  uint32_t index_buffer_format_;
  IMTLD3D11StreamOutputLayout *so_layout_;

  // Store input layout elements for serialization
  std::vector<MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC>
      stored_input_layout_elements_;
};

Com<IMTLCompiledGraphicsPipeline>
CreateGraphicsPipeline(MTLD3D11Device *pDevice,
                       MTL_GRAPHICS_PIPELINE_DESC *pDesc) {
  InitializePSOCache();
  Com<IMTLCompiledGraphicsPipeline> pipeline =
      new MTLCompiledGraphicsPipeline(pDevice, pDesc);
  pipeline->SubmitWork();
  return pipeline;
}

class MTLCompiledComputePipeline
    : public ComObject<IMTLCompiledComputePipeline> {
public:
  MTLCompiledComputePipeline(MTLD3D11Device *pDevice, ManagedShader shader)
      : ComObject<IMTLCompiledComputePipeline>(), device_(pDevice) {
    ComputeShader = shader->get_shader(ShaderVariantDefault{});
  }

  void SubmitWork() final { device_->SubmitThreadgroupWork(this); }

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(IMTLThreadpoolWork) ||
        riid == __uuidof(IMTLCompiledComputePipeline)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    return E_NOINTERFACE;
  }

  bool IsReady() final { return ready_.load(std::memory_order_relaxed); }

  void GetPipeline(MTL_COMPILED_COMPUTE_PIPELINE *pPipeline) final {
    ready_.wait(false, std::memory_order_acquire);
    *pPipeline = {state_};
  }

  IMTLThreadpoolWork *RunThreadpoolWork() {
    D3D11_ASSERT(!ready_ && "?wtf"); // TODO: should use a lock?

    auto pso_start_time = std::chrono::high_resolution_clock::now();
    TRACE("Start compiling 1 compute PSO");

    WMT::Reference<WMT::Error> err;
    MTL_COMPILED_SHADER cs;
    if (!ComputeShader->GetShader(&cs)) {
      return ComputeShader.ptr();
    }

    auto metal_compile_start = std::chrono::high_resolution_clock::now();
    state_ = device_->GetMTLDevice().newComputePipelineState(cs.Function, err);
    auto metal_compile_end = std::chrono::high_resolution_clock::now();

    if (!state_) {
      ERR("Failed to create compute PSO: ", err.description().getUTF8String());
      return this;
    }

    auto pso_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        pso_end_time - pso_start_time);
    auto metal_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        metal_compile_end - metal_compile_start);

    if (total_duration.count() > 1000) { // Log if PSO creation takes > 1ms
      WARN(" PERF: Compute PSO creation took ", total_duration.count(),
           "μs (Metal compile: ", metal_duration.count(),
           "μs) - potential stutter");
    } else {
      TRACE("Compiled 1 compute PSO in ", total_duration.count(),
            "μs (Metal: ", metal_duration.count(), "μs)");
    }

    return this;
  }

  bool GetIsDone() { return ready_; }

  void SetIsDone(bool state) {
    ready_.store(state);
    ready_.notify_all();
  }

private:
  MTLD3D11Device *device_;
  std::atomic_bool ready_;
  Com<CompiledShader> ComputeShader;
  WMT::Reference<WMT::ComputePipelineState> state_;
};

Com<IMTLCompiledComputePipeline>
CreateComputePipeline(MTLD3D11Device *pDevice, ManagedShader ComputeShader) {
  Com<IMTLCompiledComputePipeline> pipeline =
      new MTLCompiledComputePipeline(pDevice, ComputeShader);
  pipeline->SubmitWork();
  return pipeline;
}

} // namespace dxmt