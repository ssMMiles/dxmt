#include "d3d11_pso_config_cache.hpp"
#include "log/log.hpp"
#include "../util/util_env.hpp"
#include "../util/perf_log.hpp"
#include "../util/util_string.hpp"
#include "../util/com/com_pointer.hpp"
#include "d3d11_interfaces.hpp"
#include "d3d11_pipeline.hpp"
#include "d3d11_pipeline_cache.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <exception>
#ifdef _WIN32
#include <windows.h>
#else
#include "native/windows/windows_base.h"
#endif

namespace dxmt {

PSO_ConfigCache::PSO_ConfigCache() : caching_enabled_(true) {}

PSO_ConfigCache::~PSO_ConfigCache() {}

bool PSO_ConfigCache::Initialize() {
  // Check if PSO caching is disabled
  if (env::getEnvVar("DXMT_PSO_CACHE_DISABLE") == "1") {
    Logger::info(
        "PSO configuration caching disabled via DXMT_PSO_CACHE_DISABLE");
    caching_enabled_ = false;
    return true;
  }

  // Determine cache directory
  std::string cache_path = env::getEnvVar("DXMT_PSO_CACHE_PATH");
  if (cache_path.empty()) {
    // Fallback to main DXMT cache path + pso_cache subdirectory
    cache_path = env::getEnvVar("DXMT_CACHE_PATH");
    if (cache_path.empty()) {
      // Final fallback to temp directory using standard environment variables
#ifdef _WIN32
      const char *temp_dir = std::getenv("TEMP");
      if (!temp_dir)
        temp_dir = std::getenv("TMP");
      if (!temp_dir)
        temp_dir = "C:\\temp";
      cache_path = std::string(temp_dir);
#else
      const char *temp_dir = std::getenv("TMPDIR");
      if (!temp_dir)
        temp_dir = "/tmp";
      cache_path = std::string(temp_dir);
#endif
    }
    cache_directory_ = cache_path + "/dxmt_pso_cache";
  } else {
    cache_directory_ = cache_path;
  }

  // Create cache directory
  if (!CreateCacheDirectory()) {
    Logger::warn(str::format("Failed to create PSO cache directory: ",
                             cache_directory_));
    caching_enabled_ = false;
    return false;
  }

  Logger::info(str::format("PSO configuration cache initialized at: ",
                           cache_directory_));
  return true;
}

std::string PSO_ConfigCache::CheckCacheHit(const PSO_Configuration &config) {
  if (!caching_enabled_) {
    return "";
  }

  auto hash = HashConfiguration(config);
  std::string cache_file = GetCacheFilePath(hash);

  if (std::filesystem::exists(cache_file)) {
    return cache_file;
  }

  return "";
}

bool PSO_ConfigCache::SaveConfiguration(const PSO_Configuration &config) {
  if (!caching_enabled_) {
    return false;
  }

  auto hash = HashConfiguration(config);
  std::string cache_file = GetCacheFilePath(hash);

  // Don't overwrite existing files
  if (std::filesystem::exists(cache_file)) {
    return true;
  }

  // Create cache file structure
  PSO_ConfigFile file_data;
  file_data.magic = CACHE_MAGIC;
  file_data.version = CACHE_VERSION;
  file_data.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
  file_data.config_size = sizeof(PSO_Configuration);
  file_data.config = config;
  file_data.checksum = CalculateCRC32(&config, sizeof(config));

  // Write to temporary file first
  std::string temp_file = cache_file + ".tmp";
  std::ofstream out(temp_file, std::ios::binary);
  if (!out) {
    WARN("Failed to create PSO config temp file: ", temp_file);
    return false;
  }

  out.write(reinterpret_cast<const char *>(&file_data), sizeof(file_data));
  out.close();

  // Atomic rename
  try {
    std::filesystem::rename(temp_file, cache_file);
    TRACE("PSO config saved to cache: ",
          std::filesystem::path(cache_file).filename().string());
    return true;
  } catch (const std::filesystem::filesystem_error &e) {
    WARN("Failed to rename PSO config file: ", e.what());
    std::filesystem::remove(temp_file);
    return false;
  }
}

Sha1Hash PSO_ConfigCache::HashConfiguration(const PSO_Configuration &config) {
  // Hash the entire configuration structure
  return Sha1Hash::compute(reinterpret_cast<const uint8_t *>(&config),
                           sizeof(config));
}

bool PSO_ConfigCache::CreateCacheDirectory() {
  try {
    std::filesystem::create_directories(cache_directory_);
    return true;
  } catch (const std::filesystem::filesystem_error &e) {
    WARN("Failed to create PSO cache directory: ", e.what());
    return false;
  }
}

std::string PSO_ConfigCache::GetCacheFilePath(const Sha1Hash &hash) {
  return cache_directory_ + "/" + hash.toString() + ".pso_config";
}

uint32_t PSO_ConfigCache::CalculateCRC32(const void *data, size_t size) {
  // Simple CRC32 implementation
  uint32_t crc = 0xFFFFFFFF;
  const uint8_t *bytes = static_cast<const uint8_t *>(data);

  for (size_t i = 0; i < size; ++i) {
    crc ^= bytes[i];
    for (int j = 0; j < 8; ++j) {
      if (crc & 1) {
        crc = (crc >> 1) ^ 0xEDB88320;
      } else {
        crc >>= 1;
      }
    }
  }

  return crc ^ 0xFFFFFFFF;
}

MTL_GRAPHICS_PIPELINE_DESC PSO_ConfigCache::ReconstructBasicPipelineDesc(
    const PSO_Configuration &config, dxmt::PipelineCache *pipeline_cache) {
  MTL_GRAPHICS_PIPELINE_DESC desc = {};

  desc.NumColorAttachments = config.num_rtvs;
  memcpy(desc.ColorAttachmentFormats, config.rtv_formats,
         sizeof(config.rtv_formats));
  desc.DepthStencilFormat = config.depth_format;
  desc.TopologyClass = config.topology;
  desc.SampleCount = (uint8_t)config.sample_count;
  desc.SampleMask = config.sample_mask;
  desc.RasterizationEnabled = config.rasterization_enabled;
  desc.GSStripTopology = config.gs_strip_topology;
  desc.GSPassthrough = config.gs_passthrough;
  desc.IndexBufferFormat = (SM50_INDEX_BUFFER_FORAMT)config.index_buffer_format;

  IMTLD3D11PipelineCache *cache_interface =
      reinterpret_cast<IMTLD3D11PipelineCache *>(pipeline_cache);

  desc.VertexShader =
      (config.vertex_shader_hash != Sha1Hash())
          ? cache_interface->LookupShaderByHash(config.vertex_shader_hash)
          : nullptr;
  desc.PixelShader =
      (config.pixel_shader_hash != Sha1Hash())
          ? cache_interface->LookupShaderByHash(config.pixel_shader_hash)
          : nullptr;
  desc.HullShader =
      (config.hull_shader_hash != Sha1Hash())
          ? cache_interface->LookupShaderByHash(config.hull_shader_hash)
          : nullptr;
  desc.DomainShader =
      (config.domain_shader_hash != Sha1Hash())
          ? cache_interface->LookupShaderByHash(config.domain_shader_hash)
          : nullptr;
  desc.GeometryShader =
      (config.geometry_shader_hash != Sha1Hash())
          ? cache_interface->LookupShaderByHash(config.geometry_shader_hash)
          : nullptr;

  IMTLD3D11BlendState *blend_state = nullptr;
  cache_interface->AddBlendState(&kDefaultBlendDesc, &blend_state);

  desc.BlendState = blend_state;
  desc.InputLayout = ReconstructInputLayout(config, pipeline_cache);
  desc.SOLayout = ReconstructStreamOutputLayout(config, pipeline_cache);

  return desc;
}

bool PSO_ConfigCache::PrewarmPSOConfigurations(
    dxmt::PipelineCache *pipeline_cache) {
  if (!caching_enabled_ || !pipeline_cache) {
    return false;
  }

  Logger::info(
      str::format("[DXMT] Starting PSO configuration pre-warming from: ",
                  cache_directory_));

  int total_files = 0;
  int successful_prewarmed = 0;

  try {
    // Scan cache directory for .pso_config files
    for (const auto &entry :
         std::filesystem::directory_iterator(cache_directory_)) {
      if (!entry.is_regular_file())
        continue;

      std::string filename = entry.path().filename().string();
      if (filename.find(".pso_config") == std::string::npos)
        continue;

      total_files++;

      // Load and validate PSO config file
      std::ifstream file(entry.path(), std::ios::binary);
      if (!file) {
        WARN("Failed to open PSO config file: ", entry.path().string());
        continue;
      }

      PSO_ConfigFile file_data;
      file.read(reinterpret_cast<char *>(&file_data), sizeof(file_data));
      if (!file || file.gcount() != sizeof(file_data)) {
        WARN("Failed to read PSO config file: ", entry.path().string());
        continue;
      }

      // Validate file
      if (file_data.magic != CACHE_MAGIC ||
          file_data.version != CACHE_VERSION) {
        WARN("Invalid PSO config file (wrong magic/version): ",
             entry.path().string());
        continue;
      }

      // Verify checksum
      uint32_t computed_crc =
          CalculateCRC32(&file_data.config, sizeof(file_data.config));
      if (computed_crc != file_data.checksum) {
        WARN("PSO config file checksum mismatch: ", entry.path().string());
        continue;
      }

      // Reconstruct and create pipeline
      if (PrewarmSinglePSOConfiguration(file_data.config, pipeline_cache)) {
        successful_prewarmed++;
      }
    }

    Logger::info(str::format(
        "[DXMT] PSO pre-warming completed. Successfully pre-warmed ",
        successful_prewarmed, "/", total_files, " configurations"));
    return successful_prewarmed > 0;

  } catch (const std::filesystem::filesystem_error &e) {
    WARN("Error scanning PSO cache directory: ", e.what());
    return false;
  }
}

bool PSO_ConfigCache::PrewarmSinglePSOConfiguration(
    const PSO_Configuration &config, dxmt::PipelineCache *pipeline_cache) {
  auto config_hash = HashConfiguration(config);

  try {
    MTL_GRAPHICS_PIPELINE_DESC pipeline_desc =
        ReconstructBasicPipelineDesc(config, pipeline_cache);

    if (!pipeline_desc.VertexShader &&
        config.vertex_shader_hash != Sha1Hash()) {
      WARN("Missing vertex shader for PSO: ",
           config_hash.toString().substr(0, 8),
           " VS hash=", config.vertex_shader_hash.toString().substr(0, 8));
      return false;
    }

    if (!pipeline_desc.PixelShader && config.pixel_shader_hash != Sha1Hash()) {
      WARN(
          "Missing pixel shader for PSO: ", config_hash.toString().substr(0, 8),
          " PS hash=", config.pixel_shader_hash.toString().substr(0, 8),
          " (shader lookup failed - may not be cached yet)");
    }

    IMTLD3D11PipelineCache *cache_interface =
        reinterpret_cast<IMTLD3D11PipelineCache *>(pipeline_cache);
    IMTLCompiledGraphicsPipeline *compiled_pipeline = nullptr;

    cache_interface->GetGraphicsPipeline(&pipeline_desc, &compiled_pipeline);

    if (compiled_pipeline) {
      compiled_pipeline->Release();
      return true;
    } else {
      return false;
    }

  } catch (const std::exception &e) {
    WARN("Exception during PSO reconstruction: ", e.what());
    return false;
  }
}

ManagedInputLayout
PSO_ConfigCache::ReconstructInputLayout(const PSO_Configuration &config,
                                        dxmt::PipelineCache *pipeline_cache) {
  // Check if input layout data exists
  if (config.input_layout_size == 0 ||
      config.input_layout_size > sizeof(config.input_layout_data)) {
    return nullptr;
  }

  // Calculate number of elements
  const size_t element_size = sizeof(MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC);
  if (config.input_layout_size % element_size != 0) {
    WARN("Invalid input layout data size: ", config.input_layout_size,
         " (not aligned to element size ", element_size, ")");
    WARN("This suggests cached data was created with a different structure "
         "layout or is corrupted");
    return nullptr;
  }

  uint32_t num_elements =
      config.input_layout_size / sizeof(MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC);
  if (num_elements == 0) {
    return nullptr;
  }

  // Deserialize MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC array from cached data
  const MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC *elements =
      reinterpret_cast<const MTL_SHADER_INPUT_LAYOUT_ELEMENT_DESC *>(
          config.input_layout_data);

  // Create MTL_INPUT_LAYOUT_DESC
  MTL_INPUT_LAYOUT_DESC layout_desc(elements, elements + num_elements);

  // Get interface to pipeline cache
  IMTLD3D11PipelineCache *cache_interface =
      reinterpret_cast<IMTLD3D11PipelineCache *>(pipeline_cache);

  // Use the new direct method to create input layout from MTL structures
  IMTLD3D11InputLayout *input_layout = nullptr;
  HRESULT hr =
      cache_interface->AddInputLayoutDirect(layout_desc, &input_layout);

  if (FAILED(hr) || !input_layout) {
    WARN("Failed to reconstruct input layout from cached data, HRESULT: ", hr);
    return nullptr;
  }

  TRACE("Successfully reconstructed input layout from cached data (",
        num_elements, " elements)");
  return input_layout->GetManagedInputLayout();
}

IMTLD3D11StreamOutputLayout *PSO_ConfigCache::ReconstructStreamOutputLayout(
    const PSO_Configuration &config, dxmt::PipelineCache *pipeline_cache) {
  // Check if stream output data exists
  if (config.stream_output_size == 0 ||
      config.stream_output_size > sizeof(config.stream_output_data)) {
    return nullptr;
  }

  // The stream output data should contain:
  // - 4 uint32_t strides (16 bytes)
  // - followed by MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC elements
  if (config.stream_output_size < 16) { // Need at least space for strides
    WARN("Invalid stream output data size: ", config.stream_output_size,
         " (too small for strides)");
    return nullptr;
  }

  const uint8_t *data = config.stream_output_data;

  // Extract strides (first 16 bytes)
  const uint32_t *strides = reinterpret_cast<const uint32_t *>(data);
  data += 16;
  uint32_t remaining_size = config.stream_output_size - 16;

  // Calculate number of elements
  if (remaining_size % sizeof(MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC) != 0) {
    WARN("Invalid stream output element data size: ", remaining_size,
         " (not aligned to element size)");
    return nullptr;
  }

  uint32_t num_elements =
      remaining_size / sizeof(MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC);

  // Deserialize MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC array
  const MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC *elements =
      reinterpret_cast<const MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC *>(data);

  // Create MTL_STREAM_OUTPUT_DESC
  MTL_STREAM_OUTPUT_DESC so_desc;
  so_desc.Strides[0] = strides[0];
  so_desc.Strides[1] = strides[1];
  so_desc.Strides[2] = strides[2];
  so_desc.Strides[3] = strides[3];
  so_desc.Elements = std::vector<MTL_SHADER_STREAM_OUTPUT_ELEMENT_DESC>(
      elements, elements + num_elements);

  // Get interface to pipeline cache
  IMTLD3D11PipelineCache *cache_interface =
      reinterpret_cast<IMTLD3D11PipelineCache *>(pipeline_cache);

  // Use the new direct method to create stream output layout from MTL
  // structures
  IMTLD3D11StreamOutputLayout *so_layout = nullptr;
  HRESULT hr =
      cache_interface->AddStreamOutputLayoutDirect(so_desc, &so_layout);

  if (FAILED(hr) || !so_layout) {
    WARN("Failed to reconstruct stream output layout from cached data, "
         "HRESULT: ",
         hr);
    return nullptr;
  }

  TRACE("Successfully reconstructed stream output layout from cached data (",
        num_elements, " elements)");
  return so_layout;
}

} // namespace dxmt