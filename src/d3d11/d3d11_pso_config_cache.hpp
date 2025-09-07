#pragma once

#include "../util/sha1/sha1_util.hpp"
#include "d3d11_shader.hpp"
#include "d3d11_device_child.hpp"
#include "d3d11_state_object.hpp"
#include "airconv_public.h"
#include <cstdint>
#include <string>

// Forward declarations
struct IMTLD3D11PipelineCache;
struct MTL_GRAPHICS_PIPELINE_DESC;

namespace dxmt {
class PipelineCache;
}

namespace dxmt {

#pragma pack(push, 1)

// Deterministic shader variant serialization (instead of std::variant)
struct SerializedVertexVariant {
  uint8_t variant_type; // 0=Default, 1=Vertex, 2=StreamOutput, etc.
  uint64_t input_layout_handle;
  uint32_t gs_passthrough;
  bool rasterization_disabled;
  uint64_t stream_output_layout_handle; // Only used for StreamOutput variant
  uint8_t padding[3];                   // Ensure consistent alignment
};

struct SerializedPixelVariant {
  uint8_t variant_type; // 0=Default, 1=Pixel, etc.
  uint32_t sample_mask;
  bool dual_source_blending;
  bool disable_depth_output;
  uint32_t unorm_output_reg_mask;
  uint8_t padding[3]; // Ensure consistent alignment
};

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
};

struct PSO_Configuration {
  // Shader identifiers (including all shader stages)
  Sha1Hash vertex_shader_hash;
  Sha1Hash pixel_shader_hash;
  Sha1Hash hull_shader_hash;     // NEW: Tessellation hull shader
  Sha1Hash domain_shader_hash;   // NEW: Tessellation domain shader
  Sha1Hash geometry_shader_hash; // NEW: Geometry shader

  // Shader variants (deterministically serialized)
  SerializedVertexVariant vertex_variant;
  SerializedPixelVariant pixel_variant;

  // Render target configuration
  WMTPixelFormat rtv_formats[8];
  uint8_t num_rtvs;
  WMTPixelFormat depth_format;
  WMTPixelFormat stencil_format;

  // Pipeline configuration
  WMTPrimitiveTopologyClass topology;
  uint32_t sample_count;
  uint32_t sample_mask; // NEW: Sample mask (different from sample_count)
  bool rasterization_enabled;
  bool gs_strip_topology;       // NEW: Geometry shader strip topology
  uint32_t gs_passthrough;      // NEW: Geometry shader passthrough flag
  uint32_t index_buffer_format; // NEW: Index buffer format
                                // (SM50_INDEX_BUFFER_FORMAT)

  // DeterministicBlendState blend_state; // NEW: Blend state data

  uint8_t input_layout_data[1024];
  uint32_t input_layout_size;
  uint8_t stream_output_data[1024]; // NEW: Stream output layout data
  uint32_t stream_output_size;      // NEW: Stream output layout size

  // Padding for alignment
  uint8_t padding[3];
};
#pragma pack(pop)

struct PSO_ConfigFile {
  uint32_t magic;   // 'PSOC' = 0x43434F5053
  uint32_t version; // 1
  uint64_t timestamp;
  uint32_t config_size; // sizeof(PSO_Configuration)
  uint32_t checksum;    // CRC32 of config data
  PSO_Configuration config;
};

class PSO_ConfigCache {
public:
  PSO_ConfigCache();
  ~PSO_ConfigCache();

  // Initialize cache directory
  bool Initialize();

  // Check if configuration exists in cache (returns filename if found)
  std::string CheckCacheHit(const PSO_Configuration &config);

  // Save configuration to cache
  bool SaveConfiguration(const PSO_Configuration &config);

  // Generate deterministic hash from configuration
  Sha1Hash HashConfiguration(const PSO_Configuration &config);

  // Get cache directory path
  const std::string &GetCacheDirectory() const { return cache_directory_; }

  // Check if caching is enabled
  bool IsCachingEnabled() const { return caching_enabled_; }

  // Pre-warm PSO configurations by loading and creating pipelines from cached
  // configs
  bool PrewarmPSOConfigurations(dxmt::PipelineCache *pipeline_cache);

private:
  std::string cache_directory_;
  bool caching_enabled_;

  // Helper methods
  bool CreateCacheDirectory();
  std::string GetCacheFilePath(const Sha1Hash &hash);
  uint32_t CalculateCRC32(const void *data, size_t size);
  bool PrewarmSinglePSOConfiguration(const PSO_Configuration &config,
                                     dxmt::PipelineCache *pipeline_cache);
  MTL_GRAPHICS_PIPELINE_DESC
  ReconstructBasicPipelineDesc(const PSO_Configuration &config,
                               dxmt::PipelineCache *pipeline_cache);

  ManagedInputLayout
  ReconstructInputLayout(const PSO_Configuration &config,
                         dxmt::PipelineCache *pipeline_cache);
  IMTLD3D11StreamOutputLayout *
  ReconstructStreamOutputLayout(const PSO_Configuration &config,
                                dxmt::PipelineCache *pipeline_cache);

  static constexpr uint32_t CACHE_MAGIC = 0x434F5350; // 'PSOC'
  static constexpr uint32_t CACHE_VERSION = 1;
};

} // namespace dxmt