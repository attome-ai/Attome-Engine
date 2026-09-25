#pragma once

// SPIR-V embedded at build time (CMake: glslangValidator + EmbedSpirv.cmake).

#include <cstddef>
#include <cstdint>

namespace atm::render::vk {

enum class ShaderId {
  VoxelVert,
  VoxelFrag,
  ModelVert,
  CullComp,
  FullscreenVert,
  SkyFrag,
  HighlightVert,
  HighlightFrag,
  BloomComp,
  TonemapFrag,
  ShadowVert,
  ShadowModelVert,
  SsaoComp,
  Count
};

struct SpirvBlob {
  const uint32_t *code;
  size_t sizeBytes;
};

SpirvBlob shaderSpirv(ShaderId id);

} // namespace atm::render::vk
