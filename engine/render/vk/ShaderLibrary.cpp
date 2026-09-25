#include "ShaderLibrary.h"

// Generated into ${CMAKE_CURRENT_BINARY_DIR}/shaders_gen by EmbedSpirv.cmake.
#include "voxel_vert.spv.h"
#include "voxel_frag.spv.h"
#include "model_vert.spv.h"
#include "cull_comp.spv.h"
#include "fullscreen_vert.spv.h"
#include "sky_frag.spv.h"
#include "highlight_vert.spv.h"
#include "highlight_frag.spv.h"
#include "bloom_comp.spv.h"
#include "tonemap_frag.spv.h"
#include "shadow_vert.spv.h"
#include "shadow_model_vert.spv.h"

namespace atm::render::vk {

SpirvBlob shaderSpirv(ShaderId id) {
  switch (id) {
  case ShaderId::VoxelVert: return {kSpv_voxel_vert, kSpv_voxel_vert_size};
  case ShaderId::VoxelFrag: return {kSpv_voxel_frag, kSpv_voxel_frag_size};
  case ShaderId::ModelVert: return {kSpv_model_vert, kSpv_model_vert_size};
  case ShaderId::CullComp: return {kSpv_cull_comp, kSpv_cull_comp_size};
  case ShaderId::FullscreenVert: return {kSpv_fullscreen_vert, kSpv_fullscreen_vert_size};
  case ShaderId::SkyFrag: return {kSpv_sky_frag, kSpv_sky_frag_size};
  case ShaderId::HighlightVert: return {kSpv_highlight_vert, kSpv_highlight_vert_size};
  case ShaderId::HighlightFrag: return {kSpv_highlight_frag, kSpv_highlight_frag_size};
  case ShaderId::BloomComp: return {kSpv_bloom_comp, kSpv_bloom_comp_size};
  case ShaderId::TonemapFrag: return {kSpv_tonemap_frag, kSpv_tonemap_frag_size};
  case ShaderId::ShadowVert: return {kSpv_shadow_vert, kSpv_shadow_vert_size};
  case ShaderId::ShadowModelVert: return {kSpv_shadow_model_vert, kSpv_shadow_model_vert_size};
  case ShaderId::Count: break;
  }
  return {nullptr, 0};
}

} // namespace atm::render::vk
