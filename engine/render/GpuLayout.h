#pragma once

// GPU-visible data layout (docs/GPU_3D_PLAN.md §8.1). Must match
// shaders/voxel_common.glsl exactly: std140 for the uniform block, std430 for
// storage buffers, only 4-byte scalars and 16-byte vectors, no implicit
// padding. Bump kGpuLayoutVersion (and the GLSL define) on any change.

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>

namespace atm::render::gpu {

inline constexpr uint32_t kGpuLayoutVersion = 2;

// Descriptor bindings of the scene set (set 0), shared by every 3D pipeline.
enum SceneBinding : uint32_t {
  kBindFrame = 0,       // UBO    FrameUniforms (per frame)
  kBindFaces = 1,       // SSBO   uvec2 faces[]    chunk face arena
  kBindChunks = 2,      // SSBO   ChunkGpu[]       per chunk slot
  kBindMaterials = 3,   // SSBO   MaterialGpu[]
  kBindVisible = 4,     // SSBO   uint visible[]   (per frame)
  kBindDraws = 5,       // SSBO   VkDrawIndexedIndirectCommand[] (per frame)
  kBindDrawCount = 6,   // SSBO   uint count       (per frame)
  kBindModelFaces = 7,  // SSBO   uvec2 modelFaces[]
  kBindInstances = 8,   // SSBO   ModelInstanceGpu[] (per frame)
  kBindShadowMap = 9,   // sampler2DShadow  sun shadow map (depth compare)
  kBindShadowDepth = 10, // sampler2D  same map, raw depth (PCSS blocker search)
  kSceneBindingCount = 11
};

struct FrameUniforms {           // std140, 384 bytes
  glm::mat4 viewProj;            // camera-relative (no translation)
  glm::mat4 invViewProj;
  glm::ivec4 camBlock;           // floor(camera position)
  glm::vec4 camFrac;             // camera position - camBlock (xyz), w = time (s)
  glm::vec4 sunDir;              // xyz normalised, w = ambient
  glm::vec4 skyColor;            // rgb, w = timeOfDay
  glm::vec4 fogColor;            // rgb, w = fog start (blocks)
  glm::vec4 fogParams;           // x = fog end, y = 1/(end-start), z = sun intensity, w = unused
  glm::vec4 sunColor;            // rgb, w = unused
  glm::uvec4 counts;             // x = visible chunks, y = max draws, z = unused, w = unused
  glm::mat4 lightViewProj;       // camera-relative position -> shadow map clip space
  glm::vec4 shadowParams;        // x = texel size (blocks), y = strength (0 = off), z = depth range (blocks), w = unused
  glm::vec4 style0;              // x = sun strength, y = haze strength, z = haze density, w = shadow softness
  glm::vec4 style1;              // x = AO darkness, y = bevel, z = grain, w = block variation
  glm::vec4 style2;              // x = colour patches, y = water reflection, z = foliage glow, w = rim light
};
static_assert(sizeof(FrameUniforms) == 384);
static_assert(offsetof(FrameUniforms, lightViewProj) == 256);
static_assert(offsetof(FrameUniforms, camBlock) == 128);
static_assert(offsetof(FrameUniforms, counts) == 240);

// Per resident chunk slot. Faces of the chunk live at
// arena[faceBase + dirOffset[d] .. faceBase + dirOffset[d+1]) for d in 0..5,
// translucent faces at arena[faceBase + dirOffset[6] .. + translucentCount).
struct ChunkGpu {                // std430, 64 bytes
  int32_t originX, originY, originZ; // chunk origin in blocks
  uint32_t flags;                    // bit 0 = valid
  uint32_t faceBase;                 // absolute face index in the arena
  uint32_t translucentCount;
  uint32_t dirOffset[7];             // relative to faceBase; [6] = opaque total
  uint32_t aabbMin;                  // x | y << 8 | z << 16   (local 0..32)
  uint32_t aabbMax;                  // x | y << 8 | z << 16
  uint32_t pad0;
};
static_assert(sizeof(ChunkGpu) == 64);
static_assert(offsetof(ChunkGpu, faceBase) == 16);
static_assert(offsetof(ChunkGpu, dirOffset) == 24);
static_assert(offsetof(ChunkGpu, aabbMin) == 52);

struct MaterialGpu {             // std430, 32 bytes
  uint32_t top, side, bottom;    // bytes R,G,B,A (0xAABBGGRR)
  float emissive;
  float alpha;
  uint32_t flags;                // kMaterialWater | kMaterialFoliage
  uint32_t pad1, pad2;
};
static_assert(sizeof(MaterialGpu) == 32);

struct ModelInstanceGpu {        // std430, 80 bytes
  glm::mat4 model;               // part voxel space -> camera-relative blocks
  uint32_t tint;                 // bytes R,G,B,A (0xAABBGGRR)
  uint32_t paletteOffset;        // added to face material ids
  uint32_t pad0, pad1;
};
static_assert(sizeof(ModelInstanceGpu) == 80);
static_assert(offsetof(ModelInstanceGpu, tint) == 64);

struct DrawIndexedIndirect {     // == VkDrawIndexedIndirectCommand
  uint32_t indexCount;
  uint32_t instanceCount;
  uint32_t firstIndex;
  int32_t vertexOffset;
  uint32_t firstInstance;
};
static_assert(sizeof(DrawIndexedIndirect) == 20);

// Push constants.
struct HighlightPush {           // line box around a block
  glm::vec4 minCorner;           // camera-relative
  glm::vec4 maxCorner;
  glm::vec4 color;
};
static_assert(sizeof(HighlightPush) == 48);

struct BloomPush {
  glm::vec2 srcTexel;            // 1 / source size
  float threshold;               // prefilter (first downsample) only
  float intensity;               // upsample blend weight
  uint32_t mode;                 // 0 = prefilter downsample, 1 = downsample, 2 = upsample
  uint32_t pad0, pad1, pad2;
};
static_assert(sizeof(BloomPush) == 32);

struct TonemapPush {
  float exposure;
  float bloomStrength;
  uint32_t bloomEnabled;
  uint32_t srgbOutput;           // 1 = swapchain applies sRGB itself
  glm::vec2 sunUv;               // sun on screen (0..1, may be outside)
  float shaftStrength;           // sun shafts, 0 = off
  float aoStrength;              // SSAO, 0 = off
  glm::vec4 sunColor;            // linear rgb
  float contrast;                // S-curve amount
  float vibrance;                // muted-colour boost
  float vignette;                // corner darkening
  float pad0;
};
static_assert(sizeof(TonemapPush) == 64);

struct SsaoPush {
  glm::mat4 viewProj;            // camera-relative
  glm::mat4 invViewProj;
};
static_assert(sizeof(SsaoPush) == 128);

} // namespace atm::render::gpu
