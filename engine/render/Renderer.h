#pragma once

// Public API of the Vulkan voxel renderer. Game code only sees this header;
// all Vulkan types stay inside engine/render/vk/ (docs/GPU_3D_PLAN.md §4.3).
//
// Threading: every call is main-thread only. Chunk meshes are built on
// worker threads, then handed over with setChunkMesh() (which copies into the
// renderer's upload queue and returns immediately).

#include "../voxel/MeshTypes.h"
#include "../voxel/VoxelTypes.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>

struct SDL_Window;

namespace atm::render {

struct RendererConfig {
  bool vsync = true;
  int msaaSamples = 4;             // degrades to what the device supports
  bool validation = false;         // Vulkan validation layers (debug builds)
  int framesInFlight = 2;
  uint32_t uploadBudgetBytes = 8u << 20; // chunk upload bytes per frame
  uint32_t chunkArenaBytes = 512u << 20; // GPU memory for chunk faces
  float fovYDegrees = 70.0f;
  float viewDistanceBlocks = 384.0f;     // fog end + far culling
  bool bloom = true;
};

// Right-handed, +Y up. View direction =
//   (-sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch))
// i.e. yaw is a counter-clockwise rotation about +Y seen from above.
struct Camera {
  glm::dvec3 position{0.0};  // world position in blocks (double precision)
  float yaw = 0.0f;          // radians, 0 = looking toward -Z
  float pitch = 0.0f;        // radians, + = up
  float fovYDegrees = 70.0f;
  float nearPlane = 0.05f;
};

struct Environment {
  glm::vec3 sunDirection{0.4f, 0.8f, 0.3f}; // normalised by the renderer
  glm::vec3 skyColor{0.55f, 0.75f, 1.0f};
  glm::vec3 fogColor{0.62f, 0.78f, 1.0f};
  float timeOfDay = 0.35f;                  // 0..1, drives sky tint
  float ambient = 0.35f;
  bool underwater = false;                  // camera inside water: blue fog, sky tint

  // Look tuning (the game's graphics panel). Defaults = the shipped look.
  float exposure = 0.8f;        // tonemap input scale
  float contrast = 0.3f;        // S-curve amount after tonemapping
  float vibrance = 0.12f;       // boosts muted colours
  float bloom = 0.6f;           // bloom add strength
  float vignette = 0.2f;        // corner darkening (0 = off)
  float sunStrength = 1.05f;    // direct sunlight
  float hazeStrength = 0.3f;    // max distance haze
  float hazeDensity = 0.003f;   // haze per block
  float shadowSoftness = 1.0f;  // penumbra width multiplier
  float aoDarkness = 0.8f;      // voxel corner darkening (0 = off)
  float tileBevel = 1.0f;       // block edge bevel strength
  float tileGrain = 1.0f;       // sub-voxel grain strength
  float blockVariation = 1.0f;  // per-block shade variation
  float colorPatches = 1.0f;    // broad painterly colour patches
  float waterReflection = 0.6f; // sky reflection on water
  float foliageGlow = 0.45f;    // sun shining through leaves
  float rimLight = 0.55f;       // character rim light
  bool contactShadows = false;  // true: contact-hardening (PCSS), false: simple soft shadows
};

// One entry per block id: flat colours (Trove style) + emissive strength.
// Colours are packed as memory bytes R, G, B, A (uint32 value 0xAABBGGRR),
// see atm::voxel::packRGBA.
struct Material {
  uint32_t top = 0xFFFFFFFF, side = 0xFFFFFFFF, bottom = 0xFFFFFFFF; // RGBA8
  float emissive = 0.0f;   // > 0 glows and feeds bloom
  float alpha = 1.0f;      // < 1 for translucent blocks
  uint32_t flags = 0;      // kMaterialWater / kMaterialFoliage (shader effects)
};

inline constexpr uint32_t kMaterialWater = 1u;   // animated waves, reflections
inline constexpr uint32_t kMaterialFoliage = 2u; // sways in the wind
inline constexpr uint32_t kMaterialGrassTop = 4u; // top colour drips over the sides
// ModelInstance::flags
inline constexpr uint32_t kInstanceNoRim = 1u;     // no rim light (props, decoration, particles)
inline constexpr uint32_t kInstanceNoShadow = 2u;  // not drawn into the shadow map

// Material ids >= this are reserved for setModelMaterials().
inline constexpr uint32_t kModelMaterialBase = 16384;
inline constexpr uint32_t kMaxMaterials = 65536;

using ModelMeshId = uint32_t;
inline constexpr ModelMeshId kInvalidModelMesh = 0xFFFFFFFFu;

// A model part instance: part mesh in its own voxel space, placed in the
// world. `worldFromPart` maps part voxel coordinates to world blocks; the
// renderer makes it camera-relative internally (double -> float).
struct ModelInstance {
  ModelMeshId mesh = kInvalidModelMesh;
  glm::dvec3 origin{0.0};        // world position (blocks) of the part's pivot
  glm::quat rotation{1, 0, 0, 0};
  glm::vec3 pivot{0.0f};         // pivot inside the part, in voxels
  float voxelScale = 1.0f / 12;  // world blocks per model voxel
  uint32_t tint = 0xFFFFFFFF;    // multiplied colour, bytes R,G,B,A (packRGBA)
  uint16_t paletteOffset = 0;    // material index offset (dye palettes)
  uint32_t flags = 0;            // kInstanceNoRim | kInstanceNoShadow
};

struct FrameStats {
  float cpuMs = 0.0f;
  float gpuMs = -1.0f;           // from timestamp queries (-1 = unavailable)
  uint32_t drawCalls = 0;
  uint32_t chunksVisible = 0, chunksResident = 0;
  uint64_t facesDrawn = 0;
  uint64_t uploadBytes = 0;
  uint64_t chunkArenaUsed = 0, chunkArenaCapacity = 0;
  uint32_t modelInstances = 0;
  float chunkArenaFragmentation = 0.0f; // 0 = one free range .. 1 = scattered
  uint32_t chunkUploadsPending = 0;     // chunk meshes waiting for upload budget
};

class Renderer {
public:
  Renderer();
  ~Renderer();
  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;

  // `window` must be created with SDL_WINDOW_VULKAN.
  bool init(SDL_Window *window, const RendererConfig &config, std::string *error = nullptr);
  void shutdown();
  void onWindowResized();

  // --- world geometry -------------------------------------------------------
  // Block materials: index = BlockId (0 .. min(size, kModelMaterialBase)-1).
  void setMaterials(std::span<const Material> materials);
  // Appends model palette materials (voxel characters, props) after the block
  // range and returns the base index to pass as meshPart()'s materialBase.
  // Returns 0xFFFF when the 65536-entry material table is full.
  uint16_t setModelMaterials(std::span<const Material> materials);
  // Replaces the chunk's mesh (empty mesh = remove). Copies the data.
  void setChunkMesh(voxel::ChunkCoord coord, const voxel::ChunkMeshData &mesh);
  void removeChunk(voxel::ChunkCoord coord);

  // --- models ---------------------------------------------------------------
  // Uploads a model part mesh once (faces in part voxel space, 0..32).
  ModelMeshId createModelMesh(const voxel::ChunkMeshData &mesh);
  void destroyModelMesh(ModelMeshId id);

  // --- frame ----------------------------------------------------------------
  // Returns false when nothing can be drawn (minimised window); skip the frame.
  bool beginFrame(const Camera &camera, const Environment &env);
  void drawModel(const ModelInstance &instance);       // any number per frame
  void drawBlockHighlight(voxel::BlockPos block);      // targeted-block outline
  // Debug wireframe box (world space, drawn on top of everything). Colour is
  // bytes R,G,B,A. Queued for this frame only.
  void drawDebugBox(const glm::dvec3 &min, const glm::dvec3 &max, uint32_t rgba);
  // 2D overlay: ImGui is rendered by the renderer; call ImGui::NewFrame()
  // after beginFrame() and build UI before endFrame().
  void endFrame();

  const FrameStats &stats() const;

  // Saves the next presented frame as a BMP (SDL_SaveBMP; visual checks and
  // bug reports). The file is written one or two frames later.
  void requestScreenshot(const std::string &path);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace atm::render
