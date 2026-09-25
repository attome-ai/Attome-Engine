#pragma once

// CPU visibility pre-pass (pure CPU, no Vulkan, unit-testable):
//   1. Cave culling: breadth-first search over chunks starting at the camera
//      chunk (Checchi / Minecraft visibility graph). A chunk is entered through
//      one face and may be left through face g only when the mesher marked
//      (entry, g) as connected, and g never points back toward the camera
//      (the path may not use a direction opposite to one it already used).
//      Chunks with no mesh (air, not loaded) are treated as fully open.
//   2. Frustum test (camera-relative planes) + view distance on every visited
//      chunk; the search does not expand out of chunks outside the frustum.
// Output is in BFS order = roughly front to back (good for early-Z).

#include "../voxel/VoxelTypes.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace atm::render {

struct CullChunk {
  voxel::ChunkCoord coord;
  uint32_t slot = 0;               // renderer chunk slot (GPU metadata index)
  uint64_t connectivity = ~0ull;   // ChunkMeshData::faceConnectivity
  uint8_t aabbMin[3]{0, 0, 0};     // local content bounds 0..32
  uint8_t aabbMax[3]{32, 32, 32};
  bool drawable = false;           // has uploaded faces
};

struct Frustum {
  glm::vec4 planes[5]; // left, right, bottom, top, near; inside when dot(n,p)+d >= 0
  // Extracts planes from a camera-relative Vulkan clip matrix (reverse-Z).
  static Frustum fromViewProj(const glm::mat4 &viewProj);
  bool aabbVisible(const glm::vec3 &mn, const glm::vec3 &mx) const;
};

class ChunkCuller {
public:
  struct Params {
    glm::ivec3 camBlock{0};        // floor(camera position)
    glm::vec3 camFrac{0.0f};       // camera - camBlock
    float viewDistance = 384.0f;   // blocks
    bool caveCulling = true;
  };

  // `chunks` = every resident chunk. Writes visible drawable slots into
  // `outSlots` (cleared first). Reuses internal buffers: no allocations once
  // warmed up for a given view distance.
  void run(std::span<const CullChunk> chunks, const Params &params,
           const Frustum &frustum, std::vector<uint32_t> &outSlots);

  uint32_t visitedLastRun() const { return visited_; }

private:
  struct Node { int32_t gx, gy, gz; uint8_t entryFace; uint8_t usedDirs; };
  int radius_ = -1, dimXZ_ = 0;
  int yMin_ = 0, yCount_ = 0;
  std::vector<uint32_t> grid_;        // chunk index + 1 per cell (0 = none)
  std::vector<uint8_t> visitedMask_;  // per cell
  std::vector<Node> queue_;
  uint32_t visited_ = 0;
};

} // namespace atm::render
