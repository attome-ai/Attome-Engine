#include "Culling.h"

#include "../voxel/MeshTypes.h"

#include <algorithm>
#include <cmath>

namespace atm::render {

using voxel::kChunkSize;
using voxel::kFaceNormal;

Frustum Frustum::fromViewProj(const glm::mat4 &m) {
  auto row = [&](int i) { return glm::vec4(m[0][i], m[1][i], m[2][i], m[3][i]); };
  const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  Frustum f{};
  f.planes[0] = r3 + r0; // left
  f.planes[1] = r3 - r0; // right
  f.planes[2] = r3 + r1; // bottom (y flipped in clip space: still a side plane)
  f.planes[3] = r3 - r1; // top
  f.planes[4] = r3 - r2; // near (reverse-Z: z_clip <= w); far is at infinity
  for (glm::vec4 &p : f.planes) {
    const float len = glm::length(glm::vec3(p));
    if (len > 0.0f) p /= len;
  }
  return f;
}

bool Frustum::aabbVisible(const glm::vec3 &mn, const glm::vec3 &mx) const {
  for (const glm::vec4 &p : planes) {
    const glm::vec3 v(p.x > 0.0f ? mx.x : mn.x, p.y > 0.0f ? mx.y : mn.y,
                      p.z > 0.0f ? mx.z : mn.z);
    if (p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0.0f) return false;
  }
  return true;
}

void ChunkCuller::run(std::span<const CullChunk> chunks, const Params &params,
                      const Frustum &frustum, std::vector<uint32_t> &outSlots) {
  outSlots.clear();
  visited_ = 0;

  const int radius = int(std::ceil(params.viewDistance / float(kChunkSize))) + 1;
  if (radius != radius_) {
    radius_ = radius;
    dimXZ_ = 2 * radius + 1;
    yMin_ = 0;
    yCount_ = voxel::kWorldChunksY;
    const size_t cells = size_t(dimXZ_) * size_t(dimXZ_) * size_t(yCount_);
    grid_.assign(cells, 0u);
    visitedMask_.assign(cells, 0u);
    queue_.clear();
    queue_.reserve(cells);
  } else {
    std::fill(grid_.begin(), grid_.end(), 0u);
    std::fill(visitedMask_.begin(), visitedMask_.end(), uint8_t(0));
    queue_.clear();
  }

  const glm::ivec3 camChunk(params.camBlock.x >> voxel::kChunkShift,
                            params.camBlock.y >> voxel::kChunkShift,
                            params.camBlock.z >> voxel::kChunkShift);
  auto cellIndex = [&](int gx, int gy, int gz) {
    return (size_t(gy) * size_t(dimXZ_) + size_t(gz)) * size_t(dimXZ_) + size_t(gx);
  };
  auto inGrid = [&](int gx, int gy, int gz) {
    return gx >= 0 && gz >= 0 && gy >= 0 && gx < dimXZ_ && gz < dimXZ_ && gy < yCount_;
  };

  for (size_t i = 0; i < chunks.size(); ++i) {
    const voxel::ChunkCoord c = chunks[i].coord;
    const int gx = c.x - camChunk.x + radius_, gz = c.z - camChunk.z + radius_;
    const int gy = c.y - yMin_;
    if (inGrid(gx, gy, gz)) grid_[cellIndex(gx, gy, gz)] = uint32_t(i + 1);
  }

  // Camera-relative box of a cell (full chunk or content bounds).
  const glm::vec3 camFrac = params.camFrac;
  auto relOrigin = [&](int gx, int gy, int gz) {
    const glm::ivec3 origin((camChunk.x + gx - radius_) << voxel::kChunkShift,
                            (gy + yMin_) << voxel::kChunkShift,
                            (camChunk.z + gz - radius_) << voxel::kChunkShift);
    return glm::vec3(origin - params.camBlock) - camFrac;
  };
  const float maxDist2 = params.viewDistance * params.viewDistance;
  auto withinDistance = [&](const glm::vec3 &mn) {
    // Horizontal distance from the camera to the chunk column box.
    const float dx = std::max({mn.x, 0.0f, -(mn.x + kChunkSize)});
    const float dz = std::max({mn.z, 0.0f, -(mn.z + kChunkSize)});
    return dx * dx + dz * dz <= maxDist2;
  };

  // Start cell (clamped into the world's vertical range).
  const int sgx = radius_, sgz = radius_;
  const int sgy = std::clamp(camChunk.y - yMin_, 0, yCount_ - 1);
  const bool startInside = camChunk.y - yMin_ == sgy;

  auto emit = [&](uint32_t chunkPlusOne, const glm::vec3 &rel) {
    if (chunkPlusOne == 0) return;
    const CullChunk &ch = chunks[chunkPlusOne - 1];
    if (!ch.drawable) return;
    const glm::vec3 mn = rel + glm::vec3(ch.aabbMin[0], ch.aabbMin[1], ch.aabbMin[2]);
    const glm::vec3 mx = rel + glm::vec3(ch.aabbMax[0], ch.aabbMax[1], ch.aabbMax[2]);
    if (frustum.aabbVisible(mn, mx)) outSlots.push_back(ch.slot);
  };

  if (!params.caveCulling || !startInside) {
    // Plain frustum + distance culling over all resident chunks, sorted front
    // to back is skipped: callers accept BFS-less order in this mode.
    for (size_t i = 0; i < chunks.size(); ++i) {
      const voxel::ChunkCoord c = chunks[i].coord;
      const int gx = c.x - camChunk.x + radius_, gz = c.z - camChunk.z + radius_;
      const int gy = c.y - yMin_;
      if (!inGrid(gx, gy, gz)) continue;
      const glm::vec3 rel = relOrigin(gx, gy, gz);
      if (!withinDistance(rel)) continue;
      ++visited_;
      emit(uint32_t(i + 1), rel);
    }
    return;
  }

  visitedMask_[cellIndex(sgx, sgy, sgz)] = 1;
  queue_.push_back({sgx, sgy, sgz, 0xFF, 0});
  for (size_t head = 0; head < queue_.size(); ++head) {
    const Node n = queue_[head];
    ++visited_;
    const uint32_t chunkPlusOne = grid_[cellIndex(n.gx, n.gy, n.gz)];
    const glm::vec3 rel = relOrigin(n.gx, n.gy, n.gz);
    emit(chunkPlusOne, rel);

    const uint64_t conn = chunkPlusOne ? chunks[chunkPlusOne - 1].connectivity : ~0ull;
    for (int g = 0; g < voxel::kFaceDirCount; ++g) {
      if (n.usedDirs & (1u << (g ^ 1))) continue;          // never turn back
      if (n.entryFace != 0xFF) {
        if (n.entryFace == g) continue;
        const int bit = voxel::ChunkMeshData::connectivityBit(n.entryFace, g);
        if (!((conn >> bit) & 1u)) continue;               // sealed off
      }
      const int nx = n.gx + kFaceNormal[g][0], ny = n.gy + kFaceNormal[g][1],
                nz = n.gz + kFaceNormal[g][2];
      if (!inGrid(nx, ny, nz)) continue;
      const size_t idx = cellIndex(nx, ny, nz);
      if (visitedMask_[idx]) continue;
      const glm::vec3 nrel = relOrigin(nx, ny, nz);
      if (!withinDistance(nrel)) continue;
      if (!frustum.aabbVisible(nrel, nrel + glm::vec3(float(kChunkSize)))) continue;
      visitedMask_[idx] = 1;
      queue_.push_back({nx, ny, nz, uint8_t(g ^ 1), uint8_t(n.usedDirs | (1u << g))});
    }
  }
}

} // namespace atm::render
