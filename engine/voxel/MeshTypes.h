#pragma once

// GPU mesh format for voxel geometry (chunks and voxel model parts).
//
// One merged quad = one 64-bit PackedFace. The vertex shader expands each
// face into 2 triangles from gl_VertexIndex ("vertex pulling"), reading faces
// from a storage buffer: 8 bytes per quad instead of 4 vertices + 6 indices.
//
// Bit layout (little endian, must match shaders/voxel_common.glsl):
//   [ 0.. 5]  x        6 bits  quad origin in chunk/part space, 0..32
//   [ 6..11]  y        6 bits
//   [12..17]  z        6 bits
//   [18..22]  w - 1    5 bits  quad size along the face's first tangent axis
//   [23..27]  h - 1    5 bits  quad size along the face's second tangent axis
//   [28..30]  dir      3 bits  FaceDir (0..5)
//   [31..38]  ao       8 bits  4 corners x 2 bits (0 = darkest, 3 = none)
//   [39..54]  material 16 bits BlockId / palette index
//   [55..58]  sky      4 bits  sky light 0..15
//   [59..62]  block    4 bits  block light 0..15
//   [63]      reserved
//
// Tangent axes per direction (w along U, h along V):
//   ±X: U = Z, V = Y     ±Y: U = X, V = Z     ±Z: U = X, V = Y
// Corner order for AO: (u0,v0), (u1,v0), (u1,v1), (u0,v1).
//
// The mesher only merges faces whose material, AO and light values are all
// equal, so per-quad attributes are exact.

#include "VoxelTypes.h"

#include <array>
#include <cstdint>
#include <vector>

namespace atm::voxel {

using PackedFace = uint64_t;
inline constexpr uint32_t kGpuFaceLayoutVersion = 1;

struct FaceFields {
  uint32_t x, y, z;    // 0..32
  uint32_t w, h;       // 1..32
  FaceDir dir;
  uint8_t ao;          // 4 x 2 bits
  uint16_t material;
  uint8_t sky, block;  // 0..15
};

inline constexpr PackedFace packFace(const FaceFields &f) {
  return uint64_t(f.x & 63u) | (uint64_t(f.y & 63u) << 6) |
         (uint64_t(f.z & 63u) << 12) | (uint64_t((f.w - 1) & 31u) << 18) |
         (uint64_t((f.h - 1) & 31u) << 23) | (uint64_t(uint8_t(f.dir) & 7u) << 28) |
         (uint64_t(f.ao) << 31) | (uint64_t(f.material) << 39) |
         (uint64_t(f.sky & 15u) << 55) | (uint64_t(f.block & 15u) << 59);
}

inline constexpr FaceFields unpackFace(PackedFace p) {
  FaceFields f{};
  f.x = uint32_t(p & 63u);
  f.y = uint32_t((p >> 6) & 63u);
  f.z = uint32_t((p >> 12) & 63u);
  f.w = uint32_t((p >> 18) & 31u) + 1;
  f.h = uint32_t((p >> 23) & 31u) + 1;
  f.dir = FaceDir((p >> 28) & 7u);
  f.ao = uint8_t((p >> 31) & 0xFFu);
  f.material = uint16_t((p >> 39) & 0xFFFFu);
  f.sky = uint8_t((p >> 55) & 15u);
  f.block = uint8_t((p >> 59) & 15u);
  return f;
}

// Mesh of one chunk (or one model part), faces grouped by direction so the
// renderer can skip whole direction groups that face away from the camera.
struct ChunkMeshData {
  // faces[dirOffset[d] .. dirOffset[d+1]) have direction d. Opaque first.
  std::vector<PackedFace> opaque;
  std::array<uint32_t, kFaceDirCount + 1> opaqueDirOffset{};
  // Translucent faces (water, glass, leaves with alpha) — drawn after opaque,
  // sorted back-to-front per chunk by the renderer.
  std::vector<PackedFace> translucent;
  // Axis-aligned bounds of non-empty content, local 0..32 (for culling).
  uint8_t minX = 32, minY = 32, minZ = 32, maxX = 0, maxY = 0, maxZ = 0;
  // Visibility graph (cave culling): bit (a*6+b) set when chunk faces a and b
  // (FaceDir) are connected through non-opaque blocks. Only a<b bits are
  // used (15 pairs). All bits set = fully open (e.g. air chunk).
  uint64_t faceConnectivity = ~0ull;

  static constexpr int connectivityBit(int a, int b) {
    return a < b ? a * 6 + b : b * 6 + a;
  }
  bool connects(FaceDir a, FaceDir b) const {
    return (faceConnectivity >> connectivityBit(int(a), int(b))) & 1u;
  }

  bool empty() const { return opaque.empty() && translucent.empty(); }
  void clear() {
    opaque.clear();
    translucent.clear();
    opaqueDirOffset.fill(0);
    minX = minY = minZ = 32;
    maxX = maxY = maxZ = 0;
    faceConnectivity = ~0ull;
  }
};

} // namespace atm::voxel
