#pragma once

// Core voxel-world types shared by client, server, renderer and tools.
// Coordinate contract: docs/GPU_3D_PLAN.md §5.
//
//   Block   int32 x, y, z      one voxel (~1 m); y = height, 0 .. kWorldHeight-1
//   Chunk   int32 cx, cy, cz   32x32x32 blocks
//   Local   0..31 per axis     position inside a chunk
//   Sector  int32 sx, sz       256x256 blocks, full height (ownership/zones)
//   Entity positions: double, in blocks (exact enough for any world size we
//   will reach); the renderer converts to camera-relative float.

#include <cstdint>
#include <functional>

namespace atm::voxel {

inline constexpr int kChunkShift = 5;
inline constexpr int kChunkSize = 1 << kChunkShift; // 32
inline constexpr int kChunkMask = kChunkSize - 1;
inline constexpr int kChunkVolume = kChunkSize * kChunkSize * kChunkSize;
inline constexpr int kWorldHeight = 512;
inline constexpr int kWorldChunksY = kWorldHeight / kChunkSize; // 16
inline constexpr int kSectorSize = 256;
inline constexpr int kSectorChunks = kSectorSize / kChunkSize; // 8

using BlockId = uint16_t;
inline constexpr BlockId kAir = 0;

// Floor division that is correct for negative coordinates.
inline constexpr int32_t floorShift(int32_t v, int shift) { return v >> shift; }
inline constexpr int32_t floorDiv(int32_t v, int32_t d) {
  return (v >= 0) ? (v / d) : -((-v + d - 1) / d);
}

struct BlockPos {
  int32_t x = 0, y = 0, z = 0;
  friend constexpr bool operator==(const BlockPos &a, const BlockPos &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
};

struct ChunkCoord {
  int32_t x = 0, y = 0, z = 0;
  friend constexpr bool operator==(const ChunkCoord &a, const ChunkCoord &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
  // Packs into 64 bits: 26 bits x, 12 bits y, 26 bits z (±33M chunks = ±1B
  // blocks horizontally). Used as hash-map key and on the wire.
  constexpr uint64_t key() const {
    return (uint64_t(uint32_t(x) & 0x3FFFFFFu) << 38) |
           (uint64_t(uint32_t(y) & 0xFFFu) << 26) |
           uint64_t(uint32_t(z) & 0x3FFFFFFu);
  }
  static constexpr ChunkCoord fromKey(uint64_t k) {
    auto sext = [](uint32_t v, int bits) {
      const uint32_t m = 1u << (bits - 1);
      return int32_t((v ^ m) - m);
    };
    return {sext(uint32_t(k >> 38) & 0x3FFFFFFu, 26),
            sext(uint32_t(k >> 26) & 0xFFFu, 12),
            sext(uint32_t(k) & 0x3FFFFFFu, 26)};
  }
};

struct LocalPos {
  uint8_t x = 0, y = 0, z = 0;
};

inline constexpr ChunkCoord chunkOf(BlockPos p) {
  return {floorShift(p.x, kChunkShift), floorShift(p.y, kChunkShift),
          floorShift(p.z, kChunkShift)};
}
inline constexpr LocalPos localOf(BlockPos p) {
  return {uint8_t(p.x & kChunkMask), uint8_t(p.y & kChunkMask),
          uint8_t(p.z & kChunkMask)};
}
inline constexpr BlockPos chunkOrigin(ChunkCoord c) {
  return {c.x << kChunkShift, c.y << kChunkShift, c.z << kChunkShift};
}
inline constexpr int localIndex(int x, int y, int z) {
  // y-major inside a chunk keeps vertical columns contiguous (world gen,
  // light propagation and binary meshing all walk columns).
  return (y << (2 * kChunkShift)) | (z << kChunkShift) | x;
}

// Face directions. Order matters: renderer draws per direction and culls the
// three directions facing away from the camera.
enum class FaceDir : uint8_t { PosX = 0, NegX = 1, PosY = 2, NegY = 3, PosZ = 4, NegZ = 5 };
inline constexpr int kFaceDirCount = 6;
inline constexpr int kFaceNormal[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                          {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};

struct ChunkCoordHash {
  size_t operator()(const ChunkCoord &c) const noexcept {
    uint64_t k = c.key();
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return size_t(k);
  }
};

} // namespace atm::voxel
