#pragma once

// Flood-fill lighting (sky + block light) for one chunk, computed on worker
// threads from an immutable snapshot of the chunk and its 26 neighbours.
//
// The chunk is lit inside a 64^3 working region: the chunk (32^3) plus a
// 15-block margin gathered from its neighbours, plus a 1-cell opaque wall so
// the BFS needs no bounds checks. Light therefore crosses chunk borders
// correctly for anything within 15 blocks (the maximum light range).
//
// Approximations (TODO(demo)): above the region top (15 blocks above the
// chunk) the sky is assumed open; missing (not loaded) neighbours are
// treated as air lit by the sky; chunks below the world (cy < 0) as bedrock.
// Sky light passes straight down through air at 15, loses 1 per block through
// liquids/cutout blocks, and loses 1 per step when spreading sideways.
//
// Deterministic and thread-safe: all mutable state lives in Scratch (one per
// worker thread).

#include "Chunk.h"
#include "VoxelTypes.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace atm::voxel {

class BlockRegistry;

// 3x3x3 chunks around `center`; index = (dy+1)*9 + (dz+1)*3 + (dx+1).
// nullptr = not loaded (see approximations above).
struct ChunkNeighbourhood {
  ChunkCoord center{};
  std::array<const Chunk *, 27> chunks{};

  static constexpr int index(int dx, int dy, int dz) { return (dy + 1) * 9 + (dz + 1) * 3 + (dx + 1); }
  const Chunk *at(int dx, int dy, int dz) const { return chunks[size_t(index(dx, dy, dz))]; }
};

class ChunkLighting {
public:
  static constexpr int kMargin = 15;
  static constexpr int kRegion = 64; // 1 wall + 15 margin + 32 + 15 margin + 1 wall
  static constexpr int kOffset = kMargin + 1; // region coordinate of local 0
  static constexpr int kRegionVolume = kRegion * kRegion * kRegion;

  struct Scratch; // per-thread working memory, reused between calls
  static std::unique_ptr<Scratch> makeScratch();

  // Fills out.blocks (34^3 incl. 1-block neighbour border), out.light and
  // out.coord for the neighbourhood's centre chunk.
  static void buildMeshInput(const ChunkNeighbourhood &hood, const BlockRegistry &registry,
                             Scratch &scratch, MeshInput &out);
};

// Defined here (not hidden) so unique_ptr<Scratch> can be destroyed anywhere.
struct ChunkLighting::Scratch {
  std::vector<BlockId> ids = std::vector<BlockId>(kRegionVolume);
  std::vector<uint8_t> info = std::vector<uint8_t>(kRegionVolume); // pass | atten | emission << 4
  std::vector<uint8_t> sky = std::vector<uint8_t>(kRegionVolume);
  std::vector<uint8_t> blk = std::vector<uint8_t>(kRegionVolume);
  std::vector<BlockId> chunkIds = std::vector<BlockId>(kChunkVolume);
  std::vector<uint8_t> table;  // per BlockId info
  std::vector<uint32_t> queue; // FIFO (reused)
};

} // namespace atm::voxel
