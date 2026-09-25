#pragma once

// Palette-compressed chunk storage + the world-level interfaces used by
// physics, meshing, light and networking.
//
// Memory: a chunk stores a palette of the block ids it contains and a packed
// array of palette indices using the fewest bits that fit (0, 1, 2, 4, 8 or
// 16 bits per block). A chunk of pure air or pure stone is ~40 bytes; typical
// surface chunks with 4-12 block types use 2-4 KB instead of 64 KB.

#include "MeshTypes.h"
#include "VoxelTypes.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace atm::voxel {

class BlockRegistry;

class Chunk {
public:
  Chunk();
  explicit Chunk(BlockId fill);

  BlockId get(int x, int y, int z) const;          // local 0..31
  void set(int x, int y, int z, BlockId id);       // grows palette/bit width as needed
  void fill(BlockId id);
  bool isUniform() const { return bits_ == 0; }
  BlockId uniformBlock() const { return palette_.empty() ? kAir : palette_[0]; }

  // Decode all 32768 ids at once (fast path for meshing / light).
  void decodeAll(BlockId *out /* kChunkVolume */) const;
  // Replace contents from a full array (world generation output).
  void encodeAll(const BlockId *in /* kChunkVolume */);

  // Drops palette entries no longer used and shrinks the bit width.
  void compact();

  // Serialisation for network + disk: palette, bit width, packed data, then
  // run-length encoded. Self-describing with a version byte.
  void serialize(std::vector<uint8_t> &out) const;
  bool deserialize(std::span<const uint8_t> in); // false on malformed input (never crashes)

  size_t memoryBytes() const;

  // Edit tracking (set by set(), cleared by the owner after re-meshing/saving).
  uint32_t revision() const { return revision_; }

private:
  uint32_t readIndex(int i) const;
  void writeIndex(int i, uint32_t v);
  void growBits(int newBits);

  std::vector<BlockId> palette_;
  std::vector<uint64_t> data_; // packed indices, bits_ per entry, no entry spans two words
  uint8_t bits_ = 0;
  uint32_t revision_ = 0;
};

// Read-only block access across chunk borders, implemented by the client and
// server worlds. Out-of-range or not-loaded positions return `unloaded`.
class IBlockAccess {
public:
  virtual ~IBlockAccess() = default;
  virtual BlockId blockAt(BlockPos p) const = 0;
  virtual bool isLoaded(ChunkCoord c) const = 0;
};

// A chunk plus a 1-block border of its six neighbours' faces, copied once so
// the mesher runs without touching shared state (worker-thread safe).
struct MeshInput {
  // (34 x 34 x 34) block ids, index = (y+1)*34*34 + (z+1)*34 + (x+1).
  static constexpr int kPad = kChunkSize + 2;
  std::vector<BlockId> blocks = std::vector<BlockId>(kPad * kPad * kPad, kAir);
  // Light values matching `blocks` (sky << 4 | block), same indexing.
  std::vector<uint8_t> light = std::vector<uint8_t>(kPad * kPad * kPad, 0xF0);
  ChunkCoord coord{};

  BlockId &at(int x, int y, int z) { return blocks[((y + 1) * kPad + (z + 1)) * kPad + (x + 1)]; }
  BlockId at(int x, int y, int z) const { return blocks[((y + 1) * kPad + (z + 1)) * kPad + (x + 1)]; }
  uint8_t lightAt(int x, int y, int z) const { return light[((y + 1) * kPad + (z + 1)) * kPad + (x + 1)]; }
};

// Builds a chunk mesh with binary greedy meshing (bitmask face culling and
// merging, 64-bit column masks), per-corner AO and light. Stateless and
// thread-safe; keep one scratch object per worker thread.
class ChunkMesher {
public:
  struct Scratch; // per-thread working memory, reused between calls
  static std::unique_ptr<Scratch> makeScratch();

  static void mesh(const MeshInput &in, const BlockRegistry &blocks,
                   Scratch &scratch, ChunkMeshData &out);
};

// Defined in the header so unique_ptr<Scratch> can be destroyed in any
// translation unit. Sized once; mesh() never allocates after warm-up.
struct ChunkMesher::Scratch {
  static constexpr int kPadCells = MeshInput::kPad * MeshInput::kPad * MeshInput::kPad;
  static constexpr int kCols = MeshInput::kPad * MeshInput::kPad;
  std::vector<uint8_t> table;                               // BlockId -> class
  std::vector<uint8_t> cls = std::vector<uint8_t>(kPadCells); // class per padded cell
  std::vector<uint8_t> occ = std::vector<uint8_t>(kPadCells); // AO occluder per padded cell
  // Column occupancy per normal axis (X, Y, Z); index = v * 34 + u (padded),
  // bit = padded coordinate along the axis.
  std::array<std::vector<uint64_t>, 3> opaqueCols{std::vector<uint64_t>(kCols), std::vector<uint64_t>(kCols),
                                                  std::vector<uint64_t>(kCols)};
  std::array<std::vector<uint64_t>, 3> drawCols{std::vector<uint64_t>(kCols), std::vector<uint64_t>(kCols),
                                                std::vector<uint64_t>(kCols)};
  std::array<std::vector<uint64_t>, 3> transCols{std::vector<uint64_t>(kCols), std::vector<uint64_t>(kCols),
                                                 std::vector<uint64_t>(kCols)};
  // Visible faces of one direction: [layer * 32 + v] bits along u.
  std::vector<uint32_t> gridOpaque = std::vector<uint32_t>(kChunkSize * kChunkSize);
  std::vector<uint32_t> gridTrans = std::vector<uint32_t>(kChunkSize * kChunkSize);
  std::vector<uint32_t> keys = std::vector<uint32_t>(kChunkSize * kChunkSize); // per layer
  // Cave-culling flood fill: open rows [y * 32 + z] bits along x.
  std::vector<uint32_t> open = std::vector<uint32_t>(kChunkSize * kChunkSize);
  std::vector<uint32_t> visited = std::vector<uint32_t>(kChunkSize * kChunkSize);
  std::vector<uint64_t> stack; // (row << 32) | bits
};

// Deterministic procedural terrain: the same seed produces the same world on
// every machine, so only player edits need storing or sending.
class WorldGenerator {
public:
  explicit WorldGenerator(uint64_t seed);
  // Fills one chunk (thread-safe; no shared mutable state).
  void generate(ChunkCoord c, Chunk &out) const;
  // Terrain height at a column (for spawning, AI and LOD).
  int surfaceHeight(int32_t x, int32_t z) const;
  uint64_t seed() const { return seed_; }

private:
  uint64_t seed_;
};

} // namespace atm::voxel
