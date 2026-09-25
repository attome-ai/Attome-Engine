#include "Lighting.h"

#include "BlockRegistry.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace atm::voxel {

namespace {
constexpr int R = ChunkLighting::kRegion;      // 64
constexpr int kRegionVolume = ChunkLighting::kRegionVolume;
constexpr int kOff = ChunkLighting::kOffset;   // 16
constexpr int kStepX = 1, kStepZ = R, kStepY = R * R;

constexpr uint8_t kPass = 1;  // light passes through the cell
constexpr uint8_t kAtten = 2; // sky light loses 1 going down through it

inline int rIndex(int x, int y, int z) { return (y << 12) | (z << 6) | x; }
} // namespace


std::unique_ptr<ChunkLighting::Scratch> ChunkLighting::makeScratch() {
  auto s = std::make_unique<Scratch>();
  s->queue.reserve(size_t(kRegionVolume));
  return s;
}

namespace {

// Standard BFS flood: each step loses one level; walls (pass = 0) stop it.
void flood(std::vector<uint32_t> &queue, uint8_t *light, const uint8_t *info) {
  static constexpr int kSteps[6] = {kStepX, -kStepX, kStepY, -kStepY, kStepZ, -kStepZ};
  size_t head = 0;
  while (head < queue.size()) {
    const uint32_t c = queue[head++];
    const uint8_t L = light[c];
    if (L <= 1)
      continue;
    const uint8_t next = uint8_t(L - 1);
    for (int s : kSteps) {
      const uint32_t n = uint32_t(int(c) + s);
      if ((info[n] & kPass) && light[n] < next) {
        light[n] = next;
        queue.push_back(n);
      }
    }
  }
  queue.clear();
}

} // namespace

void ChunkLighting::buildMeshInput(const ChunkNeighbourhood &hood, const BlockRegistry &registry,
                                   Scratch &s, MeshInput &out) {
  // --- 1. gather block ids of the 64^3 region (local -16 .. 47) ---------
  BlockId *ids = s.ids.data();
  for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz)
      for (int dx = -1; dx <= 1; ++dx) {
        const int lx0 = std::max(dx * kChunkSize, -kOff), lx1 = std::min(dx * kChunkSize + kChunkSize, R - kOff);
        const int ly0 = std::max(dy * kChunkSize, -kOff), ly1 = std::min(dy * kChunkSize + kChunkSize, R - kOff);
        const int lz0 = std::max(dz * kChunkSize, -kOff), lz1 = std::min(dz * kChunkSize + kChunkSize, R - kOff);
        const Chunk *ch = hood.at(dx, dy, dz);
        const bool belowWorld = hood.center.y + dy < 0;
        if (!ch || ch->isUniform()) {
          const BlockId fillId = ch ? ch->uniformBlock() : (belowWorld ? blocks::Bedrock : kAir);
          for (int y = ly0; y < ly1; ++y)
            for (int z = lz0; z < lz1; ++z) {
              BlockId *row = ids + rIndex(lx0 + kOff, y + kOff, z + kOff);
              std::fill(row, row + (lx1 - lx0), fillId);
            }
          continue;
        }
        ch->decodeAll(s.chunkIds.data());
        const int bx = dx * kChunkSize, by = dy * kChunkSize, bz = dz * kChunkSize;
        for (int y = ly0; y < ly1; ++y)
          for (int z = lz0; z < lz1; ++z) {
            const BlockId *src = s.chunkIds.data() + localIndex(lx0 - bx, y - by, z - bz);
            BlockId *dst = ids + rIndex(lx0 + kOff, y + kOff, z + kOff);
            std::memcpy(dst, src, size_t(lx1 - lx0) * sizeof(BlockId));
          }
      }

  // --- 2. per-cell light info -------------------------------------------
  const size_t nDefs = registry.size();
  s.table.assign(nDefs, 0);
  for (size_t i = 0; i < nDefs; ++i) {
    const BlockDef &d = registry.get(BlockId(i));
    uint8_t v = 0;
    if (d.render != BlockRender::Opaque)
      v |= kPass;
    if (d.liquid || d.render == BlockRender::Cutout)
      v |= kAtten;
    v |= uint8_t(std::min<int>(d.emission, 15) << 4);
    s.table[i] = v;
  }
  uint8_t *info = s.info.data();
  const uint8_t *table = s.table.data();
  for (int i = 0; i < kRegionVolume; ++i) {
    const BlockId id = ids[i];
    info[i] = id < nDefs ? table[id] : uint8_t(0); // unknown ids: opaque, dark
  }
  // Opaque, dark wall around the region (no bounds checks in the BFS).
  for (int a = 0; a < R; ++a)
    for (int b = 0; b < R; ++b) {
      info[rIndex(0, a, b)] = info[rIndex(R - 1, a, b)] = 0;
      info[rIndex(a, 0, b)] = info[rIndex(a, R - 1, b)] = 0;
      info[rIndex(a, b, 0)] = info[rIndex(a, b, R - 1)] = 0;
    }

  uint8_t *sky = s.sky.data();
  uint8_t *blk = s.blk.data();
  std::memset(sky, 0, size_t(kRegionVolume));
  std::memset(blk, 0, size_t(kRegionVolume));

  // --- 3. sky light: straight down, then spread sideways ----------------
  for (int z = 1; z < R - 1; ++z)
    for (int x = 1; x < R - 1; ++x) {
      int level = 15; // assume open sky above the region
      for (int y = R - 2; y >= 1; --y) {
        const int c = rIndex(x, y, z);
        const uint8_t inf = info[c];
        if (!(inf & kPass))
          level = 0;
        else if ((inf & kAtten) && level > 0)
          --level;
        if (level == 0)
          break; // everything below stays 0 until the BFS
        sky[c] = uint8_t(level);
      }
    }
  std::vector<uint32_t> &q = s.queue;
  q.clear();
  for (int y = 1; y < R - 1; ++y)
    for (int z = 1; z < R - 1; ++z)
      for (int x = 1; x < R - 1; ++x) {
        const int c = rIndex(x, y, z);
        const uint8_t L = sky[c];
        if (L <= 1)
          continue;
        const uint8_t next = uint8_t(L - 1);
        // Up neighbour is never darker than the column pass leaves us.
        if (((info[c + kStepX] & kPass) && sky[c + kStepX] < next) ||
            ((info[c - kStepX] & kPass) && sky[c - kStepX] < next) ||
            ((info[c + kStepZ] & kPass) && sky[c + kStepZ] < next) ||
            ((info[c - kStepZ] & kPass) && sky[c - kStepZ] < next) ||
            ((info[c - kStepY] & kPass) && sky[c - kStepY] < next))
          q.push_back(uint32_t(c));
      }
  flood(q, sky, info);

  // --- 4. block light from emitters ---------------------------------------
  for (int y = 1; y < R - 1; ++y)
    for (int z = 1; z < R - 1; ++z)
      for (int x = 1; x < R - 1; ++x) {
        const int c = rIndex(x, y, z);
        const BlockId id = ids[c];
        const uint8_t e = id < nDefs ? uint8_t(table[id] >> 4) : uint8_t(0);
        if (e > 0) {
          blk[c] = e;
          q.push_back(uint32_t(c));
        }
      }
  flood(q, blk, info);

  // --- 5. copy the 34^3 mesher window --------------------------------------
  constexpr int P = MeshInput::kPad;
  out.coord = hood.center;
  out.blocks.resize(size_t(P) * P * P);
  out.light.resize(size_t(P) * P * P);
  BlockId *ob = out.blocks.data();
  uint8_t *ol = out.light.data();
  for (int ly = -1; ly <= kChunkSize; ++ly)
    for (int lz = -1; lz <= kChunkSize; ++lz) {
      const int rc = rIndex(-1 + kOff, ly + kOff, lz + kOff);
      const int oc = ((ly + 1) * P + (lz + 1)) * P;
      std::memcpy(ob + oc, ids + rc, size_t(P) * sizeof(BlockId));
      for (int i = 0; i < P; ++i)
        ol[oc + i] = uint8_t((sky[rc + i] << 4) | blk[rc + i]);
    }
}

} // namespace atm::voxel
