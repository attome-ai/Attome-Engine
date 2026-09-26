// Deterministic procedural terrain. No global state: everything derives from
// the seed through integer hashing + gradient noise, so every machine (and
// every chunk, in any order) produces the same blocks.
//
//   height   = continents + hills (flattened in plains) + ridged mountains,
//              clamped to 40..160; sea level 64 (water fills y < 64)
//   surface  = grass/dirt, sand beaches and sea floor, stone on high peaks,
//              snow above ~128, bedrock at y = 0
//   caves    = "spaghetti" tunnels (two 3D noise zero-sets intersecting) +
//              deep caverns; 3D noise sampled on a 4-block lattice and
//              trilinearly interpolated; tunnels may break the surface in
//              some regions
//   ores     = 4^3 cells pick an ore by depth, voxels inside get it by hash
//   trees    = one candidate per 6x6-column cell (position/height from the
//              cell hash), so any chunk reproduces trees overlapping it

#include "BlockRegistry.h"
#include "Chunk.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace atm::voxel {

namespace {

constexpr int kSeaLevel = 64; // water fills y < 64
constexpr int kMinHeight = 40, kMaxHeight = 160;
constexpr int kTreeCell = 8;
constexpr int kTreeReach = 5;     // canopy / branches extend up to 5 blocks from the trunk
constexpr int kTreeCanopyUp = 4;  // canopy rises up to 4 blocks above the trunk top
constexpr int kMaxTreeHeight = 15; // trunk (<= 10) + canopy above it
constexpr int kCaveStep = 4;      // cave lattice spacing

inline uint64_t mix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Fast 32-bit lattice hash for noise corners.
inline uint32_t hash2i(uint32_t seed, int32_t x, int32_t y) {
  uint32_t h = seed ^ (uint32_t(x) * 0x27D4EB2Du) ^ (uint32_t(y) * 0x165667B1u);
  h = (h ^ (h >> 15)) * 0x2C1B3C6Du;
  h = (h ^ (h >> 12)) * 0x297A2D39u;
  return h ^ (h >> 15);
}
inline uint32_t hash3i(uint32_t seed, int32_t x, int32_t y, int32_t z) {
  uint32_t h = seed ^ (uint32_t(x) * 0x27D4EB2Du) ^ (uint32_t(y) * 0x165667B1u) ^ (uint32_t(z) * 0x9E3779B1u);
  h = (h ^ (h >> 15)) * 0x2C1B3C6Du;
  h = (h ^ (h >> 12)) * 0x297A2D39u;
  return h ^ (h >> 15);
}
// Uniform [0, 1) from a hash.
inline double unit(uint32_t h) { return double(h >> 8) * (1.0 / 16777216.0); }

inline double fade(double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
inline double lerp(double a, double b, double t) { return a + (b - a) * t; }
inline double smoothstep(double e0, double e1, double x) {
  const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

constexpr double kG2[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1},
                              {0.70710678, 0.70710678}, {-0.70710678, 0.70710678},
                              {0.70710678, -0.70710678}, {-0.70710678, -0.70710678}};

inline double grad2(uint32_t h, double dx, double dy) {
  const double *g = kG2[h & 7u];
  return g[0] * dx + g[1] * dy;
}

// 2D gradient (Perlin) noise, roughly -1..1.
double perlin2(uint32_t seed, double x, double y) {
  const double fx0 = std::floor(x), fy0 = std::floor(y);
  const int32_t x0 = int32_t(fx0), y0 = int32_t(fy0);
  const double dx = x - fx0, dy = y - fy0;
  const double n00 = grad2(hash2i(seed, x0, y0), dx, dy);
  const double n10 = grad2(hash2i(seed, x0 + 1, y0), dx - 1, dy);
  const double n01 = grad2(hash2i(seed, x0, y0 + 1), dx, dy - 1);
  const double n11 = grad2(hash2i(seed, x0 + 1, y0 + 1), dx - 1, dy - 1);
  const double u = fade(dx), v = fade(dy);
  return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v) * 1.41;
}

inline double grad3(uint32_t h, double x, double y, double z) {
  switch (h % 12u) {
  case 0: return x + y;
  case 1: return -x + y;
  case 2: return x - y;
  case 3: return -x - y;
  case 4: return x + z;
  case 5: return -x + z;
  case 6: return x - z;
  case 7: return -x - z;
  case 8: return y + z;
  case 9: return -y + z;
  case 10: return y - z;
  default: return -y - z;
  }
}

// 3D gradient noise, roughly -1..1.
double perlin3(uint32_t seed, double x, double y, double z) {
  const double fx0 = std::floor(x), fy0 = std::floor(y), fz0 = std::floor(z);
  const int32_t x0 = int32_t(fx0), y0 = int32_t(fy0), z0 = int32_t(fz0);
  const double dx = x - fx0, dy = y - fy0, dz = z - fz0;
  const double u = fade(dx), v = fade(dy), w = fade(dz);
  double c[2][2][2];
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        c[k][j][i] = grad3(hash3i(seed, x0 + i, y0 + j, z0 + k), dx - i, dy - j, dz - k);
  const double x00 = lerp(c[0][0][0], c[0][0][1], u), x10 = lerp(c[0][1][0], c[0][1][1], u);
  const double x01 = lerp(c[1][0][0], c[1][0][1], u), x11 = lerp(c[1][1][0], c[1][1][1], u);
  return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

double fbm2(uint32_t seed, double x, double y, int octaves) {
  double sum = 0, amp = 1, norm = 0;
  for (int o = 0; o < octaves; ++o) {
    sum += perlin2(seed + uint32_t(o) * 0x9E3779B9u, x, y) * amp;
    norm += amp;
    amp *= 0.5;
    x *= 2.0;
    y *= 2.0;
  }
  return sum / norm;
}

struct Seeds {
  uint32_t cont, hills, detail, mountMask, ridge, plains, forest, snow;
  uint32_t cave1, cave2, cavern, caveOpen, ore, tree;
  explicit Seeds(uint64_t seed) {
    uint64_t s = seed;
    auto next = [&s] {
      s = mix64(s);
      return uint32_t(s);
    };
    cont = next();
    hills = next();
    detail = next();
    mountMask = next();
    ridge = next();
    plains = next();
    forest = next();
    snow = next();
    cave1 = next();
    cave2 = next();
    cavern = next();
    caveOpen = next();
    ore = next();
    tree = next();
  }
};

struct ColumnInfo {
  int h = 64;            // top terrain block (before caves)
  BlockId top = blocks::Grass, filler = blocks::Dirt;
  int fillerDepth = 3;
  bool treeable = false;
  bool caveOpen = false; // tunnels may break the surface here
};

ColumnInfo columnInfo(const Seeds &s, int32_t x, int32_t z) {
  const double fx = double(x), fz = double(z);
  // Mostly flat, gently rolling land (MMO-style open fields): broad low
  // undulation, soft wide hills, a little surface detail. Mountains only far
  // from the spawn (the highlands), fading in from ~300 blocks out.
  const double cont = fbm2(s.cont, fx / 900.0, fz / 900.0, 3);
  const double hills = fbm2(s.hills, fx / 240.0, fz / 240.0, 3);
  const double detail = fbm2(s.detail, fx / 60.0, fz / 60.0, 2);
  const double dist = std::sqrt(fx * fx + fz * fz);
  const double far = smoothstep(300.0, 520.0, dist);
  const double mmask = far * smoothstep(0.15, 0.45, fbm2(s.mountMask, fx / 700.0, fz / 700.0, 2));
  double ridge = 0.0;
  if (mmask > 0.0) {
    ridge = 1.0 - std::fabs(fbm2(s.ridge, fx / 260.0, fz / 260.0, 4));
    ridge *= ridge;
  }
  const double plains = smoothstep(-0.1, 0.3, fbm2(s.plains, fx / 500.0, fz / 500.0, 2));
  const double hh = 70.0 + cont * 11.0 + hills * 6.0 * (1.0 - 0.7 * plains) + detail * 1.2 +
                    mmask * ridge * 48.0;
  ColumnInfo c;
  c.h = std::clamp(int(std::floor(hh)), kMinHeight, kMaxHeight);
  const int snowLine = 126 + int(hash2i(s.snow, x, z) & 3u);
  if (c.h >= snowLine) {
    c.top = blocks::Snow;
    c.filler = blocks::Stone;
    c.fillerDepth = 1;
  } else if (c.h >= 108) {
    c.top = blocks::Stone;
    c.filler = blocks::Stone;
    c.fillerDepth = 0;
  } else if (c.h <= kSeaLevel + 1) {
    // beaches and sea floor
    c.top = c.h >= kSeaLevel - 8 ? blocks::Sand : blocks::Dirt;
    c.filler = blocks::Sand;
    c.fillerDepth = 3;
  } else {
    c.top = blocks::Grass;
    c.filler = blocks::Dirt;
    c.fillerDepth = 3;
    c.treeable = c.h >= kSeaLevel + 2;
  }
  c.caveOpen = c.h >= kSeaLevel + 2 && fbm2(s.caveOpen, fx / 120.0, fz / 120.0, 2) > 0.3;
  return c;
}

// Noise values at one cave lattice point (world coords multiple of 4).
struct CaveSample {
  float a, b, cavern;
};
CaveSample caveSample(const Seeds &s, int32_t x, int32_t y, int32_t z) {
  const double fx = double(x), fy = double(y), fz = double(z);
  CaveSample c;
  c.a = float(perlin3(s.cave1, fx / 48.0, fy / 32.0, fz / 48.0));
  c.b = float(perlin3(s.cave2, fx / 48.0, fy / 32.0, fz / 48.0));
  c.cavern = y < 48 ? float(perlin3(s.cavern, fx / 90.0, fy / 60.0, fz / 90.0)) : -1.0f;
  return c;
}

inline bool caveCarves(const CaveSample &c, int32_t y, const ColumnInfo &col) {
  if (y < 1)
    return false;
  if (y > col.h - 4 && !(col.caveOpen && y <= col.h))
    return false;
  const float t = c.a * c.a + c.b * c.b;
  return t < 0.012f || c.cavern > 0.55f;
}

inline CaveSample trilerp(const CaveSample (&k)[2][2][2], float fx, float fy, float fz) {
  auto one = [&](float CaveSample::*m) {
    const float x00 = k[0][0][0].*m + (k[0][0][1].*m - k[0][0][0].*m) * fx;
    const float x10 = k[0][1][0].*m + (k[0][1][1].*m - k[0][1][0].*m) * fx;
    const float x01 = k[1][0][0].*m + (k[1][0][1].*m - k[1][0][0].*m) * fx;
    const float x11 = k[1][1][0].*m + (k[1][1][1].*m - k[1][1][0].*m) * fx;
    const float y0 = x00 + (x10 - x00) * fy;
    const float y1 = x01 + (x11 - x01) * fy;
    return y0 + (y1 - y0) * fz;
  };
  return {one(&CaveSample::a), one(&CaveSample::b), one(&CaveSample::cavern)};
}

// Point evaluation, identical to the chunk lattice path (same lattice, same
// interpolation order), used by tree placement.
CaveSample cavePoint(const Seeds &s, int32_t x, int32_t y, int32_t z) {
  const int32_t lx = floorDiv(x, kCaveStep), ly = floorDiv(y, kCaveStep), lz = floorDiv(z, kCaveStep);
  CaveSample k[2][2][2];
  for (int dz = 0; dz < 2; ++dz)
    for (int dy = 0; dy < 2; ++dy)
      for (int dx = 0; dx < 2; ++dx)
        k[dz][dy][dx] = caveSample(s, (lx + dx) * kCaveStep, (ly + dy) * kCaveStep, (lz + dz) * kCaveStep);
  return trilerp(k, float(x - lx * kCaveStep) / kCaveStep, float(y - ly * kCaveStep) / kCaveStep,
                 float(z - lz * kCaveStep) / kCaveStep);
}

BlockId oreAt(const Seeds &s, int32_t x, int32_t y, int32_t z) {
  const double r = unit(hash3i(s.ore, x >> 2, y >> 2, z >> 2));
  BlockId ore = kAir;
  double p = 0.0, fill = 0.4;
  if (y <= 25 && r < (p += 0.006)) {
    ore = blocks::Crystal;
    fill = 0.3;
  } else if (y <= 35 && r < (p += 0.015)) {
    ore = blocks::GoldOre;
  } else if (y >= 10 && y <= 64 && r < (p += 0.03)) {
    ore = blocks::IronOre;
  } else if (y >= 28 && r < (p += 0.05)) {
    ore = blocks::CopperOre;
  }
  if (ore == kAir)
    return kAir;
  return unit(hash3i(s.ore ^ 0x5bd1e995u, x, y, z)) < fill ? ore : kAir;
}

// Tree kinds: a round oak with a lumpy two-tone canopy and side branches, a
// tall layered pine, and a wide tree with several canopy clusters.
enum class TreeKind : uint8_t { Oak, Pine, Wide };

struct Tree {
  int32_t x, z;
  int base;   // first trunk block y
  int trunk;  // trunk height
  bool valid;
  TreeKind kind = TreeKind::Oak;
  float rx = 3.0f, ry = 2.6f; // canopy radii (oak / wide)
  uint32_t seed = 0;
};

Tree treeInCell(const Seeds &s, int32_t cx, int32_t cz) {
  Tree t{0, 0, 0, 0, false};
  const uint32_t h = hash2i(s.tree, cx, cz);
  t.x = cx * kTreeCell + 2 + int32_t(h % 4u);
  t.z = cz * kTreeCell + 2 + int32_t((h >> 4) % 4u);
  const double forest = fbm2(s.forest, double(t.x) / 300.0, double(t.z) / 300.0, 2);
  const double prob = 0.12 + 0.6 * smoothstep(-0.05, 0.35, forest);
  if (unit(hash2i(s.tree ^ 0xA511E9B3u, cx, cz)) >= prob)
    return t;
  const ColumnInfo col = columnInfo(s, t.x, t.z);
  if (!col.treeable)
    return t;
  if (caveCarves(cavePoint(s, t.x, col.h, t.z), col.h, col))
    return t; // ground under the trunk was carved away
  t.base = col.h + 1;
  t.seed = hash2i(s.tree ^ 0x3C6EF372u, cx, cz);
  // More pines in deep forest, more wide trees in the open.
  const double k = unit(t.seed);
  const double pine = 0.15 + 0.25 * smoothstep(0.1, 0.5, forest);
  if (k < pine) {
    t.kind = TreeKind::Pine;
    t.trunk = 7 + int((h >> 8) % 4u);
  } else if (k < pine + 0.18) {
    t.kind = TreeKind::Wide;
    t.trunk = 4 + int((h >> 8) % 2u);
    t.rx = 3.2f + float(unit(t.seed >> 3)) * 0.6f;
    t.ry = 2.2f;
  } else {
    t.kind = TreeKind::Oak;
    t.trunk = 4 + int((h >> 8) % 3u);
    t.rx = 2.6f + float(unit(t.seed >> 5)) * 1.3f;
    t.ry = 2.3f + float(unit(t.seed >> 7)) * 0.9f;
  }
  t.valid = true;
  return t;
}

// Canopy / branch block of a tree at offset (dx, dy, dz) from the trunk top
// (kAir = nothing). Deterministic per world position.
BlockId treeBlockAt(const Tree &t, int dx, int dy, int dz) {
  const uint32_t hv = hash3i(t.seed, t.x + dx, dy, t.z + dz);
  const float jitter = float(hv & 1023u) / 1023.0f - 0.5f; // -0.5..0.5
  auto shade = [&](float ny) { // underside dark, crown light, speckled between
    if (ny < -0.25f) return blocks::OakLeavesDark;
    if (ny > 0.45f + jitter * 0.3f) return blocks::OakLeavesLight;
    return (hv >> 10) % 5u == 0 ? blocks::OakLeavesDark : blocks::OakLeaves;
  };
  switch (t.kind) {
  case TreeKind::Oak: {
    // Lumpy ellipsoid centred a little above the trunk top.
    const float fx = dx / t.rx, fy = (dy - 1.0f) / t.ry, fz = dz / t.rx;
    const float d = fx * fx + fy * fy + fz * fz + jitter * 0.45f;
    if (d <= 1.0f) return shade(fy);
    // Two short branches under the canopy.
    const int bx = (t.seed & 1u) ? 1 : -1, bz = (t.seed & 2u) ? 1 : -1;
    if ((dy == -1 && dx == bx && dz == 0) || (dy == 0 && dx == 2 * bx && dz == 0) || (dy == -2 && dz == bz && dx == 0))
      return blocks::OakLog;
    return kAir;
  }
  case TreeKind::Wide: {
    // Main crown plus two side clusters, flatter.
    const int ox = (t.seed & 4u) ? 2 : -2, oz = (t.seed & 8u) ? 2 : -2;
    const float c[3][4] = {{0.0f, 0.5f, 0.0f, 1.0f}, {float(ox), -0.5f, float(oz) * 0.5f, 0.72f},
                           {float(-ox) * 0.6f, 0.0f, float(-oz) * 1.2f, 0.66f}};
    for (const auto &k : c) {
      const float r = t.rx * k[3];
      const float fx = (dx - k[0]) / r, fy = (dy - k[1]) / (t.ry * (k[3] + 0.15f)), fz = (dz - k[2]) / r;
      const float d = fx * fx + fy * fy + fz * fz + jitter * 0.4f;
      if (d <= 1.0f) return shade(fy);
    }
    // Limbs out to the side clusters.
    if (dy == -1 && ((dx == ox / 2 && dz == 0) || (dx == ox && dz == oz / 2))) return blocks::OakLog;
    return kAir;
  }
  case TreeKind::Pine: {
    // Stacked tiers from the lower trunk to a point, each tier wide at the
    // bottom and narrowing; tiers step in as they go up.
    const int bottom = -(t.trunk - 3); // first tier a few blocks above the ground
    if (dy < bottom || dy > 2) return kAir;
    const int fromTop = 2 - dy;
    const float tierPhase = float(fromTop % 3) / 3.0f; // 0 at a tier's top
    const float r = std::min(0.8f + float(fromTop) * 0.42f, 3.6f) * (0.55f + 0.45f * tierPhase) + jitter * 0.3f;
    const float d = std::sqrt(float(dx * dx + dz * dz));
    if (fromTop == 0) return (dx == 0 && dz == 0) ? blocks::PineLeaves : kAir;
    if (d <= r) return (hv >> 11) % 6u == 0 ? blocks::OakLeavesDark : blocks::PineLeaves;
    return kAir;
  }
  }
  return kAir;
}

} // namespace

WorldGenerator::WorldGenerator(uint64_t seed) : seed_(seed) {}

int WorldGenerator::surfaceHeight(int32_t x, int32_t z) const {
  const Seeds s(seed_);
  return columnInfo(s, x, z).h;
}

void WorldGenerator::generate(ChunkCoord c, Chunk &out) const {
  if (c.y < 0 || c.y >= kWorldChunksY) {
    out.fill(kAir);
    return;
  }
  const Seeds s(seed_);
  const BlockPos o = chunkOrigin(c);

  struct Scratch {
    std::vector<BlockId> ids = std::vector<BlockId>(kChunkVolume);
    std::vector<ColumnInfo> cols = std::vector<ColumnInfo>(kChunkSize * kChunkSize);
    std::vector<CaveSample> lattice;
  };
  thread_local Scratch sc;
  BlockId *ids = sc.ids.data();

  int maxH = 0, minH = kMaxHeight;
  for (int lz = 0; lz < kChunkSize; ++lz)
    for (int lx = 0; lx < kChunkSize; ++lx) {
      ColumnInfo &col = sc.cols[size_t(lz * kChunkSize + lx)];
      col = columnInfo(s, o.x + lx, o.z + lz);
      maxH = std::max(maxH, col.h);
      minH = std::min(minH, col.h);
    }

  const int y0 = o.y, y1 = o.y + kChunkSize - 1;
  bool anything = false;

  if (y0 > std::max(maxH, kSeaLevel - 1)) {
    std::fill(ids, ids + kChunkVolume, kAir);
  } else {
    anything = true;
    // Cave lattice: 9^3 samples at multiples of 4 covering the chunk.
    constexpr int L = kChunkSize / kCaveStep + 1;
    const bool caves = y0 <= maxH;
    if (caves) {
      sc.lattice.resize(size_t(L) * L * L);
      for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j)
          for (int i = 0; i < L; ++i)
            sc.lattice[size_t((k * L + j) * L + i)] =
                caveSample(s, o.x + i * kCaveStep, o.y + j * kCaveStep, o.z + k * kCaveStep);
    }
    for (int lz = 0; lz < kChunkSize; ++lz)
      for (int lx = 0; lx < kChunkSize; ++lx) {
        const ColumnInfo &col = sc.cols[size_t(lz * kChunkSize + lx)];
        const int32_t wx = o.x + lx, wz = o.z + lz;
        for (int ly = 0; ly < kChunkSize; ++ly) {
          const int32_t wy = y0 + ly;
          BlockId b;
          if (wy == 0) {
            b = blocks::Bedrock;
          } else if (wy > col.h) {
            b = wy < kSeaLevel ? blocks::Water : kAir;
          } else {
            const int depth = col.h - wy;
            b = depth == 0 ? col.top : depth <= col.fillerDepth ? col.filler : blocks::Stone;
            if (caves) {
              const int i = lx / kCaveStep, j = ly / kCaveStep, k = lz / kCaveStep;
              CaveSample kk[2][2][2];
              for (int dz = 0; dz < 2; ++dz)
                for (int dy = 0; dy < 2; ++dy)
                  for (int dx = 0; dx < 2; ++dx)
                    kk[dz][dy][dx] = sc.lattice[size_t(((k + dz) * L + (j + dy)) * L + (i + dx))];
              const CaveSample cs = trilerp(kk, float(lx % kCaveStep) / kCaveStep,
                                            float(ly % kCaveStep) / kCaveStep, float(lz % kCaveStep) / kCaveStep);
              if (caveCarves(cs, wy, col))
                b = kAir;
            }
            if (b == blocks::Stone) {
              const BlockId ore = oreAt(s, wx, wy, wz);
              if (ore != kAir)
                b = ore;
            }
          }
          ids[localIndex(lx, ly, lz)] = b;
        }
      }
  }

  // Trees from every cell whose canopy can reach this chunk.
  if (y1 >= kSeaLevel && y0 <= kMaxHeight + kMaxTreeHeight) {
    const int32_t cx0 = floorDiv(o.x - kTreeReach, kTreeCell), cx1 = floorDiv(o.x + kChunkSize + kTreeReach, kTreeCell);
    const int32_t cz0 = floorDiv(o.z - kTreeReach, kTreeCell), cz1 = floorDiv(o.z + kChunkSize + kTreeReach, kTreeCell);
    for (int32_t cz = cz0; cz <= cz1; ++cz)
      for (int32_t cx = cx0; cx <= cx1; ++cx) {
        const Tree t = treeInCell(s, cx, cz);
        if (!t.valid)
          continue;
        const int top = t.base + t.trunk - 1;
        if (top + kTreeCanopyUp < y0 || t.base > y1)
          continue;
        if (t.x + kTreeReach < o.x || t.x - kTreeReach >= o.x + kChunkSize || t.z + kTreeReach < o.z ||
            t.z - kTreeReach >= o.z + kChunkSize)
          continue;
        if (!anything) {
          anything = true; // ids already all air
        }
        // Canopy first (leaves only into air; branches over leaves), then
        // the trunk: the result is independent of tree order.
        for (int dy = -(t.trunk - 1); dy <= kTreeCanopyUp; ++dy) {
          const int ly = top + dy - o.y;
          if (unsigned(ly) >= unsigned(kChunkSize))
            continue;
          for (int dz = -kTreeReach; dz <= kTreeReach; ++dz)
            for (int dx = -kTreeReach; dx <= kTreeReach; ++dx) {
              const int lx = t.x + dx - o.x, lz = t.z + dz - o.z;
              if (unsigned(lx) >= unsigned(kChunkSize) || unsigned(lz) >= unsigned(kChunkSize))
                continue;
              const BlockId tb = treeBlockAt(t, dx, dy, dz);
              if (tb == kAir)
                continue;
              BlockId &b = ids[localIndex(lx, ly, lz)];
              if (b == kAir || (tb == blocks::OakLog && blocks::isLeaves(b)))
                b = tb;
            }
        }
        for (int y = t.base; y <= top; ++y) {
          const int lx = t.x - o.x, ly = y - o.y, lz = t.z - o.z;
          if (unsigned(lx) >= unsigned(kChunkSize) || unsigned(ly) >= unsigned(kChunkSize) ||
              unsigned(lz) >= unsigned(kChunkSize))
            continue;
          BlockId &b = ids[localIndex(lx, ly, lz)];
          if (b == kAir || blocks::isLeaves(b))
            b = blocks::OakLog;
        }
      }
  }

  if (!anything) {
    out.fill(kAir);
    return;
  }
  out.encodeAll(ids);
}

} // namespace atm::voxel
