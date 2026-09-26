#include "Town.h"

#include "TownMicro.h"

#include "../../../engine/voxel/BlockRegistry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace ao::world {

namespace vb = atm::voxel::blocks;
using atm::voxel::BlockId;
using atm::voxel::kAir;

// Block template of the whole town: x, z in [-R, R] around the centre,
// y in [0, H) above the ground surface (y = 0 is the ground layer itself).
// Fine-voxel structures appear here as Barrier (collision) and Light blocks.
struct TownGrid {
  static constexpr int R = 64, H = 64, W = 2 * R + 1;
  std::vector<BlockId> blocks = std::vector<BlockId>(size_t(W) * W * H, kAir);
  std::vector<uint8_t> top = std::vector<uint8_t>(size_t(W) * W, 0);     // highest non-air y per column
  std::vector<uint32_t> mapColor = std::vector<uint32_t>(size_t(W) * W, 0); // 0 = use the block colour
  std::vector<int16_t> mapY = std::vector<int16_t>(size_t(W) * W, -1);     // fine y of mapColor
  std::vector<TownProp> props;
  std::vector<MicroModel> structures;

  static bool in(int x, int y, int z) { return std::abs(x) <= R && std::abs(z) <= R && y >= 0 && y < H; }
  static size_t idx(int x, int y, int z) { return (size_t(y) * W + size_t(z + R)) * W + size_t(x + R); }
  static size_t col(int x, int z) { return size_t(z + R) * W + size_t(x + R); }
  BlockId get(int x, int y, int z) const { return in(x, y, z) ? blocks[idx(x, y, z)] : kAir; }
};

namespace {

constexpr int kSeaLevel = 64;
constexpr double kPi = 3.14159265358979323846;

uint32_t hash3(int x, int y, int z, uint32_t seed = 0) {
  uint32_t h = seed ^ (uint32_t(x) * 73856093u) ^ (uint32_t(y) * 19349663u) ^ (uint32_t(z) * 83492791u);
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return h;
}
float rand01(int x, int y, int z, uint32_t seed) { return float(hash3(x, y, z, seed) & 0xFFFFu) / 65535.0f; }

struct Builder {
  TownGrid &g;

  void set(int x, int y, int z, BlockId b) {
    if (TownGrid::in(x, y, z)) g.blocks[TownGrid::idx(x, y, z)] = b;
  }
  BlockId get(int x, int y, int z) const { return g.get(x, y, z); }
  bool air(int x, int y, int z) const { return get(x, y, z) == kAir; }
  void box(int x0, int y0, int z0, int x1, int y1, int z1, BlockId b) {
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    if (z0 > z1) std::swap(z0, z1);
    for (int y = y0; y <= y1; ++y)
      for (int z = z0; z <= z1; ++z)
        for (int x = x0; x <= x1; ++x) set(x, y, z, b);
  }
  void prop(const char *name, float x, float y, float z, float yaw = 0.0f) { g.props.push_back({name, x, y, z, yaw}); }
  template <class F> void discEach(float cx, float cz, float r, F &&f) {
    const int R = int(std::ceil(r)) + 1;
    for (int z = int(cz) - R; z <= int(cz) + R; ++z)
      for (int x = int(cx) - R; x <= int(cx) + R; ++x) {
        const float dx = x + 0.5f - cx, dz = z + 0.5f - cz;
        const float d = std::sqrt(dx * dx + dz * dz);
        if (d <= r) f(x, z, d);
      }
  }

  // Adds a fine-voxel structure: blocks it fills become Barrier (collision),
  // lamp and crystal voxels light their block with an invisible Light.
  // Floors (models starting at the ground layer) replace the ground there.
  void structure(MicroModel m) {
    const int bw = m.sx / kMicro, bh = m.sy / kMicro, bd = m.sz / kMicro;
    for (int cy = 0; cy < bh; ++cy)
      for (int cz = 0; cz < bd; ++cz)
        for (int cx = 0; cx < bw; ++cx) {
          int filled = 0, glow = 0;
          for (int y = 0; y < kMicro; ++y)
            for (int z = 0; z < kMicro; ++z)
              for (int x = 0; x < kMicro; ++x) {
                const uint8_t c = m.at(cx * kMicro + x, cy * kMicro + y, cz * kMicro + z);
                if (!c) continue;
                ++filled;
                if (c == mc::GlowLantern || c == mc::Crystal || c == mc::CrystalBright || c == mc::GlowGold) ++glow;
              }
          const int X = m.bx + cx, Y = m.by + cy, Z = m.bz + cz;
          if (!TownGrid::in(X, Y, Z)) continue;
          const BlockId here = get(X, Y, Z);
          if (filled >= 16 && (here == kAir || here == vb::Light || (Y == 0 && here != vb::Water)))
            set(X, Y, Z, vb::Barrier);
          else if (glow >= 2 && here == kAir)
            set(X, Y, Z, vb::Light);
        }
    // Map colour: the highest fine voxel at the centre of each block column.
    for (int cz = 0; cz < bd; ++cz)
      for (int cx = 0; cx < bw; ++cx) {
        const int X = m.bx + cx, Z = m.bz + cz;
        if (std::abs(X) > TownGrid::R || std::abs(Z) > TownGrid::R) continue;
        for (int y = m.sy - 1; y >= 0; --y) {
          const uint8_t c = m.at(cx * kMicro + 2, y, cz * kMicro + 2);
          if (!c) continue;
          const int fy = m.by * kMicro + y;
          const size_t i = TownGrid::col(X, Z);
          if (fy > g.mapY[i]) {
            g.mapY[i] = int16_t(fy);
            g.mapColor[i] = microPalette()[c];
          }
          break;
        }
      }
    g.structures.push_back(std::move(m));
  }
};

// ---------------------------------------------------------------------------
// Town layout (north is -Z)
// ---------------------------------------------------------------------------

constexpr int kWall = 58;       // town wall at |x| or |z| = 57..58
constexpr int kRoadHalf = 3;    // roads: |offset| <= 2 paving, 3 = curb
constexpr float kPlazaR = 16.5f, kStreetR = 21.5f;
constexpr int kTerraceY = 6;    // castle terrace height (blocks above ground)

bool onRoad(int x, int z) {
  const int ax = std::abs(x), az = std::abs(z);
  if (az <= kRoadHalf && ax >= 15) return true; // east-west road
  if (ax <= kRoadHalf && z >= 15) return true;  // south road
  return false;
}

void buildGround(Builder &b) {
  const int R = TownGrid::R;
  for (int z = -R; z <= R; ++z)
    for (int x = -R; x <= R; ++x) {
      const float r = std::sqrt(float(x * x + z * z));
      BlockId g = vb::Grass;
      if (r <= kPlazaR) {
        g = vb::PlazaTile; // covered by the fine-voxel square
      } else if (r <= kPlazaR + 1.0f) {
        g = vb::WhiteStoneTrim;
      } else if (r <= kStreetR) {
        g = vb::Cobblestone;
      } else if (onRoad(x, z)) {
        const bool ew = std::abs(z) <= kRoadHalf && std::abs(x) >= 15;
        const int off = ew ? std::abs(z) : std::abs(x);
        const int along = ew ? std::abs(x) : std::abs(z);
        if (along > kWall) g = off == kRoadHalf ? vb::Grass : vb::Path; // outside the gates
        else if (off == kRoadHalf) g = vb::WhiteStoneTrim;
        else g = rand01(x, 0, z, 13) < 0.15f ? vb::PlazaTileDark : vb::Cobblestone;
      } else if (r <= kStreetR + 1.0f) {
        g = vb::WhiteStoneTrim; // curb around the square
      }
      b.set(x, 0, z, g);
    }
  // Hedge ring around the square, with flower beds, broken by the roads.
  for (int z = -26; z <= 26; ++z)
    for (int x = -26; x <= 26; ++x) {
      const float r = std::sqrt(float(x * x + z * z));
      if (r <= kStreetR + 1.0f || r > kStreetR + 2.0f || b.get(x, 0, z) != vb::Grass) continue;
      if (z < -12 && std::abs(x) <= 7) continue; // castle stairs
      const float a = float(std::atan2(double(z), double(x)));
      const int seg = int(std::floor((a + float(kPi)) / float(kPi) * 24.0f));
      b.set(x, 1, z, seg % 4 == 0 ? (seg % 8 == 0 ? vb::FlowersRed : vb::FlowersYellow) : vb::HedgeLeaves);
    }
}

void buildSquare(Builder &b) {
  b.structure(townmicro::plazaFloor());
  // Fountain water (world blocks, so it looks and behaves like water).
  b.discEach(0.5f, 0.5f, 5.4f, [&](int x, int z, float d) {
    if (d > 1.9f) b.set(x, 1, z, vb::Water);
  });
  b.structure(townmicro::fountain());
  for (int k = 0; k < 4; ++k) {
    const double a = kPi / 4.0 + k * kPi / 2.0;
    b.prop("lantern", float(0.5 + std::cos(a) * 6.4), 3.0f, float(0.5 + std::sin(a) * 6.4));
  }
  // Eight banner poles between the roads, banners facing the square.
  for (int k = 0; k < 8; ++k) {
    const double a = kPi / 8.0 + k * kPi / 4.0;
    const int px = int(std::lround(std::cos(a) * 14.0)), pz = int(std::lround(std::sin(a) * 14.0));
    const int tx = int(std::lround(-std::sin(a))), tz = int(std::lround(std::cos(a)));
    b.structure(townmicro::bannerPole(px, pz, tx != 0 && std::abs(std::sin(a)) > 0.5 ? tx : 0,
                                      tx != 0 && std::abs(std::sin(a)) > 0.5 ? 0 : tz));
  }
  for (int k = 0; k < 4; ++k) {
    const double a = kPi / 4.0 + k * kPi / 2.0;
    b.prop("street_lamp", float(0.5 + std::cos(a) * 19.5), 1.0f, float(0.5 + std::sin(a) * 19.5));
    b.prop("potted_bush", float(0.5 + std::cos(a) * 9.5), 1.0f, float(0.5 + std::sin(a) * 9.5));
  }
  b.prop("bench", 12.5f, 1.0f, 0.5f, float(kPi / 2.0));
  b.prop("bench", -11.5f, 1.0f, 0.5f, float(-kPi / 2.0));
  b.prop("sign_post", 5.5f, 1.0f, 15.5f, float(kPi));
}

void buildCastle(Builder &b) {
  b.structure(townmicro::castle(kTerraceY));
  b.prop("street_lamp", -6.5f, 1.0f, -16.5f);
  b.prop("street_lamp", 7.5f, 1.0f, -16.5f);
  const float y0 = float(kTerraceY + 1);
  for (int sx : {-1, 1}) {
    b.prop("street_lamp", sx * 6.5f + 0.5f, y0, -30.5f);
    b.prop("barrel", sx * 17.5f + 0.5f, y0, -32.5f);
    b.prop("barrel", sx * 16.5f + 0.5f, y0, -31.5f);
    b.prop("small_crate", sx * 17.5f + 0.5f, y0, -30.5f);
    b.prop("potted_bush", sx * 5.5f + 0.5f, y0, -33.5f);
    for (int x : {10, 16, 22}) b.prop("torch", sx * x + 0.5f, float(kTerraceY) + 2.8f, -22.5f);
  }
}

void buildWalls(Builder &b) {
  for (int side = 0; side < 4; ++side) b.structure(townmicro::wallSide(side, kWall));
  for (int s : {-1, 1}) {
    b.structure(townmicro::gatehouse(s * kWall, -6, 11));
    b.structure(townmicro::gatehouse(s * kWall, 6, 11));
  }
  b.structure(townmicro::gatehouse(-6, kWall, 11));
  b.structure(townmicro::gatehouse(6, kWall, 11));
  for (int sx : {-1, 1})
    for (int sz : {-1, 1}) b.structure(townmicro::cornerTower(sx * kWall, sz * kWall));
}

// Houses: the door faces the nearest road; a path leads to it.
void buildHouses(Builder &b) {
  struct H { int x0, z0, x1, z1, floors; bool red; };
  static constexpr H kHouses[] = {
      {30, -20, 38, -11, 2, false}, {43, -22, 52, -12, 1, true},  {30, -46, 39, -34, 1, false},
      {44, -50, 53, -38, 2, false}, {-38, -20, -30, -11, 2, true}, {-52, -22, -43, -12, 1, false},
      {-39, -46, -30, -34, 2, false}, {-53, -50, -44, -38, 1, true}, {12, 26, 21, 35, 2, false},
      {28, 24, 38, 34, 1, true},    {44, 10, 53, 21, 2, false},   {26, 42, 36, 52, 2, false},
      {43, 40, 53, 52, 1, true},    {-21, 26, -12, 35, 1, true},  {-38, 24, -28, 34, 2, false},
      {-53, 12, -44, 21, 1, false}, {-36, 42, -26, 52, 1, false}, {-53, 40, -43, 52, 2, false},
  };
  uint32_t seed = 100;
  for (const H &h : kHouses) {
    const float cx = (h.x0 + h.x1) * 0.5f, cz = (h.z0 + h.z1) * 0.5f;
    int dx = 0, dz = 0;
    if (std::abs(cz) <= std::abs(cx)) dz = cz < 0 ? 1 : -1; // toward the east-west road
    else dx = cx < 0 ? 1 : -1;                              // toward the south road / castle axis
    b.structure(townmicro::house({h.x0, h.z0, h.x1, h.z1, h.floors, h.red, seed++, dx, dz}));
    // Door block and a path from it to the road.
    const int W = h.x1 - h.x0 + 1, D = h.z1 - h.z0 + 1;
    int doorX, doorZ;
    if (dz != 0) {
      doorX = h.x0 + W / 2;
      doorZ = dz > 0 ? h.z1 + 1 : h.z0 - 1;
    } else {
      doorZ = h.z0 + D / 2;
      doorX = dx > 0 ? h.x1 + 1 : h.x0 - 1;
    }
    b.prop("flower_pot", doorX + (dz != 0 ? -1.5f : 0.5f), 1.0f, doorZ + (dx != 0 ? -1.5f : 0.5f));
    if (rand01(h.x0, 0, h.z0, seed) < 0.6f) b.prop("barrel", doorX + (dz != 0 ? 3.5f : 0.5f), 1.0f, doorZ + (dx != 0 ? 3.5f : 0.5f));
    for (int s = 0, px = doorX, pz = doorZ; s < 40; ++s, px += dx, pz += dz) {
      const BlockId g = b.get(px, 0, pz);
      if (s > 0 && (g != vb::Grass || !b.air(px, 1, pz))) break;
      if (g != vb::Grass) continue;
      b.set(px, 0, pz, vb::Path);
      if (dx != 0 && b.get(px, 0, pz - 1) == vb::Grass) b.set(px, 0, pz - 1, vb::Path);
      if (dz != 0 && b.get(px - 1, 0, pz) == vb::Grass) b.set(px - 1, 0, pz, vb::Path);
    }
  }
}

// Market stalls along the west road, both sides.
void buildMarket(Builder &b) {
  uint32_t seed = 300;
  for (int x0 : {-27, -34, -41, -48})
    for (int side : {-1, 1}) {
      b.structure(townmicro::marketStall(x0, side, seed++));
      b.prop("coin_sign", x0 + 2.5f, 4.6f, float(side * 4) + 0.5f);
      b.prop("barrel", x0 - 0.5f, 1.0f, side * 6.5f + 0.5f);
    }
}

// Trees, bushes and flowers on free grass inside the wall.
void buildGardens(Builder &b) {
  auto clearGrass = [&](int cx, int cz, int r, int h) {
    for (int z = cz - r; z <= cz + r; ++z)
      for (int x = cx - r; x <= cx + r; ++x) {
        if (b.get(x, 0, z) != vb::Grass) return false;
        for (int y = 1; y <= h; ++y)
          if (!b.air(x, y, z)) return false;
      }
    return true;
  };
  for (int gz = -kWall + 6; gz <= kWall - 6; gz += 7)
    for (int gx = -kWall + 6; gx <= kWall - 6; gx += 7) {
      const int x = gx + int(hash3(gx, 0, gz, 31) % 3) - 1, z = gz + int(hash3(gx, 1, gz, 31) % 3) - 1;
      if (clearGrass(x, z, 3, 9)) { // round town tree (not choppable)
        const int th = 3 + int(hash3(x, 0, z, 32) % 2);
        b.box(x, 1, z, x, th, z, vb::TownLog);
        b.discEach(x + 0.5f, z + 0.5f, 3.2f, [&](int bx, int bz, float d) {
          for (int y = th; y <= th + 4; ++y) {
            const float dy = float(y - (th + 2)) * 1.1f;
            if (d * d + dy * dy <= 3.1f * 3.1f && b.air(bx, y, bz)) b.set(bx, y, bz, vb::HedgeLeaves);
          }
        });
      } else if (clearGrass(x, z, 1, 2) && rand01(x, 0, z, 33) < 0.7f) {
        b.prop("bush", x + 0.5f, 1.0f, z + 0.5f, float(hash3(x, 2, z, 34) % 628) * 0.01f);
      }
    }
  for (int z = -kWall + 2; z <= kWall - 2; ++z)
    for (int x = -kWall + 2; x <= kWall - 2; ++x)
      if (b.get(x, 0, z) == vb::Grass && b.air(x, 1, z) && rand01(x, 1, z, 35) < 0.02f)
        b.set(x, 1, z, rand01(x, 2, z, 36) < 0.5f ? vb::FlowersRed : vb::FlowersYellow);
}

void buildRoadLamps(Builder &b) {
  for (int t = 26; t <= 54; t += 9)
    for (int s : {-1, 1}) {
      b.prop("street_lamp", float(t) + 0.5f, 1.0f, s * 4.5f + 0.5f); // east road
      b.prop("street_lamp", s * 4.5f + 0.5f, 1.0f, float(t) + 0.5f); // south road
      if (t > 52) b.prop("street_lamp", float(-t) + 0.5f, 1.0f, s * 4.5f + 0.5f);
    }
}

std::shared_ptr<const TownGrid> buildTown() {
  auto grid = std::make_shared<TownGrid>();
  Builder b{*grid};
  buildGround(b);
  buildSquare(b);
  buildCastle(b);
  buildWalls(b);
  buildHouses(b);
  buildMarket(b);
  buildRoadLamps(b);
  buildGardens(b);

  const int R = TownGrid::R;
  for (int z = -R; z <= R; ++z)
    for (int x = -R; x <= R; ++x)
      for (int y = TownGrid::H - 1; y >= 0; --y) {
        const BlockId id = grid->get(x, y, z);
        if (id != kAir && id != vb::Water && id != vb::Light) {
          grid->top[TownGrid::col(x, z)] = uint8_t(y);
          break;
        }
      }
  return grid;
}

int baseHeight(const TownLayout &town, const atm::voxel::WorldGenerator &gen, int32_t x, int32_t z) {
  const int rx = x - town.centerX, rz = z - town.centerZ;
  const int r = std::max(std::abs(rx), std::abs(rz));
  if (r <= town.flatRadius) return town.groundY;
  const int natural = gen.surfaceHeight(x, z);
  if (r >= town.blendRadius) return natural;
  const float t = float(r - town.flatRadius) / float(town.blendRadius - town.flatRadius);
  const float s = t * t * (3.0f - 2.0f * t);
  return int(std::lround(float(town.groundY) * (1.0f - s) + float(natural) * s));
}

} // namespace

TownLayout townLayout(const atm::voxel::WorldGenerator &gen) {
  TownLayout t;
  // Average the natural ground around the centre, keep it above the sea.
  int sum = 0, n = 0;
  for (int dz = -32; dz <= 32; dz += 8)
    for (int dx = -32; dx <= 32; dx += 8) {
      sum += gen.surfaceHeight(t.centerX + dx, t.centerZ + dz);
      ++n;
    }
  t.groundY = std::clamp(sum / std::max(n, 1), kSeaLevel + 3, 140);
  t.grid = buildTown();
  return t;
}

int townSurfaceHeight(const TownLayout &town, const atm::voxel::WorldGenerator &gen, int32_t x, int32_t z) {
  const int rx = x - town.centerX, rz = z - town.centerZ;
  if (town.grid && std::max(std::abs(rx), std::abs(rz)) <= std::min(town.flatRadius, TownGrid::R))
    return town.groundY + town.grid->top[TownGrid::col(rx, rz)];
  return baseHeight(town, gen, x, z);
}

bool inTown(const TownLayout &town, double x, double z) {
  return std::abs(x - town.centerX) <= town.flatRadius + 0.5 && std::abs(z - town.centerZ) <= town.flatRadius + 0.5;
}

void townSpawnPoint(const TownLayout &town, double &x, double &y, double &z) {
  x = town.centerX + 0.5;
  z = town.centerZ + 10.5; // south of the fountain, on the square
  y = town.groundY + 1.05;
}

const std::vector<TownProp> &townProps(const TownLayout &town) {
  static const std::vector<TownProp> kNone;
  return town.grid ? town.grid->props : kNone;
}

const std::vector<MicroModel> &townStructures(const TownLayout &town) {
  static const std::vector<MicroModel> kNone;
  return town.grid ? town.grid->structures : kNone;
}

bool townMapColor(const TownLayout &town, int32_t x, int32_t z, uint32_t &rgba) {
  const int rx = x - town.centerX, rz = z - town.centerZ;
  if (!town.grid || std::abs(rx) > TownGrid::R || std::abs(rz) > TownGrid::R) return false;
  const uint32_t c = town.grid->mapColor[TownGrid::col(rx, rz)];
  if (!c) return false;
  rgba = c;
  return true;
}

void stampTown(const TownLayout &town, const atm::voxel::WorldGenerator &gen, atm::voxel::ChunkCoord c,
               atm::voxel::Chunk &chunk) {
  constexpr int S = atm::voxel::kChunkSize;
  const int32_t x0 = c.x * S, y0 = c.y * S, z0 = c.z * S;
  const int R = town.blendRadius;
  if (x0 + S - 1 < town.centerX - R || x0 > town.centerX + R || z0 + S - 1 < town.centerZ - R ||
      z0 > town.centerZ + R)
    return;
  for (int lz = 0; lz < S; ++lz)
    for (int lx = 0; lx < S; ++lx) {
      const int32_t x = x0 + lx, z = z0 + lz;
      const int rx = x - town.centerX, rz = z - town.centerZ;
      const int r = std::max(std::abs(rx), std::abs(rz));
      if (r > R) continue;
      const int ground = baseHeight(town, gen, x, z);
      const bool flat = r <= town.flatRadius && town.grid;
      const int natural = gen.surfaceHeight(x, z);
      const int clearTop = std::max(natural + 12, ground + (flat ? TownGrid::H : 24));
      for (int ly = 0; ly < S; ++ly) {
        const int y = y0 + ly;
        if (y > clearTop || y < ground - 6) continue;
        BlockId b;
        if (flat && y >= ground && y < ground + TownGrid::H) b = town.grid->get(rx, y - ground, rz);
        else if (y > ground) b = kAir; // clear hills and wild trees
        else if (y == ground) b = vb::Grass;
        else if (y >= ground - 3) b = vb::Dirt;
        else b = vb::Stone;
        chunk.set(lx, ly, lz, b);
      }
    }
}

const TownLayout &homeTown(const atm::voxel::WorldGenerator &gen) {
  static std::mutex m;
  static std::unordered_map<uint64_t, std::unique_ptr<TownLayout>> cache;
  std::lock_guard lock(m);
  auto &slot = cache[gen.seed()];
  if (!slot) slot = std::make_unique<TownLayout>(townLayout(gen));
  return *slot;
}

int groundHeight(const atm::voxel::WorldGenerator &gen, int32_t x, int32_t z) {
  const TownLayout &t = homeTown(gen);
  if (std::max(std::abs(x - t.centerX), std::abs(z - t.centerZ)) >= t.blendRadius) return gen.surfaceHeight(x, z);
  return townSurfaceHeight(t, gen, x, z);
}

void installHomeTown(atm::voxel::VoxelWorldConfig &wc) {
  auto gen = std::make_shared<atm::voxel::WorldGenerator>(wc.seed);
  const TownLayout town = homeTown(*gen);
  wc.postGenerate = [gen, town](atm::voxel::ChunkCoord c, atm::voxel::Chunk &ch) { stampTown(town, *gen, c, ch); };
}

} // namespace ao::world
