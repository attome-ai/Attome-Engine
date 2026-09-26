#pragma once

// Fine-voxel structures (Trove-style detail): buildings, fountains, walls
// modelled at kMicro voxels per block, drawn as model meshes on the client.
// On the world grid they become invisible Barrier blocks (collision) and
// Light blocks (lamps), so gameplay stays block-based.
//
// Size on disk: structures are generated from code (a few KB of source per
// building type); authored ones can use the compact codec in
// engine/model/VoxelCodec.h (palette + run lengths + varints).

#include <cstdint>
#include <vector>

namespace ao::world {

inline constexpr int kMicro = 4; // fine voxels per block

// One shared palette for every structure (so the client uploads it once).
// Indices >= kMicroGlowFrom glow (windows, lanterns, crystals).
namespace mc {
enum : uint8_t {
  Empty = 0,
  Stone, StoneShade, StoneTrim, StoneDark, Gold, GoldDark, RoofBlue, RoofBlueDark, RoofBlueLight,
  RoofRed, RoofRedDark, Plaster, PlasterShade, WoodDark, Wood, WoodLight, BannerBlue, BannerBlueDark,
  AwningRed, AwningWhite, Iron, Glass, Moss, Leaf, LeafDark, FlowerRed, FlowerYellow, Cobble, Teal,
  Orange, SunYellow, SunGold, Water, StoneWarm, Slate, Cream,
  GlowWindow = 48, GlowLantern, Crystal, CrystalBright, GlowGold,
  Count
};
} // namespace mc
inline constexpr uint8_t kMicroGlowFrom = mc::GlowWindow;
const std::vector<uint32_t> &microPalette(); // RGBA8 (R low byte), mc::Count entries

// A fine-voxel model placed on the block grid: its (0,0,0) voxel is the
// bottom-north-west corner of block (bx, by, bz) (town-relative blocks, by
// relative to the town ground surface).
struct MicroModel {
  int bx = 0, by = 0, bz = 0;
  int sx = 0, sy = 0, sz = 0; // size in fine voxels
  std::vector<uint8_t> v;     // palette indices, index = (y * sz + z) * sx + x

  MicroModel() = default;
  MicroModel(int bx_, int by_, int bz_, int blocksX, int blocksY, int blocksZ)
      : bx(bx_), by(by_), bz(bz_), sx(blocksX * kMicro), sy(blocksY * kMicro), sz(blocksZ * kMicro),
        v(size_t(sx) * size_t(sy) * size_t(sz), 0) {}

  bool in(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < sx && y < sy && z < sz; }
  uint8_t at(int x, int y, int z) const { return in(x, y, z) ? v[(size_t(y) * sz + z) * sx + x] : 0; }
  void set(int x, int y, int z, uint8_t c) {
    if (in(x, y, z)) v[(size_t(y) * sz + z) * sx + x] = c;
  }
  // Inclusive box, corners in any order, clipped.
  void box(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c);
  // Replaces non-empty voxels in the box with c where a position hash < chance.
  void speckle(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c, float chance, uint32_t seed);
  // Replaces voxels of colour `from` in the box with `to`.
  void recolor(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t from, uint8_t to);
  // Vertical cylinder (or ring when thick > 0) centred at (cx, cz) in voxels.
  void cylinder(float cx, float cz, float r, int y0, int y1, uint8_t c, float thick = 0.0f);
  size_t filled() const;
};

} // namespace ao::world
