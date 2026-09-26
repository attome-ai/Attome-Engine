#include "Micro.h"

#include <algorithm>
#include <cmath>

namespace ao::world {

namespace {
constexpr uint32_t rgb(int r, int g, int b) { return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | 0xFF000000u; }
} // namespace

const std::vector<uint32_t> &microPalette() {
  static const std::vector<uint32_t> p = [] {
    std::vector<uint32_t> t(mc::Count, rgb(255, 0, 255));
    t[mc::Empty] = 0;
    t[mc::Stone] = rgb(240, 234, 220);
    t[mc::StoneShade] = rgb(222, 214, 198);
    t[mc::StoneTrim] = rgb(192, 188, 178);
    t[mc::StoneDark] = rgb(150, 146, 140);
    t[mc::Gold] = rgb(252, 204, 72);
    t[mc::GoldDark] = rgb(214, 164, 48);
    t[mc::RoofBlue] = rgb(66, 108, 204);
    t[mc::RoofBlueDark] = rgb(44, 74, 160);
    t[mc::RoofBlueLight] = rgb(104, 152, 232);
    t[mc::RoofRed] = rgb(204, 78, 60);
    t[mc::RoofRedDark] = rgb(160, 56, 46);
    t[mc::Plaster] = rgb(244, 236, 216);
    t[mc::PlasterShade] = rgb(230, 220, 198);
    t[mc::WoodDark] = rgb(108, 72, 44);
    t[mc::Wood] = rgb(162, 112, 66);
    t[mc::WoodLight] = rgb(216, 172, 112);
    t[mc::BannerBlue] = rgb(44, 78, 196);
    t[mc::BannerBlueDark] = rgb(30, 54, 150);
    t[mc::AwningRed] = rgb(224, 60, 56);
    t[mc::AwningWhite] = rgb(248, 244, 234);
    t[mc::Iron] = rgb(70, 72, 84);
    t[mc::Glass] = rgb(150, 196, 226);
    t[mc::Moss] = rgb(112, 152, 92);
    t[mc::Leaf] = rgb(92, 196, 72);
    t[mc::LeafDark] = rgb(64, 150, 56);
    t[mc::FlowerRed] = rgb(236, 70, 96);
    t[mc::FlowerYellow] = rgb(252, 214, 72);
    t[mc::Cobble] = rgb(160, 158, 152);
    t[mc::Teal] = rgb(70, 196, 196);
    t[mc::Orange] = rgb(240, 150, 60);
    t[mc::SunYellow] = rgb(255, 232, 80);
    t[mc::SunGold] = rgb(250, 190, 40);
    t[mc::Water] = rgb(70, 160, 236);
    t[mc::StoneWarm] = rgb(232, 220, 196);
    t[mc::Slate] = rgb(96, 104, 124);
    t[mc::Cream] = rgb(250, 244, 226);
    t[mc::GlowWindow] = rgb(255, 214, 124);
    t[mc::GlowLantern] = rgb(255, 196, 88);
    t[mc::Crystal] = rgb(96, 200, 255);
    t[mc::CrystalBright] = rgb(200, 244, 255);
    t[mc::GlowGold] = rgb(255, 220, 110);
    return t;
  }();
  return p;
}

void MicroModel::box(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c) {
  if (x0 > x1) std::swap(x0, x1);
  if (y0 > y1) std::swap(y0, y1);
  if (z0 > z1) std::swap(z0, z1);
  x0 = std::max(x0, 0), y0 = std::max(y0, 0), z0 = std::max(z0, 0);
  x1 = std::min(x1, sx - 1), y1 = std::min(y1, sy - 1), z1 = std::min(z1, sz - 1);
  for (int y = y0; y <= y1; ++y)
    for (int z = z0; z <= z1; ++z) {
      uint8_t *row = &v[(size_t(y) * sz + z) * sx];
      for (int x = x0; x <= x1; ++x) row[x] = c;
    }
}

void MicroModel::speckle(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t c, float chance, uint32_t seed) {
  if (x0 > x1) std::swap(x0, x1);
  if (y0 > y1) std::swap(y0, y1);
  if (z0 > z1) std::swap(z0, z1);
  for (int y = std::max(y0, 0); y <= std::min(y1, sy - 1); ++y)
    for (int z = std::max(z0, 0); z <= std::min(z1, sz - 1); ++z)
      for (int x = std::max(x0, 0); x <= std::min(x1, sx - 1); ++x) {
        uint8_t &p = v[(size_t(y) * sz + z) * sx + x];
        if (!p) continue;
        uint32_t h = seed ^ (uint32_t(x) * 73856093u) ^ (uint32_t(y) * 19349663u) ^ (uint32_t(z) * 83492791u);
        h ^= h >> 13;
        h *= 0x5bd1e995u;
        h ^= h >> 15;
        if (float(h & 0xFFFFu) / 65535.0f < chance) p = c;
      }
}

void MicroModel::recolor(int x0, int y0, int z0, int x1, int y1, int z1, uint8_t from, uint8_t to) {
  if (x0 > x1) std::swap(x0, x1);
  if (y0 > y1) std::swap(y0, y1);
  if (z0 > z1) std::swap(z0, z1);
  for (int y = std::max(y0, 0); y <= std::min(y1, sy - 1); ++y)
    for (int z = std::max(z0, 0); z <= std::min(z1, sz - 1); ++z)
      for (int x = std::max(x0, 0); x <= std::min(x1, sx - 1); ++x) {
        uint8_t &p = v[(size_t(y) * sz + z) * sx + x];
        if (p == from) p = to;
      }
}

void MicroModel::cylinder(float cx, float cz, float r, int y0, int y1, uint8_t c, float thick) {
  const int R = int(std::ceil(r)) + 1;
  for (int z = int(cz) - R; z <= int(cz) + R; ++z)
    for (int x = int(cx) - R; x <= int(cx) + R; ++x) {
      const float dx = x + 0.5f - cx, dz = z + 0.5f - cz;
      const float d = std::sqrt(dx * dx + dz * dz);
      if (d > r || (thick > 0.0f && d <= r - thick)) continue;
      for (int y = y0; y <= y1; ++y) set(x, y, z, c);
    }
}

size_t MicroModel::filled() const {
  return size_t(std::count_if(v.begin(), v.end(), [](uint8_t c) { return c != 0; }));
}

} // namespace ao::world
