#pragma once

// Internal helpers for the procedural model content (not part of the public
// AttomeModel API).

#include "Character.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace atm::model::content_detail {

// RGBA8 packed with R in the low byte (memory order R, G, B, A), the same
// convention as BlockDef colours and render::Material.
constexpr uint32_t rgba(int r, int g, int b, int a = 255) {
  return uint32_t(r & 255) | (uint32_t(g & 255) << 8) | (uint32_t(b & 255) << 16) |
         (uint32_t(a & 255) << 24);
}

// Small voxel painter. Boxes are half-open [x0, x1) x [y0, y1) x [z0, z1) and
// silently clipped to the part.
struct Painter {
  VoxelPart part;

  Painter(std::string name, int sx, int sy, int sz, glm::vec3 pivot) {
    part.name = std::move(name);
    part.sx = std::clamp(sx, 1, 32);
    part.sy = std::clamp(sy, 1, 32);
    part.sz = std::clamp(sz, 1, 32);
    part.voxels.assign(size_t(part.sx) * size_t(part.sy) * size_t(part.sz), 0);
    part.palette = {0u};
    part.pivot = pivot;
  }

  // Plain colour. Add all plain colours before the first glow colour: every
  // palette index >= emissiveFrom glows.
  int c(uint32_t color) {
    for (size_t i = 1; i < part.palette.size() && i < part.emissiveFrom; ++i)
      if (part.palette[i] == color)
        return int(i);
    if (part.palette.size() >= 255)
      return 1;
    part.palette.push_back(color);
    return int(part.palette.size() - 1);
  }
  // Emissive colour.
  int g(uint32_t color) {
    if (part.palette.size() >= 255)
      return 1;
    part.palette.push_back(color);
    const int idx = int(part.palette.size() - 1);
    if (part.emissiveFrom == 255)
      part.emissiveFrom = uint8_t(idx);
    return idx;
  }

  void set(int x, int y, int z, int idx) {
    if (x < 0 || y < 0 || z < 0 || x >= part.sx || y >= part.sy || z >= part.sz)
      return;
    part.at(x, y, z) = uint8_t(idx);
  }
  void box(int x0, int y0, int z0, int x1, int y1, int z1, int idx) {
    x0 = std::max(x0, 0), y0 = std::max(y0, 0), z0 = std::max(z0, 0);
    x1 = std::min(x1, part.sx), y1 = std::min(y1, part.sy), z1 = std::min(z1, part.sz);
    for (int y = y0; y < y1; ++y)
      for (int z = z0; z < z1; ++z)
        for (int x = x0; x < x1; ++x)
          part.at(x, y, z) = uint8_t(idx);
  }
  void clearBox(int x0, int y0, int z0, int x1, int y1, int z1) { box(x0, y0, z0, x1, y1, z1, 0); }
  // Replaces non-empty voxels in the box with `idx` where a deterministic hash
  // of the position is below `chance` (0..1): two-tone speckle texture.
  void speckle(int x0, int y0, int z0, int x1, int y1, int z1, int idx, float chance, uint32_t seed) {
    x0 = std::max(x0, 0), y0 = std::max(y0, 0), z0 = std::max(z0, 0);
    x1 = std::min(x1, part.sx), y1 = std::min(y1, part.sy), z1 = std::min(z1, part.sz);
    for (int y = y0; y < y1; ++y)
      for (int z = z0; z < z1; ++z)
        for (int x = x0; x < x1; ++x) {
          if (part.at(x, y, z) == 0)
            continue;
          uint32_t h = seed ^ (uint32_t(x) * 73856093u) ^ (uint32_t(y) * 19349663u) ^ (uint32_t(z) * 83492791u);
          h ^= h >> 13;
          h *= 0x5bd1e995u;
          h ^= h >> 15;
          if (float(h & 0xFFFFu) / 65535.0f < chance)
            part.at(x, y, z) = uint8_t(idx);
        }
  }
  // Removes the 12 edges' voxels for a rounded look.
  void roundEdges() {
    for (int y = 0; y < part.sy; ++y)
      for (int z = 0; z < part.sz; ++z)
        for (int x = 0; x < part.sx; ++x) {
          const int ex = (x == 0 || x == part.sx - 1) ? 1 : 0;
          const int ey = (y == 0 || y == part.sy - 1) ? 1 : 0;
          const int ez = (z == 0 || z == part.sz - 1) ? 1 : 0;
          if (ex + ey + ez >= 2)
            part.at(x, y, z) = 0;
        }
  }
};

} // namespace atm::model::content_detail
