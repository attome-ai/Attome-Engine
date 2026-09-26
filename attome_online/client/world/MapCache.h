#pragma once

// Explored-terrain colours for the minimap and world map. Scans a few
// thousand columns per frame around the player (nearest first) and stores
// the top block's colour per column, hill-shaded and water-depth tinted, in
// 32x32 tiles. Columns are re-scanned when the player returns after a while,
// so edits show up eventually. Client-side only.

#include "../../../engine/voxel/BlockRegistry.h"
#include "../../../engine/voxel/VoxelWorld.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ao::client {

class MapCache {
public:
  static constexpr int kTile = 32; // columns per tile side

  struct Tile {
    uint32_t color[kTile * kTile] = {}; // bytes R,G,B,A; alpha 0 = not scanned
    uint16_t scanned = 0;
  };

  // Scans columns within `radius` blocks of the player for at most
  // `budgetMs` milliseconds (time-sliced: never a frame-time spike).
  void update(const atm::voxel::VoxelWorld &world, const atm::voxel::BlockRegistry &blocks,
              const glm::dvec3 &player, int radius, float budgetMs, float dt);
  // Colour of column (x, z); alpha 0 when unexplored.
  uint32_t colorAt(int32_t x, int32_t z) const;
  const std::unordered_map<uint64_t, Tile> &tiles() const { return tiles_; }
  static int32_t tileOf(int32_t v) { return v >= 0 ? v / kTile : (v + 1) / kTile - 1; }
  static int32_t tileX(uint64_t key) { return int32_t(uint32_t(key >> 32)); }
  static int32_t tileZ(uint64_t key) { return int32_t(uint32_t(key)); }

private:
  static uint64_t key(int32_t tx, int32_t tz) { return (uint64_t(uint32_t(tx)) << 32) | uint32_t(tz); }
  uint32_t scanColumn(const atm::voxel::VoxelWorld &world, const atm::voxel::BlockRegistry &blocks,
                      int32_t x, int32_t z) const;

  std::unordered_map<uint64_t, Tile> tiles_;
  std::vector<glm::ivec2> ring_;  // offsets sorted by distance (built once per radius)
  int ringRadius_ = -1;
  size_t cursor_ = 0;             // next ring offset to scan
  bool idle_ = false;             // full pass done: rest until the player moves / timer
  float idleTime_ = 0.0f;
  glm::ivec2 center_{0, 0};
};

} // namespace ao::client
