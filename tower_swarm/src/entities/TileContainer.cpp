#include "entities/TileContainer.h"

#include "Constants.h"

#include <algorithm>

namespace tower_swarm {

TileContainer::TileContainer(int type_id, std::uint8_t default_layer,
                             int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity) {}

EntityHandle TileContainer::createTile(float x, float y, int texture_id,
                                       int tile_size_px) {
  const EntityHandle id = RenderableEntityContainer::createEntity();
  if (id == INVALID_ID) {
    return INVALID_ID;
  }

  const std::uint32_t slot = getSlot(id);
  x_positions[slot] = x;
  y_positions[slot] = y;
  widths[slot] = static_cast<std::int16_t>(tile_size_px);
  heights[slot] = static_cast<std::int16_t>(tile_size_px);
  texture_ids[slot] = static_cast<std::int16_t>(texture_id);
  z_indices[slot] = kZIndexTiles;
  flags[slot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);
  return id;
}

void TileContainer::buildSolidTilemap(int cols, int rows, int tile_size_px,
                                      int texture_id) {
  cols = std::max(cols, 0);
  rows = std::max(rows, 0);
  if (cols == 0 || rows == 0 || tile_size_px <= 0) {
    return;
  }

  const int needed = cols * rows;
  if (needed > capacity) {
    resizeArrays(needed);
  }

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      const float wx = static_cast<float>(x * tile_size_px);
      const float wy = static_cast<float>(y * tile_size_px);
      createTile(wx, wy, texture_id, tile_size_px);
    }
  }
}

void TileContainer::update(float delta_time) {
  (void)delta_time;
}

} // namespace tower_swarm

