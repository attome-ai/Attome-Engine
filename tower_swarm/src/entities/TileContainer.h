#pragma once

#include "ATMEngine.h"

#include <cstdint>

namespace tower_swarm {

class TileContainer final : public RenderableEntityContainer {
public:
  TileContainer(int type_id, std::uint8_t default_layer, int initial_capacity);

  EntityHandle createTile(float x, float y, int texture_id, int tile_size_px);
  void buildSolidTilemap(int cols, int rows, int tile_size_px, int texture_id);

  void update(float delta_time) override;
};

} // namespace tower_swarm

