#pragma once

#include "Constants.h"
#include "entities/EnemyType.h"

#include <cstdint>

namespace tower_swarm {

struct LevelDefinition final {
  std::int32_t level_number{1};
  std::int32_t wave_count{0};
  float difficulty{1.0f};
  std::int32_t map_variant{0};
  Biome biome{Biome::VerdantFields};
  bool is_elite{false};
  EnemyType boss_type{EnemyType::SiegeLord};
};

} // namespace tower_swarm

