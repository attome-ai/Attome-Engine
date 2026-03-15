#pragma once

#include "Constants.h"
#include "levels/LevelDefinition.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tower_swarm {

inline int sanitize_level_number(int level_number) {
  return std::max(1, level_number);
}

inline std::int32_t wave_count_for_level(std::int32_t level_number) {
  const std::int32_t n = sanitize_level_number(level_number);
  const float waves =
      static_cast<float>(level::kBaseWaveCount) +
      static_cast<float>(n) * level::kWaveCountPerLevel;
  const std::int32_t wave_count =
      static_cast<std::int32_t>(std::floor(waves));
  return std::max<std::int32_t>(1, wave_count);
}

inline float difficulty_for_level(std::int32_t level_number) {
  const std::int32_t n = sanitize_level_number(level_number);
  return std::pow(level::kDifficultyBase, static_cast<float>(n));
}

inline std::int32_t map_variant_for_level(std::int32_t level_number) {
  const std::int32_t n = sanitize_level_number(level_number);
  const std::int32_t m = std::max<std::int32_t>(1, level::kMapTemplateCount);
  return n % m;
}

inline Biome biome_for_level(std::int32_t level_number) {
  const std::int32_t n = sanitize_level_number(level_number);
  const std::int32_t per = std::max<std::int32_t>(1, level::kBiomeLevelsPer);
  const std::int32_t idx =
      ((n - 1) / per) % static_cast<std::int32_t>(Biome::Count);
  return static_cast<Biome>(std::clamp<std::int32_t>(
      idx, 0, static_cast<std::int32_t>(Biome::Count) - 1));
}

inline bool is_elite_level(std::int32_t level_number) {
  const std::int32_t n = sanitize_level_number(level_number);
  const std::int32_t every =
      std::max<std::int32_t>(1, level::kEliteEveryLevels);
  return (n % every) == 0;
}

inline std::int32_t wave_enemy_count(std::int32_t level_number,
                                     std::int32_t wave_index) {
  const std::int32_t n = sanitize_level_number(level_number);
  const std::int32_t w = std::max<std::int32_t>(0, wave_index);

  const float base =
      level::kWaveEnemyCountBase +
      static_cast<float>(n) * level::kWaveEnemyCountLinear +
      static_cast<float>(n) * static_cast<float>(n) * level::kWaveEnemyCountQuadratic;
  const float wave_factor = 1.0f + static_cast<float>(w) * level::kWaveEnemyCountWaveFactor;
  const std::int32_t count =
      static_cast<std::int32_t>(std::floor(base * wave_factor));
  return std::max<std::int32_t>(1, count);
}

inline bool is_boss_wave(std::int32_t wave_index, std::int32_t wave_count) {
  if (wave_count <= 0) {
    return false;
  }
  return wave_index == (wave_count - 1);
}

inline float between_wave_grace_timer_sec(std::int32_t level_number) {
  // Production TODO: max(3, 8 - floor(level/10)) seconds.
  const std::int32_t n = sanitize_level_number(level_number);
  const std::int32_t step = n / 10;
  const std::int32_t raw = 8 - step;
  const std::int32_t sec = std::max<std::int32_t>(3, raw);
  return static_cast<float>(sec);
}

inline LevelDefinition generate_level_definition(std::int32_t level_number) {
  LevelDefinition def{};
  def.level_number = sanitize_level_number(level_number);
  def.wave_count = wave_count_for_level(def.level_number);
  def.difficulty = difficulty_for_level(def.level_number);
  def.map_variant = map_variant_for_level(def.level_number);
  def.biome = biome_for_level(def.level_number);
  def.is_elite = is_elite_level(def.level_number);
  def.boss_type = EnemyType::SiegeLord;
  return def;
}

} // namespace tower_swarm
