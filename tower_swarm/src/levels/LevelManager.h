#pragma once

#include "levels/LevelDefinition.h"
#include "levels/WaveSpawner.h"

#include <array>
#include <cstdint>

struct Engine;
typedef std::uint32_t EntityHandle;

namespace tower_swarm {

class BaseEntity;
class EnemyContainer;
struct GameState;

enum class LevelManagerState : std::uint8_t {
  Playing = 0,
  WaveClear = 1,
  LevelClear = 2,
  Failed = 3,
};

class LevelManager final {
public:
  void bind(Engine *engine, BaseEntity *base, EntityHandle base_id,
            EnemyContainer *enemies, GameState *game_state);

  void beginLevel(std::int32_t level_number);
  void tick(float dt);

  LevelManagerState state() const { return state_; }
  const LevelDefinition &levelDef() const { return def_; }

  std::int32_t waveIndex() const { return wave_index_; }
  std::int32_t waveCount() const { return def_.wave_count; }

  float graceRemainingSec() const { return grace_remaining_sec_; }
  bool isBossWave() const;

  const char *bannerText() const;
  float bannerRemainingSec() const { return banner_remaining_sec_; }

  std::int32_t lastLevelStars() const { return last_level_stars_; }

private:
  void setBanner(const char *fmt, ...);
  void enterWaveClear(std::int32_t cleared_wave);
  void enterLevelClear();
  void enterFailed();
  std::int32_t countVisibleEnemies() const;
  std::int32_t computeStars() const;

  Engine *engine_{nullptr};
  BaseEntity *base_{nullptr};
  EntityHandle base_id_{0xFFFFFFFFu};
  EnemyContainer *enemies_{nullptr};
  GameState *game_state_{nullptr};

  LevelDefinition def_{};
  WaveSpawner spawner_{};

  LevelManagerState state_{LevelManagerState::Playing};
  std::int32_t wave_index_{0};
  float grace_remaining_sec_{0.0f};

  std::array<char, 128> banner_{};
  float banner_remaining_sec_{0.0f};
  std::int32_t last_level_stars_{0};
};

} // namespace tower_swarm
