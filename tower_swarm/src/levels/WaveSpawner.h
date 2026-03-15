#pragma once

#include "entities/EnemyType.h"
#include "levels/LevelDefinition.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct Engine;

namespace tower_swarm {

class EnemyContainer;

class WaveSpawner final {
public:
  WaveSpawner() = default;

  void bind(Engine *engine, EnemyContainer *enemies) {
    engine_ = engine;
    enemies_ = enemies;
  }

  void beginLevel(const LevelDefinition &def, std::uint32_t seed);
  void startWave(std::int32_t wave_index, bool spawn_boss, bool is_elite,
                 float extra_hp_multiplier = 1.0f,
                 float extra_speed_multiplier = 1.0f,
                 float initial_delay_sec = 0.0f);
  void tick(float dt);

  bool isSpawning() const { return spawning_; }
  bool isDoneSpawning() const { return spawning_ && !hasPending(); }

  std::int32_t currentWave() const { return wave_index_; }
  std::int32_t waveCount() const { return def_.wave_count; }
  bool isBossWave() const;

  float interSpawnDelaySec() const { return inter_spawn_delay_sec_; }
  void setInterSpawnDelaySec(float sec);

private:
  struct SpawnBatch final {
    EnemyType type{EnemyType::Grub};
    std::int32_t remaining{0};
    float extra_hp_mult{1.0f};
    float extra_speed_mult{1.0f};
  };

  bool hasPending() const { return batch_cursor_ < batches_.size(); }
  void buildWaveBatches();
  void spawnOne();

  void nextRandom(std::uint32_t &state) const;
  float random01(std::uint32_t &state) const;
  std::uint32_t randomU32(std::uint32_t &state) const;

  void pickSpawnPosition(float &out_x, float &out_y);

  Engine *engine_{nullptr};
  EnemyContainer *enemies_{nullptr};

  LevelDefinition def_{};
  std::int32_t wave_index_{0};
  bool wave_spawn_boss_{true};
  bool wave_is_elite_{false};
  float wave_extra_hp_mult_{1.0f};
  float wave_extra_speed_mult_{1.0f};
  bool spawning_{false};
  float spawn_timer_sec_{0.0f};
  float inter_spawn_delay_sec_{0.0f};

  std::uint32_t rng_state_{0x12345678u};

  std::vector<SpawnBatch> batches_{};
  std::size_t batch_cursor_{0};
};

} // namespace tower_swarm
