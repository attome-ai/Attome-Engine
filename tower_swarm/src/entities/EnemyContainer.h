#pragma once

#include "ATMEngine.h"
#include "entities/EnemyType.h"

#include <array>
#include <cstdint>
#include <vector>

struct Engine;

namespace tower_swarm {

class BaseEntity;
class CreatureContainer;
class PickupContainer;
struct GameState;

class EnemyContainer final : public RenderableEntityContainer {
public:
  explicit EnemyContainer(Engine *engine, int type_id, std::uint8_t default_layer,
                          int initial_capacity);

  void bindBase(BaseEntity *base, EntityHandle base_id) {
    base_ = base;
    base_id_ = base_id;
  }
  void bindCreatures(CreatureContainer *creatures) { creatures_ = creatures; }
  void bindPickups(PickupContainer *pickups) { pickups_ = pickups; }
  void bindGameState(GameState *state) { game_state_ = state; }
  void setEnemyTextures(
      const std::array<int, static_cast<std::size_t>(EnemyType::Count)>
          *textures) {
    textures_ = textures;
  }

  EntityHandle spawnEnemy(EnemyType type, float x, float y, int level_number,
                          int wave_number, bool is_elite,
                          float extra_hp_multiplier = 1.0f,
                          float extra_speed_multiplier = 1.0f);

  bool applyDamage(EntityHandle id, float damage, EntityHandle source_creature);
  void scheduleDelayedDamage(EntityHandle target, float delay_sec, float damage,
                             EntityHandle source_creature);
  void scheduleDelayedKill(EntityHandle target, float delay_sec,
                           EntityHandle source_creature);
  void displace(EntityHandle id, float dx, float dy);
  EnemyType getEnemyType(EntityHandle id) const;
  float getHp(EntityHandle id) const;
  float getHpMax(EntityHandle id) const;

  void update(float delta_time) override;

  // SoA gameplay arrays.
  DynamicArray<EnemyType> enemy_type;
  DynamicArray<float> hp;
  DynamicArray<float> hp_max;
  DynamicArray<float> move_speed_px_per_sec;
  DynamicArray<float> damage_to_base;
  DynamicArray<std::int32_t> reward_essence;
  DynamicArray<float> slow_multiplier;
  DynamicArray<float> slow_time_sec;
  DynamicArray<float> frozen_time_sec;
  DynamicArray<std::uint8_t> is_child;
  DynamicArray<std::int32_t> spawn_level;
  DynamicArray<std::int32_t> spawn_wave;
  DynamicArray<std::uint8_t> spawn_is_elite;
  DynamicArray<float> spawn_extra_hp_mult;
  DynamicArray<float> spawn_extra_speed_mult;
  DynamicArray<std::uint8_t> boss_phase;
  DynamicArray<float> boss_stomp_cooldown_sec;

  DynamicArray<float> aura_speed_multiplier;
  DynamicArray<float> aura_damage_multiplier;
  DynamicArray<float> zone_speed_multiplier;
  DynamicArray<float> zone_damage_multiplier;
  DynamicArray<float> zone_heal_multiplier;
  DynamicArray<EntityHandle> kill_credit_override;

protected:
  void swapSlots(std::uint32_t a, std::uint32_t b) override;
  void resizeArrays(int new_capacity) override;

private:
  struct DelayedDamage final {
    EntityHandle target{INVALID_ID};
    float time_sec{0.0f};
    float damage{0.0f};
    EntityHandle source_creature{INVALID_ID};
  };

  void schedule_destroy(std::uint32_t slot);
  void on_death(std::uint32_t slot, EntityHandle source_creature);
  bool is_front_hit(std::uint32_t slot, EntityHandle source_creature) const;
  bool applyDamageInternal(EntityHandle id, float damage, EntityHandle source_creature,
                           bool is_echo);

  Engine *engine_{nullptr};
  BaseEntity *base_{nullptr};
  EntityHandle base_id_{INVALID_ID};
  CreatureContainer *creatures_{nullptr};
  PickupContainer *pickups_{nullptr};
  GameState *game_state_{nullptr};
  const std::array<int, static_cast<std::size_t>(EnemyType::Count)>
      *textures_{nullptr};

  std::vector<DelayedDamage> delayed_damage_{};
};

} // namespace tower_swarm
