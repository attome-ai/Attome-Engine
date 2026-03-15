#pragma once

#include "ATMEngine.h"

#include <cstdint>

struct Engine;

namespace tower_swarm {

class EnemyContainer;
class CreatureContainer;
struct GameState;

class ProjectileContainer final : public RenderableEntityContainer {
public:
  explicit ProjectileContainer(Engine *engine, int type_id,
                               std::uint8_t default_layer, int initial_capacity);

  void bindEnemies(EnemyContainer *enemies) { enemies_ = enemies; }
  void bindCreatures(CreatureContainer *creatures) { creatures_ = creatures; }
  void bindGameState(GameState *state) { game_state_ = state; }
  void setTexture(int texture_id) { texture_id_ = texture_id; }

  EntityHandle spawnProjectile(float x, float y, float vx, float vy, float damage,
                               float lifetime_sec, int pierce_count,
                               float splash_radius_px, EntityHandle source_creature,
                               int texture_id_override = -1,
                               int chain_remaining = 0,
                               float chain_range_px = 0.0f);

  void update(float delta_time) override;

  DynamicArray<float> vx;
  DynamicArray<float> vy;
  DynamicArray<float> damage;
  DynamicArray<float> age_sec;
  DynamicArray<float> lifetime_sec;
  DynamicArray<std::int16_t> pierce_remaining;
  DynamicArray<float> splash_radius_px;
  DynamicArray<EntityHandle> source_creature;
  DynamicArray<EntityHandle> last_hit_enemy;
  DynamicArray<std::int16_t> chain_remaining;
  DynamicArray<float> chain_range_px;

protected:
  void swapSlots(std::uint32_t a, std::uint32_t b) override;
  void resizeArrays(int new_capacity) override;

private:
  void schedule_destroy(std::uint32_t slot);
  bool try_hit(std::uint32_t slot);

  Engine *engine_{nullptr};
  EnemyContainer *enemies_{nullptr};
  CreatureContainer *creatures_{nullptr};
  GameState *game_state_{nullptr};
  int texture_id_{0};
};

} // namespace tower_swarm
