#pragma once

#include "ATMEngine.h"
#include "Constants.h"
#include "characters/CharacterId.h"

#include <array>
#include <cstdint>
#include <vector>

struct Engine;

namespace tower_swarm {

struct GameState;
class BaseEntity;
class PathGrid;

enum class CreatureState : std::uint8_t {
  Idle = 0,
  Attacking = 1,
  Moving = 2,
  Evolving = 3,
  Merging = 4,
  Dragging = 5
};

class CreatureContainer final : public RenderableEntityContainer {
public:
  explicit CreatureContainer(Engine *engine, int type_id,
                             std::uint8_t default_layer, int initial_capacity);

  void bindGameState(GameState *state) { game_state_ = state; }
  GameState *gameState() { return game_state_; }
  const GameState *gameState() const { return game_state_; }
  void bindBase(BaseEntity *base, EntityHandle base_id) {
    base_ = base;
    base_id_ = base_id;
  }
  void bindEnemies(class EnemyContainer *enemies) { enemies_ = enemies; }
  void bindProjectiles(class ProjectileContainer *projectiles) {
    projectiles_ = projectiles;
  }
  void bindPathGrid(const PathGrid *grid) { path_grid_ = grid; }
  void setCharacterTextures(
      const std::array<std::array<int, evolution::kVisualBandCount>,
                       static_cast<std::size_t>(CharacterId::Count)> *textures) {
    textures_ = textures;
  }
  void setProjectileTextures(
      const std::array<int, static_cast<std::size_t>(CharacterId::Count)>
          *textures) {
    projectile_textures_ = textures;
  }

  EntityHandle createCreature(float x, float y, CharacterId character_id,
                              int tier, int kills, int roster_index);

  bool applyDamage(EntityHandle id, float damage, EntityHandle source_enemy);

  CharacterId getCharacter(EntityHandle id) const;
  int getTier(EntityHandle id) const;
  int getKills(EntityHandle id) const;
  float getHp(EntityHandle id) const;
  float getHpMax(EntityHandle id) const;
  int getRosterIndex(EntityHandle id) const;
  void setAttackCooldown(EntityHandle id, float sec);
  void ensureAttackCooldownAtLeast(EntityHandle id, float sec);
  void addKills(EntityHandle id, int delta);
  void recalcStatsForCharacter(CharacterId cid);

  bool setWorldPosition(EntityHandle id, float x, float y);
  bool moveToCell(EntityHandle id, int col, int row);

  void update(float delta_time) override;

  // SoA gameplay arrays.
  DynamicArray<CharacterId> character;
  DynamicArray<std::int32_t> roster_index;
  DynamicArray<std::int32_t> tier;
  DynamicArray<std::int32_t> kills;

  DynamicArray<float> hp;
  DynamicArray<float> hp_max;
  DynamicArray<float> attack_damage;
  DynamicArray<float> attack_range_px;
  DynamicArray<float> attack_rate_per_sec;
  DynamicArray<float> attack_cooldown_sec;
  DynamicArray<float> move_speed_px_per_sec;

  DynamicArray<float> signature_cooldown_sec;
  DynamicArray<float> ability_cooldown_sec;
  DynamicArray<float> clone_remaining_sec;
  DynamicArray<float> frenzy_remaining_sec;
  DynamicArray<std::uint32_t> rng_state;

  DynamicArray<CreatureState> state;
  DynamicArray<float> state_time_sec;

protected:
  void swapSlots(std::uint32_t a, std::uint32_t b) override;
  void resizeArrays(int new_capacity) override;

private:
  void schedule_destroy(std::uint32_t slot);
  void on_death(std::uint32_t slot, EntityHandle source_enemy);

  Engine *engine_{nullptr};
  GameState *game_state_{nullptr};
  BaseEntity *base_{nullptr};
  EntityHandle base_id_{INVALID_ID};
  class EnemyContainer *enemies_{nullptr};
  class ProjectileContainer *projectiles_{nullptr};
  const PathGrid *path_grid_{nullptr};
  const std::array<std::array<int, evolution::kVisualBandCount>,
                   static_cast<std::size_t>(CharacterId::Count)> *textures_{nullptr};
  const std::array<int, static_cast<std::size_t>(CharacterId::Count)>
      *projectile_textures_{nullptr};

  DynamicArray<float> move_recalc_remaining_sec_;
  DynamicArray<float> move_segment_elapsed_sec_;
  DynamicArray<float> move_segment_duration_sec_;
  DynamicArray<float> move_segment_start_x_;
  DynamicArray<float> move_segment_start_y_;
  DynamicArray<float> move_segment_end_x_;
  DynamicArray<float> move_segment_end_y_;
  DynamicArray<std::uint16_t> move_waypoint_index_;
  std::vector<std::vector<std::uint16_t>> move_path_indices_{};
};

} // namespace tower_swarm
