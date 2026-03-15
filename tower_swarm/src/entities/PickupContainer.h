#pragma once

#include "ATMEngine.h"

#include <cstdint>

struct Engine;

namespace tower_swarm {

class CreatureContainer;
struct GameState;

enum class PickupState : std::uint8_t { FloatUp = 0, Homing = 1 };

class PickupContainer final : public RenderableEntityContainer {
public:
  explicit PickupContainer(Engine *engine, int type_id,
                           std::uint8_t default_layer, int initial_capacity);

  void bindCreatures(CreatureContainer *creatures) { creatures_ = creatures; }
  void bindGameState(GameState *state) { game_state_ = state; }
  void setTexture(int texture_id) { texture_id_ = texture_id; }

  EntityHandle spawnPickup(float x, float y, int value);

  void update(float delta_time) override;

  DynamicArray<PickupState> pickup_state;
  DynamicArray<float> age_sec;
  DynamicArray<float> base_x;
  DynamicArray<float> base_y;
  DynamicArray<float> homing_target_x;
  DynamicArray<float> homing_target_y;
  DynamicArray<std::int32_t> value;

protected:
  void swapSlots(std::uint32_t a, std::uint32_t b) override;
  void resizeArrays(int new_capacity) override;

private:
  void schedule_destroy(std::uint32_t slot);

  Engine *engine_{nullptr};
  CreatureContainer *creatures_{nullptr};
  GameState *game_state_{nullptr};
  int texture_id_{0};
};

} // namespace tower_swarm

