#include "entities/PickupContainer.h"

#include "Constants.h"
#include "TowerSwarmMath.h"
#include "entities/CreatureContainer.h"
#include "levels/GameState.h"

#include "ATMEngine.h"

#include <algorithm>

namespace tower_swarm {
namespace {

void move_with_grid(Engine *engine, RenderableEntityContainer *container,
                    std::uint32_t slot, float x, float y) {
  if (!engine || !container || slot >= static_cast<std::uint32_t>(container->count)) {
    return;
  }

  container->x_positions[slot] = x;
  container->y_positions[slot] = y;

  const int32_t node_idx = container->grid_node_indices[slot];
  if (node_idx != -1) {
    engine->grid.move(node_idx, x, y);
  }

  const int cx = std::clamp(static_cast<int>(x * INV_GRID_CELL_SIZE), 0,
                            static_cast<int>(GRID_CELL_WIDTH) - 1);
  const int cy = std::clamp(static_cast<int>(y * INV_GRID_CELL_SIZE), 0,
                            static_cast<int>(GRID_CELL_HEIGHT) - 1);
  container->cell_x[slot] = static_cast<std::uint16_t>(cx);
  container->cell_y[slot] = static_cast<std::uint16_t>(cy);
}

} // namespace

PickupContainer::PickupContainer(Engine *engine, int type_id,
                                 std::uint8_t default_layer,
                                 int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity),
      pickup_state(initial_capacity),
      age_sec(initial_capacity),
      base_x(initial_capacity),
      base_y(initial_capacity),
      homing_target_x(initial_capacity),
      homing_target_y(initial_capacity),
      value(initial_capacity),
      engine_(engine) {}

EntityHandle PickupContainer::spawnPickup(float x, float y, int val) {
  if (!engine_ || val <= 0) {
    return INVALID_ID;
  }

  const EntityHandle id = engine_create_entity(engine_, getTypeId());
  if (id == INVALID_ID) {
    return INVALID_ID;
  }

  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    engine_destroy_entity(engine_, id, getTypeId());
    return INVALID_ID;
  }

  pickup_state[slot] = PickupState::FloatUp;
  age_sec[slot] = 0.0f;
  base_x[slot] = x;
  base_y[slot] = y;
  homing_target_x[slot] = x;
  homing_target_y[slot] = y;
  value[slot] = val;

  widths[slot] = static_cast<std::int16_t>(kPickupSizePx);
  heights[slot] = static_cast<std::int16_t>(kPickupSizePx);
  rotations[slot] = 0.0f;
  z_indices[slot] = kZIndexPickups;
  texture_ids[slot] = static_cast<std::int16_t>(texture_id_);
  flags[slot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);

  move_with_grid(engine_, this, slot, x, y);
  return id;
}

void PickupContainer::schedule_destroy(std::uint32_t slot) {
  if (!engine_ || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  const EntityHandle id = getStableId(slot);
  if (id == INVALID_ID) {
    return;
  }
  flags[slot] &= ~static_cast<std::uint8_t>(EntityFlag::VISIBLE);
  engine_->pending_removals.push_back(
      EntityRef{static_cast<std::uint32_t>(getTypeId()), id});
}

void PickupContainer::update(float delta_time) {
  if (!engine_) {
    return;
  }
  const float dt = std::max(delta_time, 0.0f);

  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }

    age_sec[slot] += dt;

    Vec2 pos = make_vec2(x_positions[slot], y_positions[slot]);
    const float half = static_cast<float>(widths[slot]) * 0.5f;
    Vec2 center = pos + make_vec2(half, half);

    if (pickup_state[slot] == PickupState::FloatUp) {
      const float t = clampf(age_sec[slot] / kPickupFloatUpSec, 0.0f, 1.0f);
      pos.x = base_x[slot];
      pos.y = base_y[slot] - kPickupFloatUpPx * t;
      move_with_grid(engine_, this, slot, pos.x, pos.y);
      center = pos + make_vec2(half, half);
    }

    if (creatures_) {
      const auto &refs =
          engine_->grid.queryCircle(center.x, center.y, kPickupAttractRadiusPx);
      bool found = false;
      float best_d2 = kPickupAttractRadiusPx * kPickupAttractRadiusPx;
      float target_x = center.x;
      float target_y = center.y;
      for (const EntityRef &ref : refs) {
        if (static_cast<int>(ref.type) != creatures_->getTypeId()) {
          continue;
        }
        if (!engine_is_handle_valid(engine_, ref.index, creatures_->getTypeId())) {
          continue;
        }
        const std::uint32_t cslot = creatures_->getSlot(ref.index);
        if (cslot == INVALID_ID ||
            cslot >= static_cast<std::uint32_t>(creatures_->count)) {
          continue;
        }
        const float chalf = static_cast<float>(creatures_->widths[cslot]) * 0.5f;
        const float cx = creatures_->x_positions[cslot] + chalf;
        const float cy = creatures_->y_positions[cslot] + chalf;
        const float dx = cx - center.x;
        const float dy = cy - center.y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
          best_d2 = d2;
          target_x = cx;
          target_y = cy;
          found = true;
        }
      }

      if (found) {
        pickup_state[slot] = PickupState::Homing;
        homing_target_x[slot] = target_x;
        homing_target_y[slot] = target_y;
      }
    }

    if (pickup_state[slot] == PickupState::Homing) {
      const Vec2 target =
          make_vec2(homing_target_x[slot], homing_target_y[slot]);
      const Vec2 to = target - center;
      const float d2 = length_sq(to);
      if (d2 <= kPickupCollectDistancePx * kPickupCollectDistancePx) {
        if (game_state_) {
          game_state_->essence += value[slot];
          game_state_->essence_earned_this_level += value[slot];
        }
        schedule_destroy(slot);
        continue;
      }

      const Vec2 dir = safe_normalize(to);
      const Vec2 next_center =
          center + dir * (kPickupHomingSpeedPxPerSec * dt);
      const Vec2 next_pos = next_center - make_vec2(half, half);
      move_with_grid(engine_, this, slot, next_pos.x, next_pos.y);
    }
  }
}

void PickupContainer::swapSlots(std::uint32_t a, std::uint32_t b) {
  if (a == b) {
    return;
  }
  RenderableEntityContainer::swapSlots(a, b);
  std::swap(pickup_state[a], pickup_state[b]);
  std::swap(age_sec[a], age_sec[b]);
  std::swap(base_x[a], base_x[b]);
  std::swap(base_y[a], base_y[b]);
  std::swap(homing_target_x[a], homing_target_x[b]);
  std::swap(homing_target_y[a], homing_target_y[b]);
  std::swap(value[a], value[b]);
}

void PickupContainer::resizeArrays(int new_capacity) {
  const int prev_capacity = capacity;
  RenderableEntityContainer::resizeArrays(new_capacity);
  if (capacity == prev_capacity) {
    return;
  }
  pickup_state.resize(capacity, count, PickupState::FloatUp);
  age_sec.resize(capacity, count, 0.0f);
  base_x.resize(capacity, count, 0.0f);
  base_y.resize(capacity, count, 0.0f);
  homing_target_x.resize(capacity, count, 0.0f);
  homing_target_y.resize(capacity, count, 0.0f);
  value.resize(capacity, count, 0);
}

} // namespace tower_swarm
