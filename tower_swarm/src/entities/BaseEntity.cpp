#include "entities/BaseEntity.h"

#include "Constants.h"

#include "ATMEngine.h"

#include <algorithm>

namespace tower_swarm {

BaseEntity::BaseEntity(int type_id, std::uint8_t default_layer,
                       int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity),
      hp(initial_capacity), hp_max(initial_capacity) {}

EntityHandle BaseEntity::createBase(Engine *engine, float x, float y, int max_hp,
                                   int texture_id, int size_px) {
  if (!engine) {
    return INVALID_ID;
  }

  const int type_id = getTypeId();
  const EntityHandle id = engine_create_entity(engine, type_id);
  if (id == INVALID_ID) {
    return INVALID_ID;
  }

  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    engine_destroy_entity(engine, id, type_id);
    return INVALID_ID;
  }

  widths[slot] = static_cast<std::int16_t>(std::max(size_px, 1));
  heights[slot] = static_cast<std::int16_t>(std::max(size_px, 1));
  texture_ids[slot] = static_cast<std::int16_t>(texture_id);
  rotations[slot] = 0.0f;
  z_indices[slot] = kZIndexBase;

  hp_max[slot] = std::max(max_hp, 1);
  hp[slot] = hp_max[slot];

  engine_set_entity_position(engine, id, type_id, x, y);
  engine_set_entity_visible(engine, id, type_id, true);
  engine_mark_static_dirty(engine);
  return id;
}

void BaseEntity::setMaxHp(EntityHandle id, int new_max_hp, bool refill) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  const int max_hp_value = std::max(new_max_hp, 1);
  hp_max[slot] = max_hp_value;
  if (refill) {
    hp[slot] = hp_max[slot];
  } else {
    hp[slot] = std::clamp(hp[slot], 0, hp_max[slot]);
  }
}

void BaseEntity::addMaxHp(EntityHandle id, int delta, bool also_heal) {
  if (delta <= 0) {
    return;
  }
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  const int before_max = hp_max[slot];
  const int next_max = std::max(1, before_max + delta);
  hp_max[slot] = next_max;
  if (also_heal) {
    hp[slot] = std::clamp(hp[slot] + delta, 0, hp_max[slot]);
  } else {
    hp[slot] = std::clamp(hp[slot], 0, hp_max[slot]);
  }
}

void BaseEntity::resetHp(EntityHandle id, int new_hp) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  hp[slot] = std::clamp(new_hp, 0, hp_max[slot]);
}

int BaseEntity::getHp(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0;
  }
  return hp[slot];
}

int BaseEntity::getHpMax(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0;
  }
  return hp_max[slot];
}

bool BaseEntity::applyDamage(EntityHandle id, int damage) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }
  if (damage <= 0) {
    return false;
  }
  const int before = hp[slot];
  hp[slot] = std::max(0, hp[slot] - damage);
  return before != hp[slot];
}

void BaseEntity::update(float delta_time) { (void)delta_time; }

void BaseEntity::swapSlots(std::uint32_t a, std::uint32_t b) {
  if (a == b) {
    return;
  }
  RenderableEntityContainer::swapSlots(a, b);
  std::swap(hp[a], hp[b]);
  std::swap(hp_max[a], hp_max[b]);
}

void BaseEntity::resizeArrays(int new_capacity) {
  const int prev_capacity = capacity;
  RenderableEntityContainer::resizeArrays(new_capacity);
  if (capacity == prev_capacity) {
    return;
  }
  hp.resize(capacity, count, 0);
  hp_max.resize(capacity, count, 0);
}

} // namespace tower_swarm
