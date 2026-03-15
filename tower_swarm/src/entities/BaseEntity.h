#pragma once

#include "ATMEngine.h"

#include <cstdint>

struct Engine;

namespace tower_swarm {

class BaseEntity final : public RenderableEntityContainer {
public:
  DynamicArray<std::int32_t> hp;
  DynamicArray<std::int32_t> hp_max;

  BaseEntity(int type_id, std::uint8_t default_layer, int initial_capacity);

  EntityHandle createBase(Engine *engine, float x, float y, int max_hp,
                          int texture_id, int size_px);
  void setMaxHp(EntityHandle id, int new_max_hp, bool refill);
  void addMaxHp(EntityHandle id, int delta, bool also_heal);
  void resetHp(EntityHandle id, int new_hp);
  int getHp(EntityHandle id) const;
  int getHpMax(EntityHandle id) const;
  bool applyDamage(EntityHandle id, int damage);

  void update(float delta_time) override;

protected:
  void swapSlots(std::uint32_t a, std::uint32_t b) override;
  void resizeArrays(int new_capacity) override;
};

} // namespace tower_swarm
