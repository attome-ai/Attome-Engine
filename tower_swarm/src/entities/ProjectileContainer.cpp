#include "entities/ProjectileContainer.h"

#include "Constants.h"
#include "TowerSwarmMath.h"
#include "characters/CharacterId.h"
#include "entities/CreatureContainer.h"
#include "entities/EnemyContainer.h"
#include "levels/GameState.h"

#include "ATMEngine.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

ProjectileContainer::ProjectileContainer(Engine *engine, int type_id,
                                         std::uint8_t default_layer,
                                         int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity),
      vx(initial_capacity),
      vy(initial_capacity),
      damage(initial_capacity),
      age_sec(initial_capacity),
      lifetime_sec(initial_capacity),
      pierce_remaining(initial_capacity),
      splash_radius_px(initial_capacity),
      source_creature(initial_capacity),
      last_hit_enemy(initial_capacity),
      chain_remaining(initial_capacity),
      chain_range_px(initial_capacity),
      engine_(engine) {}

EntityHandle ProjectileContainer::spawnProjectile(
    float x, float y, float vx_in, float vy_in, float dmg, float life,
    int pierce_count, float splash_radius, EntityHandle source,
    int texture_id_override, int chain_remaining_in, float chain_range_px_in) {
  if (!engine_) {
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

  vx[slot] = vx_in;
  vy[slot] = vy_in;
  damage[slot] = std::max(0.0f, dmg);
  age_sec[slot] = 0.0f;
  lifetime_sec[slot] = std::max(kProjectileMinLifetimeSec, life);
  pierce_remaining[slot] = static_cast<std::int16_t>(std::max(0, pierce_count));
  splash_radius_px[slot] = std::max(0.0f, splash_radius);
  source_creature[slot] = source;
  last_hit_enemy[slot] = INVALID_ID;
  chain_remaining[slot] = static_cast<std::int16_t>(std::clamp(
      chain_remaining_in, static_cast<int>(std::numeric_limits<std::int16_t>::min()),
      static_cast<int>(std::numeric_limits<std::int16_t>::max())));
  chain_range_px[slot] = std::max(0.0f, chain_range_px_in);

  widths[slot] = static_cast<std::int16_t>(kProjectileSizePx);
  heights[slot] = static_cast<std::int16_t>(kProjectileSizePx);
  rotations[slot] = 0.0f;
  z_indices[slot] = kZIndexProjectiles;
  const int tex = texture_id_override >= 0 ? texture_id_override : texture_id_;
  texture_ids[slot] = static_cast<std::int16_t>(tex);
  flags[slot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);

  move_with_grid(engine_, this, slot, x, y);
  return id;
}

void ProjectileContainer::schedule_destroy(std::uint32_t slot) {
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

bool ProjectileContainer::try_hit(std::uint32_t slot) {
  if (!engine_ || !enemies_ || slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }

  const float px = x_positions[slot];
  const float py = y_positions[slot];
  const float half = static_cast<float>(widths[slot]) * 0.5f;
  const float cx = px + half;
  const float cy = py + half;
  const float r = kProjectileHitRadiusPx +
                  static_cast<float>(kEnemyBaseSizePx) *
                      kProjectileEnemyHitPaddingFactor;
  const float r2 = r * r;

  const auto &refs = engine_->grid.queryCircle(cx, cy, r);
  EntityHandle best_enemy = INVALID_ID;
  float best_d2 = r2;

  for (const EntityRef &ref : refs) {
    if (static_cast<int>(ref.type) != enemies_->getTypeId()) {
      continue;
    }
    if (ref.index == last_hit_enemy[slot]) {
      continue;
    }
    if (!engine_is_handle_valid(engine_, ref.index, enemies_->getTypeId())) {
      continue;
    }
    const std::uint32_t eslot = enemies_->getSlot(ref.index);
    if (eslot == INVALID_ID || eslot >= static_cast<std::uint32_t>(enemies_->count)) {
      continue;
    }
    const float ehalf = static_cast<float>(enemies_->widths[eslot]) * 0.5f;
    const float ex = enemies_->x_positions[eslot] + ehalf;
    const float ey = enemies_->y_positions[eslot] + ehalf;
    const float dx = ex - cx;
    const float dy = ey - cy;
    const float d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_enemy = ref.index;
    }
  }

  if (best_enemy == INVALID_ID) {
    return false;
  }

  const std::uint32_t hit_eslot = enemies_->getSlot(best_enemy);
  Vec2 hit_center = make_vec2(cx, cy);
  if (hit_eslot != INVALID_ID && hit_eslot < static_cast<std::uint32_t>(enemies_->count) &&
      (enemies_->flags[hit_eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) != 0) {
    const float ehalf = static_cast<float>(enemies_->widths[hit_eslot]) * 0.5f;
    hit_center = make_vec2(enemies_->x_positions[hit_eslot] + ehalf,
                           enemies_->y_positions[hit_eslot] + ehalf);
  }

  const float dmg = damage[slot];
  if (splash_radius_px[slot] > 0.0f) {
    const float sr = splash_radius_px[slot];
    const float sr2 = sr * sr;
    const std::vector<EntityRef> aoe_refs = engine_->grid.queryCircle(cx, cy, sr);
    int hits = 0;
    for (const EntityRef &ref : aoe_refs) {
      if (static_cast<int>(ref.type) != enemies_->getTypeId()) {
        continue;
      }
      if (!engine_is_handle_valid(engine_, ref.index, enemies_->getTypeId())) {
        continue;
      }
      const std::uint32_t eslot = enemies_->getSlot(ref.index);
      if (eslot == INVALID_ID ||
          eslot >= static_cast<std::uint32_t>(enemies_->count)) {
        continue;
      }
      const float ehalf = static_cast<float>(enemies_->widths[eslot]) * 0.5f;
      const float ex = enemies_->x_positions[eslot] + ehalf;
      const float ey = enemies_->y_positions[eslot] + ehalf;
      const float dx = ex - cx;
      const float dy = ey - cy;
      if (dx * dx + dy * dy > sr2) {
        continue;
      }
      enemies_->applyDamage(ref.index, dmg, source_creature[slot]);
      hits++;
    }
    (void)hits;
  } else {
    enemies_->applyDamage(best_enemy, dmg, source_creature[slot]);
  }

  // Flara stage perks: burning ground after hit.
  if (game_state_ && creatures_ && source_creature[slot] != INVALID_ID &&
      engine_is_handle_valid(engine_, source_creature[slot], creatures_->getTypeId())) {
    const std::uint32_t cslot = creatures_->getSlot(source_creature[slot]);
      if (cslot != INVALID_ID && cslot < static_cast<std::uint32_t>(creatures_->count) &&
          (creatures_->flags[cslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) != 0) {
        const CharacterId cid = creatures_->character[cslot];
        const int ctier = std::max(1, creatures_->tier[cslot]);
        if (cid == CharacterId::Flara) {
          float dur = (ctier >= 7) ? characters::flara::kBurningGroundStage3Sec
                                   : (ctier >= 4) ? characters::flara::kBurningGroundStage2Sec
                                                  : 0.0f;
          if (game_state_ && game_state_->isRelicEquipped(RelicId::EruptionCore)) {
            dur = std::max(dur, relics::kEruptionCoreBurningGroundSec);
          }
          if (dur > 0.0f) {
            EffectZone z{};
            z.kind = EffectZoneKind::BurningGround;
            z.world_x = cx;
          z.world_y = cy;
          z.radius_px = std::max(0.0f, splash_radius_px[slot]);
          z.age_sec = 0.0f;
          z.lifetime_sec = dur;
          z.damage_per_sec = std::max(0.0f, dmg / dur);
          z.speed_multiplier = (ctier >= 7) ? wave_shop::kSlowTideSpeedMultiplier
                                            : 1.0f;
          z.damage_multiplier = 1.0f;
          z.owner_creature = source_creature[slot];
          game_state_->effect_zones.push_back(z);
        }
      }
    }
  }

  last_hit_enemy[slot] = best_enemy;

  // Chain / bounce logic (Crystalis stage 10).
  if (chain_remaining[slot] != 0) {
    const float cr = chain_range_px[slot];
    if (cr <= 0.0f) {
      schedule_destroy(slot);
      return true;
    }

    const auto &refs2 = engine_->grid.queryCircle(hit_center.x, hit_center.y, cr);
    EntityHandle next_enemy = INVALID_ID;
    float best_d2 = cr * cr;
    Vec2 next_center = hit_center;
    for (const EntityRef &ref : refs2) {
      if (static_cast<int>(ref.type) != enemies_->getTypeId()) {
        continue;
      }
      if (ref.index == last_hit_enemy[slot]) {
        continue;
      }
      if (!engine_is_handle_valid(engine_, ref.index, enemies_->getTypeId())) {
        continue;
      }
      const std::uint32_t eslot = enemies_->getSlot(ref.index);
      if (eslot == INVALID_ID ||
          eslot >= static_cast<std::uint32_t>(enemies_->count)) {
        continue;
      }
      if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const float ehalf = static_cast<float>(enemies_->widths[eslot]) * 0.5f;
      const Vec2 ecenter = make_vec2(enemies_->x_positions[eslot] + ehalf,
                                     enemies_->y_positions[eslot] + ehalf);
      const float d2 = length_sq(ecenter - hit_center);
      if (d2 < best_d2) {
        best_d2 = d2;
        next_enemy = ref.index;
        next_center = ecenter;
      }
    }

    if (next_enemy == INVALID_ID) {
      schedule_destroy(slot);
      return true;
    }

    const Vec2 dir = safe_normalize(next_center - hit_center);
    vx[slot] = dir.x * kProjectileSpeedPxPerSec;
    vy[slot] = dir.y * kProjectileSpeedPxPerSec;
    move_with_grid(engine_, this, slot, hit_center.x - half, hit_center.y - half);

    if (chain_remaining[slot] > 0) {
      chain_remaining[slot] =
          static_cast<std::int16_t>(chain_remaining[slot] - 1);
    }
    return true;
  }

  if (pierce_remaining[slot] <= 0) {
    schedule_destroy(slot);
  } else {
    pierce_remaining[slot] = static_cast<std::int16_t>(pierce_remaining[slot] - 1);
  }
  return true;
}

void ProjectileContainer::update(float delta_time) {
  if (!engine_) {
    return;
  }

  const float dt = std::max(delta_time, 0.0f);
  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }

    age_sec[slot] += dt;
    if (age_sec[slot] >= lifetime_sec[slot]) {
      schedule_destroy(slot);
      continue;
    }

    const Vec2 pos = make_vec2(x_positions[slot], y_positions[slot]);
    const Vec2 vel = make_vec2(vx[slot], vy[slot]);
    const Vec2 next = pos + vel * dt;
    move_with_grid(engine_, this, slot, next.x, next.y);

    (void)try_hit(slot);
  }
}

void ProjectileContainer::swapSlots(std::uint32_t a, std::uint32_t b) {
  if (a == b) {
    return;
  }
  RenderableEntityContainer::swapSlots(a, b);
  std::swap(vx[a], vx[b]);
  std::swap(vy[a], vy[b]);
  std::swap(damage[a], damage[b]);
  std::swap(age_sec[a], age_sec[b]);
  std::swap(lifetime_sec[a], lifetime_sec[b]);
  std::swap(pierce_remaining[a], pierce_remaining[b]);
  std::swap(splash_radius_px[a], splash_radius_px[b]);
  std::swap(source_creature[a], source_creature[b]);
  std::swap(last_hit_enemy[a], last_hit_enemy[b]);
  std::swap(chain_remaining[a], chain_remaining[b]);
  std::swap(chain_range_px[a], chain_range_px[b]);
}

void ProjectileContainer::resizeArrays(int new_capacity) {
  const int prev_capacity = capacity;
  RenderableEntityContainer::resizeArrays(new_capacity);
  if (capacity == prev_capacity) {
    return;
  }
  vx.resize(capacity, count, 0.0f);
  vy.resize(capacity, count, 0.0f);
  damage.resize(capacity, count, 0.0f);
  age_sec.resize(capacity, count, 0.0f);
  lifetime_sec.resize(capacity, count, kProjectileDefaultLifetimeSec);
  pierce_remaining.resize(capacity, count, 0);
  splash_radius_px.resize(capacity, count, 0.0f);
  source_creature.resize(capacity, count, INVALID_ID);
  last_hit_enemy.resize(capacity, count, INVALID_ID);
  chain_remaining.resize(capacity, count, 0);
  chain_range_px.resize(capacity, count, 0.0f);
}

} // namespace tower_swarm
