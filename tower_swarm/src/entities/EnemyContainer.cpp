#include "entities/EnemyContainer.h"

#include "Constants.h"
#include "TowerSwarmMath.h"
#include "entities/BaseEntity.h"
#include "entities/CreatureContainer.h"
#include "entities/PickupContainer.h"
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

float difficulty_for_level(int level_number) {
  const int n = std::max(level_number, 1);
  return std::pow(level::kDifficultyBase, static_cast<float>(n));
}

float wave_hp_multiplier(int level_number, int wave_number) {
  const int w = std::max(wave_number, 0);
  return difficulty_for_level(level_number) *
         (1.0f + static_cast<float>(w) * level::kWaveEnemyHpWaveFactor);
}

float wave_speed_multiplier(int level_number, int wave_number) {
  const int w = std::max(wave_number, 0);
  const float diff = difficulty_for_level(level_number);
  const float scaled =
      std::pow(diff, enemies::kScalingSpeedExponent) *
      (1.0f + static_cast<float>(w) * level::kWaveEnemySpeedWaveFactor);
  return std::min(scaled, enemies::kScalingSpeedCap);
}

int reward_for_level(int base_reward, int level_number) {
  const int n = std::max(level_number, 1);
  const float mult = 1.0f + static_cast<float>(n) * enemies::kScalingRewardPerLevel;
  return std::max(0, static_cast<int>(std::round(static_cast<float>(base_reward) * mult)));
}

float damage_for_level(float base_damage, int level_number) {
  const int n = std::max(level_number, 1);
  return std::max(0.0f, base_damage * (1.0f + static_cast<float>(n) * enemies::kScalingDamagePerLevel));
}

struct EnemyBaseDef final {
  float hp = 1.0f;
  float speed = 1.0f;
  int reward = 0;
  float damage = 1.0f;
};

EnemyBaseDef get_base_def(EnemyType t) {
  switch (t) {
  case EnemyType::Grub:
    return {enemies::kGrubBaseHp, enemies::kGrubBaseSpeedPxPerSec,
            enemies::kGrubBaseRewardEssence, enemies::kGrubBaseDamageToBase};
  case EnemyType::Hulk:
    return {enemies::kHulkBaseHp, enemies::kHulkBaseSpeedPxPerSec,
            enemies::kHulkBaseRewardEssence, enemies::kHulkBaseDamageToBase};
  case EnemyType::Scuttle:
    return {enemies::kScuttleBaseHp, enemies::kScuttleBaseSpeedPxPerSec,
            enemies::kScuttleBaseRewardEssence, enemies::kScuttleBaseDamageToBase};
  case EnemyType::Driftwing:
    return {enemies::kDriftwingBaseHp, enemies::kDriftwingBaseSpeedPxPerSec,
            enemies::kDriftwingBaseRewardEssence, enemies::kDriftwingBaseDamageToBase};
  case EnemyType::Divide:
    return {enemies::kDivideBaseHp, enemies::kDivideBaseSpeedPxPerSec,
            enemies::kDivideBaseRewardEssence, enemies::kDivideBaseDamageToBase};
  case EnemyType::Vanguard:
    return {enemies::kVanguardBaseHp, enemies::kVanguardBaseSpeedPxPerSec,
            enemies::kVanguardBaseRewardEssence, enemies::kVanguardBaseDamageToBase};
  case EnemyType::Mender:
    return {enemies::kMenderBaseHp, enemies::kMenderBaseSpeedPxPerSec,
            enemies::kMenderBaseRewardEssence, enemies::kMenderBaseDamageToBase};
  case EnemyType::SiegeLord: {
    const EnemyBaseDef grub = get_base_def(EnemyType::Grub);
    return {grub.hp * enemies::kBossHpMultiplier,
            grub.speed * enemies::kBossBaseSpeedMultiplier,
            enemies::kBossDeathEssenceBase, enemies::kBossBaseDamageToBase};
  }
  case EnemyType::Count:
    break;
  }
  return {enemies::kGrubBaseHp, enemies::kGrubBaseSpeedPxPerSec,
          enemies::kGrubBaseRewardEssence, enemies::kGrubBaseDamageToBase};
}

} // namespace

EnemyContainer::EnemyContainer(Engine *engine, int type_id,
                               std::uint8_t default_layer, int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity),
      enemy_type(initial_capacity),
      hp(initial_capacity),
      hp_max(initial_capacity),
      move_speed_px_per_sec(initial_capacity),
      damage_to_base(initial_capacity),
      reward_essence(initial_capacity),
      slow_multiplier(initial_capacity),
      slow_time_sec(initial_capacity),
      frozen_time_sec(initial_capacity),
      is_child(initial_capacity),
      spawn_level(initial_capacity),
      spawn_wave(initial_capacity),
      spawn_is_elite(initial_capacity),
      spawn_extra_hp_mult(initial_capacity),
      spawn_extra_speed_mult(initial_capacity),
      boss_phase(initial_capacity),
      boss_stomp_cooldown_sec(initial_capacity),
      aura_speed_multiplier(initial_capacity),
      aura_damage_multiplier(initial_capacity),
      zone_speed_multiplier(initial_capacity),
      zone_damage_multiplier(initial_capacity),
      zone_heal_multiplier(initial_capacity),
      kill_credit_override(initial_capacity),
      engine_(engine) {}

EntityHandle EnemyContainer::spawnEnemy(EnemyType type, float x, float y,
                                       int level_number, int wave_number,
                                       bool is_elite,
                                       float extra_hp_multiplier,
                                       float extra_speed_multiplier) {
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

  const EnemyBaseDef base = get_base_def(type);
  const float extra_hp_mult = std::max(extra_hp_multiplier, 0.0f);
  const float extra_spd_mult = std::max(extra_speed_multiplier, 0.0f);

  const float hp_mult =
      wave_hp_multiplier(level_number, wave_number) * extra_hp_mult;
  const float spd_mult =
      wave_speed_multiplier(level_number, wave_number) * extra_spd_mult;

  float max_hp = std::max(1.0f, base.hp * hp_mult);
  float spd = std::max(0.0f, base.speed * spd_mult);

  if (is_elite) {
    max_hp *= enemies::kEliteHpMultiplier;
    spd *= enemies::kEliteSpeedMultiplier;
  }

  enemy_type[slot] = type;
  spawn_level[slot] = level_number;
  spawn_wave[slot] = wave_number;
  spawn_is_elite[slot] = static_cast<std::uint8_t>(is_elite ? 1 : 0);
  spawn_extra_hp_mult[slot] = extra_hp_mult;
  spawn_extra_speed_mult[slot] = extra_spd_mult;
  boss_phase[slot] = (type == EnemyType::SiegeLord) ? 1 : 0;
  boss_stomp_cooldown_sec[slot] =
      (type == EnemyType::SiegeLord) ? enemies::kBossStompInitialDelaySec : 0.0f;
  hp_max[slot] = max_hp;
  hp[slot] = max_hp;
  move_speed_px_per_sec[slot] = spd;
  damage_to_base[slot] = damage_for_level(base.damage, level_number);

  const int reward =
      (type == EnemyType::SiegeLord)
          ? (enemies::kBossDeathEssenceBase +
             std::max(0, level_number) * enemies::kBossDeathEssencePerLevel)
          : reward_for_level(base.reward, level_number);
  reward_essence[slot] = reward;

  slow_multiplier[slot] = 1.0f;
  slow_time_sec[slot] = 0.0f;
  frozen_time_sec[slot] = 0.0f;
  is_child[slot] = 0;
  aura_speed_multiplier[slot] = 1.0f;
  aura_damage_multiplier[slot] = 1.0f;
  zone_speed_multiplier[slot] = 1.0f;
  zone_damage_multiplier[slot] = 1.0f;
  kill_credit_override[slot] = INVALID_ID;

  widths[slot] = static_cast<std::int16_t>(kEnemyBaseSizePx);
  heights[slot] = static_cast<std::int16_t>(kEnemyBaseSizePx);
  rotations[slot] = 0.0f;
  z_indices[slot] = kZIndexEnemies;

  int tex = 0;
  if (textures_) {
    const std::size_t idx = static_cast<std::size_t>(type);
    if (idx < textures_->size()) {
      tex = (*textures_)[idx];
    }
  }
  texture_ids[slot] = static_cast<std::int16_t>(tex);
  flags[slot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);
  move_with_grid(engine_, this, slot, x, y);
  return id;
}

EnemyType EnemyContainer::getEnemyType(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return EnemyType::Grub;
  }
  return enemy_type[slot];
}

float EnemyContainer::getHp(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0.0f;
  }
  return hp[slot];
}

float EnemyContainer::getHpMax(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0.0f;
  }
  return hp_max[slot];
}

bool EnemyContainer::is_front_hit(std::uint32_t slot,
                                  EntityHandle source_creature) const {
  if (!base_ || base_id_ == INVALID_ID || !creatures_ || !engine_ ||
      slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }

  if (!engine_is_handle_valid(engine_, source_creature, creatures_->getTypeId())) {
    return false;
  }

  const std::uint32_t src_slot = creatures_->getSlot(source_creature);
  if (src_slot == INVALID_ID || src_slot >= static_cast<std::uint32_t>(creatures_->count)) {
    return false;
  }

  const std::uint32_t base_slot = base_->getSlot(base_id_);
  if (base_slot == INVALID_ID || base_slot >= static_cast<std::uint32_t>(base_->count)) {
    return false;
  }

  const Vec2 enemy_pos = make_vec2(x_positions[slot], y_positions[slot]);
  const Vec2 base_pos = make_vec2(base_->x_positions[base_slot], base_->y_positions[base_slot]);
  const Vec2 src_pos = make_vec2(creatures_->x_positions[src_slot], creatures_->y_positions[src_slot]);

  const float enemy_half = static_cast<float>(widths[slot]) * 0.5f;
  const float base_half = static_cast<float>(base_->widths[base_slot]) * 0.5f;
  const float src_half =
      static_cast<float>(creatures_->widths[src_slot]) * 0.5f;

  const Vec2 enemy_center = enemy_pos + make_vec2(enemy_half, enemy_half);
  const Vec2 base_center = base_pos + make_vec2(base_half, base_half);
  const Vec2 src_center = src_pos + make_vec2(src_half, src_half);

  const Vec2 to_base = safe_normalize(base_center - enemy_center);
  const Vec2 to_src = safe_normalize(src_center - enemy_center);
  return dot(to_src, to_base) > enemies::kFrontHitDotThreshold;
}

bool EnemyContainer::applyDamage(EntityHandle id, float damage,
                                EntityHandle source_creature) {
  return applyDamageInternal(id, damage, source_creature, false);
}

void EnemyContainer::scheduleDelayedDamage(EntityHandle target, float delay_sec,
                                          float damage,
                                          EntityHandle source_creature) {
  if (target == INVALID_ID || delay_sec <= 0.0f || damage <= 0.0f) {
    return;
  }
  delayed_damage_.push_back(DelayedDamage{
      target, std::max(0.0f, delay_sec), std::max(0.0f, damage), source_creature});
}

void EnemyContainer::scheduleDelayedKill(EntityHandle target, float delay_sec,
                                        EntityHandle source_creature) {
  if (target == INVALID_ID || delay_sec <= 0.0f) {
    return;
  }
  if (!engine_ || !engine_is_handle_valid(engine_, target, getTypeId())) {
    return;
  }
  const std::uint32_t slot = getSlot(target);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
    return;
  }
  const float kill_dmg = std::max(1.0f, hp_max[slot] + 1.0f);
  scheduleDelayedDamage(target, delay_sec, kill_dmg, source_creature);
}

void EnemyContainer::displace(EntityHandle id, float dx, float dy) {
  if (!engine_ || id == INVALID_ID) {
    return;
  }
  if (!engine_is_handle_valid(engine_, id, getTypeId())) {
    return;
  }
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
    return;
  }

  const float nx =
      std::clamp(x_positions[slot] + dx, 0.0f,
                 std::max(0.0f, static_cast<float>(kWorldWidthPx - widths[slot])));
  const float ny = std::clamp(
      y_positions[slot] + dy, 0.0f,
      std::max(0.0f, static_cast<float>(kWorldHeightPx - heights[slot])));
  move_with_grid(engine_, this, slot, nx, ny);
}

bool EnemyContainer::applyDamageInternal(EntityHandle id, float damage,
                                        EntityHandle source_creature,
                                        bool is_echo) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count) ||
      damage <= 0.0f) {
    return false;
  }
  if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
    return false;
  }

  float dmg = damage;
  const EnemyType t = enemy_type[slot];

  CharacterId src_cid = CharacterId::Brix;
  int src_tier = 1;
  if (creatures_ && engine_ &&
      engine_is_handle_valid(engine_, source_creature, creatures_->getTypeId())) {
    const std::uint32_t cslot = creatures_->getSlot(source_creature);
    if (cslot != INVALID_ID &&
        cslot < static_cast<std::uint32_t>(creatures_->count) &&
        (creatures_->flags[cslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) != 0) {
      src_cid = creatures_->character[cslot];
      src_tier = std::max(1, creatures_->tier[cslot]);
    }
  }

  if (t == EnemyType::Hulk && is_front_hit(slot, source_creature)) {
    dmg *= enemies::kHulkFrontDamageTakenMultiplier;
  }
  if (t == EnemyType::Vanguard && is_front_hit(slot, source_creature)) {
    float resist = enemies::kVanguardFrontResist;
    if (src_cid == CharacterId::Wraith && src_tier >= 4) {
      resist *= (1.0f - characters::wraith::kArmorIgnoreFraction);
    }
    dmg *= std::max(0.0f, 1.0f - resist);
  }

  if (src_cid == CharacterId::Wraith && src_tier >= 7) {
    const float denom = std::max(1.0f, hp_max[slot]);
    const float frac = std::clamp(hp[slot] / denom, 0.0f, 1.0f);
    if (frac <= characters::wraith::kExecuteBelowHpFraction) {
      dmg = std::max(dmg, hp[slot] + 1.0f);
    }
  }

  hp[slot] = std::max(0.0f, hp[slot] - dmg);

  if (!is_echo && game_state_ && game_state_->hasWaveBuff(WaveBuffId::EchoStrike)) {
    const float echo_dmg =
        std::max(0.0f, dmg * wave_shop::kEchoStrikeDamageRepeatFraction);
    if (echo_dmg > 0.0f) {
      delayed_damage_.push_back(DelayedDamage{
          id, wave_shop::kEchoStrikeRepeatDelaySec, echo_dmg, source_creature});
    }
  }

  if (hp[slot] > 0.0f) {
    return false;
  }

  on_death(slot, source_creature);
  return true;
}

void EnemyContainer::schedule_destroy(std::uint32_t slot) {
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

void EnemyContainer::on_death(std::uint32_t slot, EntityHandle source_creature) {
  if (slot >= static_cast<std::uint32_t>(count)) {
    return;
  }

  const EntityHandle dead_id = getStableId(slot);
  EntityHandle credited_source = source_creature;
  if (engine_ && creatures_) {
    const EntityHandle override_id = kill_credit_override[slot];
    if (override_id != INVALID_ID &&
        engine_is_handle_valid(engine_, override_id, creatures_->getTypeId())) {
      credited_source = override_id;
    }
  }
  const float half = static_cast<float>(widths[slot]) * 0.5f;
  const float cx = x_positions[slot] + half;
  const float cy = y_positions[slot] + half;

  if (game_state_) {
    game_state_->enemies_killed_this_level += 1;
    if (enemy_type[slot] == EnemyType::SiegeLord && is_child[slot] == 0) {
      game_state_->lifetime_bosses_killed =
          std::max(0, game_state_->lifetime_bosses_killed + 1);
      game_state_->recomputeMetaProgression();
    }
  }

  if (creatures_ && engine_ &&
      engine_is_handle_valid(engine_, credited_source, creatures_->getTypeId())) {
    creatures_->addKills(credited_source, 1);
  }

  if (pickups_ && game_state_) {
    int value = reward_essence[slot];
    if (game_state_->hasWaveBuff(WaveBuffId::FrenziedBlood)) {
      value += wave_shop::kFrenziedBloodEssencePerKill;
    }
    if (value > 0) {
      float mult = 1.0f;
      if (game_state_->isRelicEquipped(RelicId::EssenceMagnet)) {
        mult *= (1.0f + std::max(0.0f, relics::kEssenceMagnetDropBonus));
      }
      mult *= std::max(0.0f, game_state_->voidAppetiteEssenceDropMultiplier());
      if (mult != 1.0f) {
        value = std::max(
            0, static_cast<int>(std::lround(static_cast<float>(value) * mult)));
      }
    }
    if (value > 0) {
      pickups_->spawnPickup(cx - static_cast<float>(kPickupSizePx) * 0.5f,
                            cy - static_cast<float>(kPickupSizePx) * 0.5f,
                            value);
    }
  }

  if (game_state_ && game_state_->hasWaveBuff(WaveBuffId::VoidPulse)) {
    game_state_->void_pulse_kill_counter += 1;
    if (game_state_->void_pulse_kill_counter > 0 &&
        (game_state_->void_pulse_kill_counter % wave_shop::kVoidPulseKillInterval) ==
            0) {
      const float pulse_dmg = std::max(
          0.0f, hp_max[slot] * wave_shop::kVoidPulseDamageFractionOfKilledEnemyMaxHp);
      if (pulse_dmg > 0.0f && engine_) {
        const std::vector<EntityRef> refs =
            engine_->grid.queryCircle(cx, cy, wave_shop::kVoidPulseRadiusPx);
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (ref.index == dead_id) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          (void)applyDamageInternal(ref.index, pulse_dmg, source_creature, true);
        }
      }
    }
  }

  if (creatures_ && engine_ &&
      engine_is_handle_valid(engine_, credited_source, creatures_->getTypeId())) {
    const std::uint32_t cslot = creatures_->getSlot(credited_source);
    if (cslot != INVALID_ID &&
        cslot < static_cast<std::uint32_t>(creatures_->count) &&
        (creatures_->flags[cslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) != 0) {
      const CharacterId cid = creatures_->character[cslot];
      const int ctier = std::max(1, creatures_->tier[cslot]);
      if (cid == CharacterId::Wraith && ctier >= characters::kEvolutionStage4MinTier) {
        EntityHandle nearest = INVALID_ID;
        float best_d2 = std::numeric_limits<float>::max();
        for (std::uint32_t eslot = 0; eslot < static_cast<std::uint32_t>(count); ++eslot) {
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const EntityHandle eid = getStableId(eslot);
          if (eid == INVALID_ID || eid == dead_id) {
            continue;
          }
          const float ehalf = static_cast<float>(widths[eslot]) * 0.5f;
          const float ex = x_positions[eslot] + ehalf;
          const float ey = y_positions[eslot] + ehalf;
          const float dx = ex - cx;
          const float dy = ey - cy;
          const float d2 = dx * dx + dy * dy;
          if (d2 < best_d2) {
            best_d2 = d2;
            nearest = eid;
          }
        }
        if (nearest != INVALID_ID) {
          const float bolt_dmg = std::max(0.0f, creatures_->attack_damage[cslot]);
          if (bolt_dmg > 0.0f) {
            (void)applyDamageInternal(nearest, bolt_dmg, credited_source, true);
          }
        }
      }

      if (game_state_ && game_state_->isRelicEquipped(RelicId::ChainStrike) &&
          cid == CharacterId::Ironjaw) {
        const float shock_dmg = std::max(0.0f, creatures_->attack_damage[cslot]);
        const float radius = std::max(0.0f, relics::kChainStrikeShockwaveRadiusPx);
        if (shock_dmg > 0.0f && radius > 0.0f && engine_) {
          const std::vector<EntityRef> refs =
              engine_->grid.queryCircle(cx, cy, radius);
          for (const EntityRef &ref : refs) {
            if (static_cast<int>(ref.type) != getTypeId()) {
              continue;
            }
            if (ref.index == dead_id) {
              continue;
            }
            if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
              continue;
            }
            (void)applyDamageInternal(ref.index, shock_dmg, credited_source, true);
          }
        }
      }
    }
  }

  if (enemy_type[slot] == EnemyType::Divide && is_child[slot] == 0) {
    const float cx = x_positions[slot];
    const float cy = y_positions[slot];
    const float parent_hp_mult = spawn_extra_hp_mult[slot];
    const float parent_spd_mult = spawn_extra_speed_mult[slot];
    for (int i = 0; i < enemies::kDivideChildrenCount; ++i) {
      const float ox = (i == 0) ? -enemies::kDivideChildSpawnOffsetXPx
                                : enemies::kDivideChildSpawnOffsetXPx;
      const EntityHandle child = spawnEnemy(
          EnemyType::Divide, cx + ox,
          cy + enemies::kDivideChildSpawnOffsetYPx, spawn_level[slot],
          spawn_wave[slot], spawn_is_elite[slot] != 0,
          parent_hp_mult * enemies::kDivideChildHpFactor, parent_spd_mult);
      if (child != INVALID_ID) {
        const std::uint32_t child_slot = getSlot(child);
        if (child_slot != INVALID_ID &&
            child_slot < static_cast<std::uint32_t>(count)) {
          is_child[child_slot] = 1;
          reward_essence[child_slot] = 0;
        }
      }
    }
  }

  schedule_destroy(slot);
}

void EnemyContainer::update(float delta_time) {
  if (!engine_ || !base_ || base_id_ == INVALID_ID) {
    return;
  }

  const float dt = std::max(delta_time, 0.0f);
  const bool cold_bloom =
      game_state_ && game_state_->isRelicEquipped(RelicId::ColdBloom);

  if (!delayed_damage_.empty()) {
    for (std::size_t i = 0; i < delayed_damage_.size();) {
      DelayedDamage &hit = delayed_damage_[i];
      hit.time_sec = std::max(0.0f, hit.time_sec - dt);
      if (hit.time_sec > 0.0f) {
        ++i;
        continue;
      }

      if (engine_ && engine_is_handle_valid(engine_, hit.target, getTypeId())) {
        (void)applyDamageInternal(hit.target, hit.damage, hit.source_creature,
                                  true);
      }

      delayed_damage_[i] = delayed_damage_.back();
      delayed_damage_.pop_back();
    }
  }

  const std::uint32_t base_slot = base_->getSlot(base_id_);
  const Vec2 base_pos =
      (base_slot == INVALID_ID || base_slot >= static_cast<std::uint32_t>(base_->count))
          ? make_vec2(kWorldWidthPx * 0.5f - static_cast<float>(kBaseSizePx) * 0.5f,
                      kWorldHeightPx * 0.5f - static_cast<float>(kBaseSizePx) * 0.5f)
          : make_vec2(base_->x_positions[base_slot], base_->y_positions[base_slot]);
  const float base_half =
      (base_slot == INVALID_ID || base_slot >= static_cast<std::uint32_t>(base_->count))
          ? static_cast<float>(kBaseSizePx) * 0.5f
          : static_cast<float>(base_->widths[base_slot]) * 0.5f;
  const Vec2 base_center = base_pos + make_vec2(base_half, base_half);

  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    aura_speed_multiplier[slot] = 1.0f;
    aura_damage_multiplier[slot] = 1.0f;
    zone_speed_multiplier[slot] = 1.0f;
    zone_damage_multiplier[slot] = 1.0f;
    zone_heal_multiplier[slot] = 1.0f;
    kill_credit_override[slot] = INVALID_ID;
  }

  // Creature auras that affect enemies (Null drain, Mossling slow aura).
  if (creatures_ && engine_) {
    for (std::uint32_t cslot = 0; cslot < static_cast<std::uint32_t>(creatures_->count); ++cslot) {
      if ((creatures_->flags[cslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const CharacterId cid = creatures_->character[cslot];
      const int ctier = std::max(1, creatures_->tier[cslot]);
      const float chalf = static_cast<float>(creatures_->widths[cslot]) * 0.5f;
      const float ccx = creatures_->x_positions[cslot] + chalf;
      const float ccy = creatures_->y_positions[cslot] + chalf;

      if (cid == CharacterId::NullSeed) {
        float dmg_mult = 1.0f;
        float spd_mult = 1.0f;
        bool credit_kills = false;
        if (ctier >= characters::kEvolutionStage4MinTier) {
          dmg_mult = std::max(0.0f, 1.0f - characters::null_seed::kDrainDamageStage3);
          spd_mult = std::max(0.0f, 1.0f - characters::null_seed::kDrainSpeedStage2);
          credit_kills = true;
        } else if (ctier >= 6) {
          dmg_mult = std::max(0.0f, 1.0f - characters::null_seed::kDrainDamageStage2);
          spd_mult = std::max(0.0f, 1.0f - characters::null_seed::kDrainSpeedStage2);
        } else {
          dmg_mult = std::max(0.0f, 1.0f - characters::null_seed::kDrainDamageStage1);
        }

        const auto &refs = engine_->grid.queryCircle(
            ccx, ccy, characters::base_stats::kNullDrainRadiusPx);
        const EntityHandle owner_id = creatures_->getStableId(cslot);
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          const std::uint32_t eslot = getSlot(ref.index);
          if (eslot == INVALID_ID || eslot >= static_cast<std::uint32_t>(count)) {
            continue;
          }
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }

          aura_damage_multiplier[eslot] =
              std::min(aura_damage_multiplier[eslot], dmg_mult);
          aura_speed_multiplier[eslot] =
              std::min(aura_speed_multiplier[eslot], spd_mult);
          if (credit_kills && owner_id != INVALID_ID) {
            kill_credit_override[eslot] = owner_id;
          }
        }
      } else if (cid == CharacterId::Mossling &&
                 ctier >= characters::kEvolutionStage4MinTier) {
        const auto &refs = engine_->grid.queryCircle(
            ccx, ccy, characters::mossling::kAuraRadiusStage4Px);
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          const std::uint32_t eslot = getSlot(ref.index);
          if (eslot == INVALID_ID || eslot >= static_cast<std::uint32_t>(count)) {
            continue;
          }
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          aura_speed_multiplier[eslot] = std::min(
              aura_speed_multiplier[eslot], wave_shop::kSlowTideSpeedMultiplier);
        }
      }
    }
  }

  // Effect zones (burning ground, Glitch orbs).
  if (game_state_ && !game_state_->effect_zones.empty() && engine_) {
    std::vector<EffectZone> new_zones;
    new_zones.reserve(4);

    for (std::size_t i = 0; i < game_state_->effect_zones.size();) {
      EffectZone &z = game_state_->effect_zones[i];
      z.age_sec = std::max(0.0f, z.age_sec + dt);
      const float life = std::max(0.0f, z.lifetime_sec);

      const bool field_active =
          (z.kind == EffectZoneKind::BurningGround) ||
          (z.kind == EffectZoneKind::GlitchOrb &&
           z.age_sec <= std::max(0.0f, z.slow_duration_sec));

      const bool dot_active =
          (z.kind == EffectZoneKind::BurningGround && z.damage_per_sec > 0.0f);

      if ((field_active || dot_active) && z.radius_px > 0.0f) {
        const std::vector<EntityRef> refs =
            engine_->grid.queryCircle(z.world_x, z.world_y, z.radius_px);
        const float r2 = z.radius_px * z.radius_px;
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          const std::uint32_t eslot = getSlot(ref.index);
          if (eslot == INVALID_ID ||
              eslot >= static_cast<std::uint32_t>(count)) {
            continue;
          }
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const float ehalf = static_cast<float>(widths[eslot]) * 0.5f;
          const float ex = x_positions[eslot] + ehalf;
          const float ey = y_positions[eslot] + ehalf;
          const float dx = ex - z.world_x;
          const float dy = ey - z.world_y;
          if (dx * dx + dy * dy > r2) {
            continue;
          }

          if (field_active) {
            if (z.speed_multiplier < 1.0f) {
              zone_speed_multiplier[eslot] =
                  std::min(zone_speed_multiplier[eslot], z.speed_multiplier);
            }
            if (z.damage_multiplier < 1.0f) {
              zone_damage_multiplier[eslot] =
                  std::min(zone_damage_multiplier[eslot], z.damage_multiplier);
            }
            if (cold_bloom && z.kind == EffectZoneKind::GlitchOrb) {
              zone_heal_multiplier[eslot] =
                  std::min(zone_heal_multiplier[eslot],
                           std::max(0.0f, relics::kColdBloomHealReceivedMultiplier));
            }
          }
          if (dot_active) {
            const float tick_dmg = std::max(0.0f, z.damage_per_sec * dt);
            if (tick_dmg > 0.0f) {
              applyDamage(ref.index, tick_dmg, z.owner_creature);
            }
          }
        }
      }

      if (z.kind == EffectZoneKind::GlitchOrb && !z.detonated &&
          z.detonate_after_sec > 0.0f && z.age_sec >= z.detonate_after_sec &&
          z.radius_px > 0.0f) {
        const std::vector<EntityRef> refs =
            engine_->grid.queryCircle(z.world_x, z.world_y, z.radius_px);
        const float r2 = z.radius_px * z.radius_px;
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          const std::uint32_t eslot = getSlot(ref.index);
          if (eslot == INVALID_ID ||
              eslot >= static_cast<std::uint32_t>(count)) {
            continue;
          }
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const float ehalf = static_cast<float>(widths[eslot]) * 0.5f;
          const float ex = x_positions[eslot] + ehalf;
          const float ey = y_positions[eslot] + ehalf;
          const float dx = ex - z.world_x;
          const float dy = ey - z.world_y;
          if (dx * dx + dy * dy > r2) {
            continue;
          }
          const float burst = std::max(0.0f, z.detonate_damage);
          if (burst > 0.0f) {
            applyDamage(ref.index, burst, z.owner_creature);
          }
        }

        if (z.chain_hops_remaining > 0 && z.radius_px > 0.0f) {
          const float chain_r = z.radius_px * 2.0f;
          const auto &crefs =
              engine_->grid.queryCircle(z.world_x, z.world_y, chain_r);
          EntityHandle best = INVALID_ID;
          float best_d2 = chain_r * chain_r;
          float best_x = z.world_x;
          float best_y = z.world_y;
          for (const EntityRef &ref : crefs) {
            if (static_cast<int>(ref.type) != getTypeId()) {
              continue;
            }
            if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
              continue;
            }
            const std::uint32_t eslot = getSlot(ref.index);
            if (eslot == INVALID_ID ||
                eslot >= static_cast<std::uint32_t>(count)) {
              continue;
            }
            if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const float ehalf = static_cast<float>(widths[eslot]) * 0.5f;
            const float ex = x_positions[eslot] + ehalf;
            const float ey = y_positions[eslot] + ehalf;
            const float dx = ex - z.world_x;
            const float dy = ey - z.world_y;
            const float d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
              best_d2 = d2;
              best = ref.index;
              best_x = ex;
              best_y = ey;
            }
          }

          if (best != INVALID_ID) {
            EffectZone next = z;
            next.world_x = best_x;
            next.world_y = best_y;
            next.age_sec = 0.0f;
            next.detonated = false;
            next.chain_hops_remaining =
                std::max<std::int32_t>(0, z.chain_hops_remaining - 1);
            new_zones.push_back(next);
          }
        }

        z.detonated = true;
      }

      const bool expired = life <= 0.0f || z.age_sec >= life;
      if (expired) {
        game_state_->effect_zones[i] = game_state_->effect_zones.back();
        game_state_->effect_zones.pop_back();
        continue;
      }
      ++i;
    }

    if (!new_zones.empty()) {
      for (const EffectZone &nz : new_zones) {
        game_state_->effect_zones.push_back(nz);
      }
    }
  }

  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }

    frozen_time_sec[slot] = std::max(0.0f, frozen_time_sec[slot] - dt);
    slow_time_sec[slot] = std::max(0.0f, slow_time_sec[slot] - dt);
    if (slow_time_sec[slot] <= 0.0f) {
      slow_multiplier[slot] = 1.0f;
    }

    if (frozen_time_sec[slot] > 0.0f) {
      continue;
    }

    const Vec2 pos = make_vec2(x_positions[slot], y_positions[slot]);
    const float half = static_cast<float>(widths[slot]) * 0.5f;
    const Vec2 center = pos + make_vec2(half, half);
    const Vec2 to_base = base_center - center;

    const EnemyType t = enemy_type[slot];

    if (t == EnemyType::SiegeLord && boss_phase[slot] > 0) {
      const float denom = std::max(1.0f, hp_max[slot]);
      const float frac = std::clamp(hp[slot] / denom, 0.0f, 1.0f);

      if (boss_phase[slot] == 1 && is_child[slot] == 0) {
        boss_stomp_cooldown_sec[slot] =
            std::max(0.0f, boss_stomp_cooldown_sec[slot] - dt);
        if (boss_stomp_cooldown_sec[slot] <= 0.0f && creatures_ && engine_) {
          const float rr = std::max(0.0f, enemies::kBossStompRadiusPx);
          const float r2 = rr * rr;
          float stomp_dmg = std::max(0.0f, damage_to_base[slot]);
          stomp_dmg *= aura_damage_multiplier[slot];
          stomp_dmg *= zone_damage_multiplier[slot];
          stomp_dmg *= std::max(0.0f, enemies::kBossStompDamageToCreatureMultiplier);
          stomp_dmg = std::max(0.0f, stomp_dmg);

          if (stomp_dmg > 0.0f && rr > 0.0f) {
            const auto &refs = engine_->grid.queryCircle(center.x, center.y, rr);
            const EntityHandle boss_id = getStableId(slot);
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
              if ((creatures_->flags[cslot] &
                   static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
                continue;
              }
              const float chalf = static_cast<float>(creatures_->widths[cslot]) * 0.5f;
              const float ccx = creatures_->x_positions[cslot] + chalf;
              const float ccy = creatures_->y_positions[cslot] + chalf;
              const float dx = ccx - center.x;
              const float dy = ccy - center.y;
              if (dx * dx + dy * dy > r2) {
                continue;
              }
              (void)creatures_->applyDamage(ref.index, stomp_dmg, boss_id);
            }
          }

          boss_stomp_cooldown_sec[slot] =
              std::max(0.0f, enemies::kBossStompIntervalSec);
        }
      }

      auto spawn_minions = [&](EnemyType mt, int n) {
        if (n <= 0) {
          return;
        }
        const float r = static_cast<float>(kEnemyBaseSizePx) * 2.2f;
        const float half_e = static_cast<float>(kEnemyBaseSizePx) * 0.5f;
        const float two_pi = 2.0f * std::acos(-1.0f);
        const int lvl = spawn_level[slot];
        const int wave = spawn_wave[slot];
        const bool elite = spawn_is_elite[slot] != 0;
        const float hp_mult = spawn_extra_hp_mult[slot];
        const float spd_mult = spawn_extra_speed_mult[slot];

        for (int i = 0; i < n; ++i) {
          const float a = (two_pi * static_cast<float>(i)) /
                          std::max(1.0f, static_cast<float>(n));
          const float ox = std::cos(a) * r;
          const float oy = std::sin(a) * r;
          const float px = std::clamp(
              center.x + ox - half_e, 0.0f,
              std::max(0.0f, static_cast<float>(kWorldWidthPx - kEnemyBaseSizePx)));
          const float py = std::clamp(
              center.y + oy - half_e, 0.0f,
              std::max(0.0f, static_cast<float>(kWorldHeightPx - kEnemyBaseSizePx)));
          (void)spawnEnemy(mt, px, py, lvl, wave, elite, hp_mult, spd_mult);
        }
      };

      if (boss_phase[slot] < 2 && frac <= enemies::kBossPhase2Threshold) {
        boss_phase[slot] = 2;
        move_speed_px_per_sec[slot] *= (1.0f + enemies::kBossPhase2SpeedBonus);
        spawn_minions(EnemyType::Grub, enemies::kBossPhase2SpawnGrubs);
      }
      if (boss_phase[slot] < 3 && frac <= enemies::kBossPhase3Threshold) {
        boss_phase[slot] = 3;
        spawn_minions(EnemyType::Hulk, enemies::kBossPhase3SpawnHulks);
      }
    }

    if (t == EnemyType::Mender) {
      const float heal = std::max(0.0f, enemies::kMenderHealHpPerSec) * dt;
      if (heal > 0.0f && engine_) {
        const auto &refs = engine_->grid.queryCircle(
            center.x, center.y, enemies::kMenderHealRadiusPx);
        for (const EntityRef &ref : refs) {
          if (static_cast<int>(ref.type) != getTypeId()) {
            continue;
          }
          if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
            continue;
          }
          const std::uint32_t eslot = getSlot(ref.index);
          if (eslot == INVALID_ID ||
              eslot >= static_cast<std::uint32_t>(count)) {
            continue;
          }
          if ((flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const float recv_mult = std::max(0.0f, zone_heal_multiplier[eslot]);
          const float eff_heal = std::max(0.0f, heal * recv_mult);
          if (eff_heal <= 0.0f) {
            continue;
          }
          const float nhp = std::min(hp_max[eslot], hp[eslot] + eff_heal);
          hp[eslot] = std::max(0.0f, nhp);
        }
      }
    }

    const float dist_sq = length_sq(to_base);
    const float reach = kBaseRadiusPx;
    if (dist_sq <= reach * reach) {
      if (game_state_) {
        game_state_->any_enemy_reached_base_this_level = true;
      }

      float raw_dmg = damage_to_base[slot];
      raw_dmg *= aura_damage_multiplier[slot];
      raw_dmg *= zone_damage_multiplier[slot];
      raw_dmg = std::max(0.0f, raw_dmg);

      float orin_chance = 0.0f;
      if (creatures_) {
        int orin_max_tier = 0;
        for (std::uint32_t cslot = 0; cslot < static_cast<std::uint32_t>(creatures_->count); ++cslot) {
          if ((creatures_->flags[cslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          if (creatures_->character[cslot] != CharacterId::Orin) {
            continue;
          }
          orin_max_tier = std::max(orin_max_tier, std::max(1, creatures_->tier[cslot]));
        }
        if (orin_max_tier >= characters::kEvolutionStage4MinTier) {
          orin_chance = characters::orin::kPassiveBaseShieldStage3;
        } else if (orin_max_tier >= 6) {
          orin_chance = characters::orin::kPassiveIgnoreDamageChanceStage2;
        } else if (orin_max_tier >= 1) {
          orin_chance = characters::orin::kPassiveIgnoreDamageChanceStage1;
        }
      }

      bool ignored = false;
      if (orin_chance > 0.0f) {
        static std::uint32_t rng = 0xA11CEu;
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const float roll = static_cast<float>(rng) / 4294967295.0f;
        if (roll < orin_chance) {
          ignored = true;
        }
      }

      if (!ignored) {
        const int dmg_int =
            static_cast<int>(std::ceil(std::max(0.0f, raw_dmg)));
        if (dmg_int > 0 && game_state_ &&
            game_state_->isRelicEquipped(RelicId::EternalEcho) &&
            !game_state_->eternal_echo_used_this_level) {
          const int before = base_->getHp(base_id_);
          if (before > 0 && (before - dmg_int) <= 0) {
            base_->resetHp(base_id_, 1);
            game_state_->eternal_echo_used_this_level = true;
          } else {
            base_->applyDamage(base_id_, dmg_int);
          }
        } else if (dmg_int > 0) {
          base_->applyDamage(base_id_, dmg_int);
        }
      }
      schedule_destroy(slot);
      continue;
    }

    const Vec2 dir = safe_normalize(to_base);
    const float spd = move_speed_px_per_sec[slot] * slow_multiplier[slot] *
                      aura_speed_multiplier[slot] * zone_speed_multiplier[slot];
    const Vec2 next_center = center + dir * (spd * dt);
    const Vec2 next_pos = next_center - make_vec2(half, half);
    move_with_grid(engine_, this, slot, next_pos.x, next_pos.y);
  }
}

void EnemyContainer::swapSlots(std::uint32_t a, std::uint32_t b) {
  if (a == b) {
    return;
  }
  RenderableEntityContainer::swapSlots(a, b);
  std::swap(enemy_type[a], enemy_type[b]);
  std::swap(hp[a], hp[b]);
  std::swap(hp_max[a], hp_max[b]);
  std::swap(move_speed_px_per_sec[a], move_speed_px_per_sec[b]);
  std::swap(damage_to_base[a], damage_to_base[b]);
  std::swap(reward_essence[a], reward_essence[b]);
  std::swap(slow_multiplier[a], slow_multiplier[b]);
  std::swap(slow_time_sec[a], slow_time_sec[b]);
  std::swap(frozen_time_sec[a], frozen_time_sec[b]);
  std::swap(is_child[a], is_child[b]);
  std::swap(spawn_level[a], spawn_level[b]);
  std::swap(spawn_wave[a], spawn_wave[b]);
  std::swap(spawn_is_elite[a], spawn_is_elite[b]);
  std::swap(spawn_extra_hp_mult[a], spawn_extra_hp_mult[b]);
  std::swap(spawn_extra_speed_mult[a], spawn_extra_speed_mult[b]);
  std::swap(boss_phase[a], boss_phase[b]);
  std::swap(boss_stomp_cooldown_sec[a], boss_stomp_cooldown_sec[b]);
  std::swap(aura_speed_multiplier[a], aura_speed_multiplier[b]);
  std::swap(aura_damage_multiplier[a], aura_damage_multiplier[b]);
  std::swap(zone_speed_multiplier[a], zone_speed_multiplier[b]);
  std::swap(zone_damage_multiplier[a], zone_damage_multiplier[b]);
  std::swap(zone_heal_multiplier[a], zone_heal_multiplier[b]);
  std::swap(kill_credit_override[a], kill_credit_override[b]);
}

void EnemyContainer::resizeArrays(int new_capacity) {
  const int prev_capacity = capacity;
  RenderableEntityContainer::resizeArrays(new_capacity);
  if (capacity == prev_capacity) {
    return;
  }

  enemy_type.resize(capacity, count, EnemyType::Grub);
  hp.resize(capacity, count, 0.0f);
  hp_max.resize(capacity, count, 0.0f);
  move_speed_px_per_sec.resize(capacity, count, 0.0f);
  damage_to_base.resize(capacity, count, 0.0f);
  reward_essence.resize(capacity, count, 0);
  slow_multiplier.resize(capacity, count, 1.0f);
  slow_time_sec.resize(capacity, count, 0.0f);
  frozen_time_sec.resize(capacity, count, 0.0f);
  is_child.resize(capacity, count, 0);
  spawn_level.resize(capacity, count, 1);
  spawn_wave.resize(capacity, count, 0);
  spawn_is_elite.resize(capacity, count, 0);
  spawn_extra_hp_mult.resize(capacity, count, 1.0f);
  spawn_extra_speed_mult.resize(capacity, count, 1.0f);
  boss_phase.resize(capacity, count, 0);
  boss_stomp_cooldown_sec.resize(capacity, count, 0.0f);
  aura_speed_multiplier.resize(capacity, count, 1.0f);
  aura_damage_multiplier.resize(capacity, count, 1.0f);
  zone_speed_multiplier.resize(capacity, count, 1.0f);
  zone_damage_multiplier.resize(capacity, count, 1.0f);
  zone_heal_multiplier.resize(capacity, count, 1.0f);
  kill_credit_override.resize(capacity, count, INVALID_ID);
}

} // namespace tower_swarm
