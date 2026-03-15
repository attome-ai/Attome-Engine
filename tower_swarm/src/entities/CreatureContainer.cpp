#include "entities/CreatureContainer.h"

#include "Constants.h"
#include "TowerSwarmMath.h"
#include "characters/CharacterDefinitions.h"
#include "entities/BaseEntity.h"
#include "entities/EnemyContainer.h"
#include "entities/ProjectileContainer.h"
#include "levels/GameState.h"
#include "systems/Evolution.h"
#include "systems/PathGrid.h"

#include "ATMEngine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

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

void top_left_for_tile_cell(int col, int row, int size_px, float &out_x,
                            float &out_y) {
  const int s = std::max(1, size_px);
  const float cell_cx = static_cast<float>(col * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float cell_cy = static_cast<float>(row * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float half = static_cast<float>(s) * 0.5f;

  const float max_x = std::max(0.0f, static_cast<float>(kWorldWidthPx - s));
  const float max_y = std::max(0.0f, static_cast<float>(kWorldHeightPx - s));

  out_x = std::clamp(cell_cx - half, 0.0f, max_x);
  out_y = std::clamp(cell_cy - half, 0.0f, max_y);
}

bool tile_cell_in_base_zone(int col, int row, float base_cx, float base_cy,
                            float avoid_r2) {
  const float cell_cx = static_cast<float>(col * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float cell_cy = static_cast<float>(row * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float dx = cell_cx - base_cx;
  const float dy = cell_cy - base_cy;
  return (dx * dx + dy * dy) <= avoid_r2;
}

struct MovementBlockCtx final {
  const std::vector<std::int32_t> *tile_occupant_slot{nullptr};
  int cols{0};
  int rows{0};
  std::int32_t self_slot{-1};
  int goal_idx{-1};
  float base_cx{0.0f};
  float base_cy{0.0f};
  float base_avoid_r2{0.0f};
};

bool movement_extra_blocked(int col, int row, void *user) {
  const auto *ctx = static_cast<const MovementBlockCtx *>(user);
  if (!ctx || !ctx->tile_occupant_slot || ctx->cols <= 0 || ctx->rows <= 0) {
    return true;
  }
  if (col < 0 || row < 0 || col >= ctx->cols || row >= ctx->rows) {
    return true;
  }
  const int idx = row * ctx->cols + col;
  if (idx < 0 || idx >= static_cast<int>(ctx->tile_occupant_slot->size())) {
    return true;
  }
  if (idx == ctx->goal_idx) {
    return false;
  }
  const std::int32_t occ =
      (*ctx->tile_occupant_slot)[static_cast<std::size_t>(idx)];
  if (occ != -1 && occ != ctx->self_slot) {
    return true;
  }
  return false;
}

float tier_pow(float base, int tier, float exponent) {
  const int t = std::max(tier, 1);
  return base * std::pow(static_cast<float>(t), exponent);
}

void compute_upgrade_multipliers(const GameState *state, CharacterId cid,
                                 float &out_hp_mult, float &out_dmg_mult,
                                 float &out_range_mult,
                                 float &out_rate_mult) {
  out_hp_mult = 1.0f;
  out_dmg_mult = 1.0f;
  out_range_mult = 1.0f;
  out_rate_mult = 1.0f;

  if (!state) {
    return;
  }

  int strike = 0;
  int vit = 0;
  int reach = 0;
  int tempo = 0;

  for (const RosterEntry &re : state->roster) {
    if (re.character != cid) {
      continue;
    }
    strike = std::max<int>(strike, re.upgrades[static_cast<std::size_t>(UpgradeNode::Strike)]);
    vit = std::max<int>(vit, re.upgrades[static_cast<std::size_t>(UpgradeNode::Vitality)]);
    reach = std::max<int>(reach, re.upgrades[static_cast<std::size_t>(UpgradeNode::Reach)]);
    tempo = std::max<int>(tempo, re.upgrades[static_cast<std::size_t>(UpgradeNode::Tempo)]);
  }

  strike = std::clamp(strike, 0, inter_level_shop::kUpgradeStrikeMaxRanks);
  vit = std::clamp(vit, 0, inter_level_shop::kUpgradeVitalityMaxRanks);
  reach = std::clamp(reach, 0, inter_level_shop::kUpgradeReachMaxRanks);
  tempo = std::clamp(tempo, 0, inter_level_shop::kUpgradeTempoMaxRanks);

  out_dmg_mult = 1.0f + inter_level_shop::kUpgradeStrikeDamagePerRank *
                           static_cast<float>(strike);
  out_hp_mult = 1.0f + inter_level_shop::kUpgradeVitalityHpPerRank *
                          static_cast<float>(vit);
  out_range_mult = 1.0f + inter_level_shop::kUpgradeReachRangePerRank *
                             static_cast<float>(reach);
  out_rate_mult = 1.0f + inter_level_shop::kUpgradeTempoAttackSpeedPerRank *
                            static_cast<float>(tempo);
}

void recalc_stats_for_slot(CreatureContainer &c, std::uint32_t slot, int new_tier) {
  if (slot >= static_cast<std::uint32_t>(c.count)) {
    return;
  }

  const CharacterId cid = c.character[slot];
  const CharacterDefinition &def = get_character_def(cid);
  const int t = std::max(1, new_tier);

  float hp_mult = 1.0f;
  float dmg_mult = 1.0f;
  float range_mult = 1.0f;
  float rate_mult = 1.0f;
  compute_upgrade_multipliers(c.gameState(), cid, hp_mult, dmg_mult, range_mult,
                              rate_mult);
  if (c.gameState() && c.gameState()->isRelicEquipped(RelicId::IronCore)) {
    hp_mult *= (1.0f + relics::kIronCoreHpBonus);
  }
  if (c.gameState()) {
    hp_mult *= std::max(0.0f, c.gameState()->ironResolveHpMultiplier());
  }

  const float old_max = std::max(0.0f, c.hp_max[slot]);
  const float frac = old_max > 0.0f ? (c.hp[slot] / old_max) : 1.0f;

  c.hp_max[slot] = tier_pow(def.base.base_hp, t, evolution::kHpExponent) * hp_mult;
  c.hp[slot] = std::clamp(frac * c.hp_max[slot], 0.0f, c.hp_max[slot]);

  c.attack_damage[slot] =
      tier_pow(def.base.base_damage, t, evolution::kDamageExponent) * dmg_mult;

  float range = tier_pow(def.base.base_range_px, t, evolution::kRangeExponent) *
                range_mult;
  if (cid == CharacterId::Brix && t >= 7) {
    range *= (1.0f + characters::brix::kStage3RangeBonus);
  }
  c.attack_range_px[slot] = std::min(range, evolution::kRangeCapPx);

  c.attack_rate_per_sec[slot] =
      std::min(tier_pow(def.base.base_attack_rate_per_sec, t,
                        evolution::kAttackRateExponent) *
                   rate_mult,
               evolution::kAttackRateCapPerSec);

  float spd = tier_pow(def.base.base_move_speed_px_per_sec, t,
                       evolution::kMoveSpeedExponent);
  if (c.gameState()) {
    spd *= std::max(0.0f, c.gameState()->kineticSwarmMoveSpeedMultiplier());
  }
  c.move_speed_px_per_sec[slot] = std::max(0.0f, spd);
}

std::uint32_t xorshift32(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

float signature_cooldown_for_character_sec(CharacterId cid) {
  switch (cid) {
  case CharacterId::Brix:
    return characters::brix::kSignatureCooldownSec;
  case CharacterId::Flara:
    return characters::flara::kSignatureCooldownSec;
  case CharacterId::Mossling:
    return characters::mossling::kSignatureCooldownSec;
  case CharacterId::Glitch:
    return characters::glitch::kSignatureCooldownSec;
  case CharacterId::Ironjaw:
    return characters::ironjaw::kSignatureCooldownSec;
  case CharacterId::Wraith:
    return characters::wraith::kSignatureCooldownSec;
  case CharacterId::Crystalis:
    return characters::crystalis::kSignatureCooldownSec;
  case CharacterId::Vex:
    return characters::vex::kSignatureCooldownSec;
  case CharacterId::Orin:
    return characters::orin::kSignatureCooldownSec;
  case CharacterId::NullSeed:
    return characters::null_seed::kSignatureCooldownSec;
  case CharacterId::Count:
    break;
  }
  return characters::brix::kSignatureCooldownSec;
}

float mossling_aura_radius_px(const CreatureContainer &c, int tier) {
  const int t = std::max(1, tier);
  float radius = 0.0f;
  if (t >= characters::kEvolutionStage4MinTier) {
    radius = characters::mossling::kAuraRadiusStage4Px;
  } else {
    radius = characters::mossling::kAuraRadiusStage1Px;
  }
  if (c.gameState() && c.gameState()->isRelicEquipped(RelicId::TwinPulse)) {
    radius += relics::kTwinPulseAuraRadiusBonusPx;
  }
  return std::max(0.0f, radius);
}

void compute_support_aura_bonuses(const CreatureContainer &c,
                                 std::uint32_t attacker_slot,
                                 float &out_damage_bonus,
                                 float &out_attack_speed_bonus,
                                 float &out_range_bonus) {
  out_damage_bonus = 0.0f;
  out_attack_speed_bonus = 0.0f;
  out_range_bonus = 0.0f;

  if (attacker_slot >= static_cast<std::uint32_t>(c.count)) {
    return;
  }

  const float ahalf = static_cast<float>(c.widths[attacker_slot]) * 0.5f;
  const Vec2 acenter =
      make_vec2(c.x_positions[attacker_slot] + ahalf,
                c.y_positions[attacker_slot] + ahalf);

  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(c.count); ++slot) {
    if (slot == attacker_slot) {
      continue;
    }
    if ((c.flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }
    const CharacterId cid = c.character[slot];
    const int t = std::max(1, c.tier[slot]);

    float radius = 0.0f;
    float dmg_bonus = 0.0f;
    float rate_bonus = 0.0f;
    float range_bonus = 0.0f;

    if (cid == CharacterId::Mossling) {
      radius = mossling_aura_radius_px(c, t);
      if (t >= 4) {
        rate_bonus = characters::mossling::kAuraAttackSpeedStage2;
        dmg_bonus = characters::mossling::kAuraDamageStage2;
      } else {
        rate_bonus = characters::mossling::kAuraAttackSpeedStage1;
      }
    } else if (cid == CharacterId::Crystalis) {
      radius = std::max(0.0f, c.attack_range_px[slot]);
      range_bonus = characters::base_stats::kCrystalisAuraRangeBoost;
      if (t >= 7) {
        dmg_bonus = characters::base_stats::kCrystalisAuraRangeBoost;
      }
    } else {
      continue;
    }

    if (radius <= 0.0f) {
      continue;
    }

    const float half = static_cast<float>(c.widths[slot]) * 0.5f;
    const Vec2 center = make_vec2(c.x_positions[slot] + half,
                                  c.y_positions[slot] + half);
    const Vec2 d = center - acenter;
    if (length_sq(d) > radius * radius) {
      continue;
    }

    out_damage_bonus += std::max(0.0f, dmg_bonus);
    out_attack_speed_bonus += std::max(0.0f, rate_bonus);
    out_range_bonus += std::max(0.0f, range_bonus);
  }
}

} // namespace

CreatureContainer::CreatureContainer(Engine *engine, int type_id,
                                     std::uint8_t default_layer,
                                     int initial_capacity)
    : RenderableEntityContainer(type_id, default_layer, initial_capacity),
      character(initial_capacity),
      roster_index(initial_capacity),
      tier(initial_capacity),
      kills(initial_capacity),
      hp(initial_capacity),
      hp_max(initial_capacity),
      attack_damage(initial_capacity),
      attack_range_px(initial_capacity),
      attack_rate_per_sec(initial_capacity),
      attack_cooldown_sec(initial_capacity),
      move_speed_px_per_sec(initial_capacity),
      signature_cooldown_sec(initial_capacity),
      ability_cooldown_sec(initial_capacity),
      clone_remaining_sec(initial_capacity),
      frenzy_remaining_sec(initial_capacity),
      rng_state(initial_capacity),
      state(initial_capacity),
      state_time_sec(initial_capacity),
      move_recalc_remaining_sec_(initial_capacity),
      move_segment_elapsed_sec_(initial_capacity),
      move_segment_duration_sec_(initial_capacity),
      move_segment_start_x_(initial_capacity),
      move_segment_start_y_(initial_capacity),
      move_segment_end_x_(initial_capacity),
      move_segment_end_y_(initial_capacity),
      move_waypoint_index_(initial_capacity),
      move_path_indices_(static_cast<std::size_t>(std::max(0, initial_capacity))),
      engine_(engine) {}

EntityHandle CreatureContainer::createCreature(float x, float y,
                                              CharacterId character_id,
                                              int tier_value, int kills_value,
                                              int roster_index_value) {
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

  const CharacterDefinition &def = get_character_def(character_id);
  const int t = std::max(1, tier_value);

  float hp_mult = 1.0f;
  float dmg_mult = 1.0f;
  float range_mult = 1.0f;
  float rate_mult = 1.0f;
  compute_upgrade_multipliers(game_state_, character_id, hp_mult, dmg_mult,
                              range_mult, rate_mult);
  if (game_state_ && game_state_->isRelicEquipped(RelicId::IronCore)) {
    hp_mult *= (1.0f + relics::kIronCoreHpBonus);
  }
  if (game_state_) {
    hp_mult *= std::max(0.0f, game_state_->ironResolveHpMultiplier());
  }

  character[slot] = character_id;
  roster_index[slot] = roster_index_value;
  tier[slot] = t;
  kills[slot] = std::max(0, kills_value);
  state[slot] = CreatureState::Idle;
  state_time_sec[slot] = 0.0f;

  hp_max[slot] =
      tier_pow(def.base.base_hp, t, evolution::kHpExponent) * hp_mult;
  hp[slot] = hp_max[slot];

  attack_damage[slot] =
      tier_pow(def.base.base_damage, t, evolution::kDamageExponent) * dmg_mult;

  float range = tier_pow(def.base.base_range_px, t, evolution::kRangeExponent) *
                range_mult;
  if (character_id == CharacterId::Brix && t >= 7) {
    range *= (1.0f + characters::brix::kStage3RangeBonus);
  }
  attack_range_px[slot] = std::min(range, evolution::kRangeCapPx);

  attack_rate_per_sec[slot] =
      std::min(tier_pow(def.base.base_attack_rate_per_sec, t,
                        evolution::kAttackRateExponent) *
                   rate_mult,
               evolution::kAttackRateCapPerSec);

  attack_cooldown_sec[slot] =
      attack_rate_per_sec[slot] > 0.0f ? (1.0f / attack_rate_per_sec[slot])
                                       : 0.0f;

  float spd = tier_pow(def.base.base_move_speed_px_per_sec, t,
                       evolution::kMoveSpeedExponent);
  if (game_state_) {
    spd *= std::max(0.0f, game_state_->kineticSwarmMoveSpeedMultiplier());
  }
  move_speed_px_per_sec[slot] = std::max(0.0f, spd);

  signature_cooldown_sec[slot] =
      std::max(0.0f, signature_cooldown_for_character_sec(character_id));
  ability_cooldown_sec[slot] =
      (character_id == CharacterId::Vex)
          ? std::max(0.0f, characters::vex::kRandomAbilityIntervalSec)
          : 0.0f;
  clone_remaining_sec[slot] = 0.0f;
  frenzy_remaining_sec[slot] = 0.0f;
  rng_state[slot] = 0x9E3779B9u ^ static_cast<std::uint32_t>(id) ^
                    (static_cast<std::uint32_t>(character_id) << 16);
  {
    const float interval = std::max(0.0f, movement_ai::kRecalcIntervalSec);
    const std::uint32_t r = xorshift32(rng_state[slot]);
    const float u = static_cast<float>(r & 0xFFFFu) * (1.0f / 65535.0f);
    move_recalc_remaining_sec_[slot] = u * interval;
  }
  move_segment_elapsed_sec_[slot] = 0.0f;
  move_segment_duration_sec_[slot] = movement_ai::kWaypointInterpSec;
  move_segment_start_x_[slot] = x;
  move_segment_start_y_[slot] = y;
  move_segment_end_x_[slot] = x;
  move_segment_end_y_[slot] = y;
  move_waypoint_index_[slot] = 0;
  if (slot < move_path_indices_.size()) {
    move_path_indices_[slot].clear();
  }

  const int size_px = evolution::creatureSizePxForTier(t);
  widths[slot] =
      static_cast<std::int16_t>(std::clamp(size_px, 1, 0x7FFF));
  heights[slot] = widths[slot];
  rotations[slot] = 0.0f;
  z_indices[slot] = kZIndexCreatures;

  int tex = -1;
  if (textures_) {
    const std::size_t idx = static_cast<std::size_t>(character_id);
    if (idx < textures_->size() && evolution::kVisualBandCount > 0) {
      const int band = std::clamp(evolution::visualBandIndexForTier(t), 0,
                                  evolution::kVisualBandCount - 1);
      tex = (*textures_)[idx][static_cast<std::size_t>(band)];
    }
  }
  texture_ids[slot] = static_cast<std::int16_t>(tex);

  flags[slot] |= static_cast<std::uint8_t>(EntityFlag::VISIBLE);
  move_with_grid(engine_, this, slot, x, y);
  return id;
}

CharacterId CreatureContainer::getCharacter(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return CharacterId::Brix;
  }
  return character[slot];
}

int CreatureContainer::getTier(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 1;
  }
  return tier[slot];
}

int CreatureContainer::getKills(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0;
  }
  return kills[slot];
}

float CreatureContainer::getHp(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0.0f;
  }
  return hp[slot];
}

float CreatureContainer::getHpMax(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return 0.0f;
  }
  return hp_max[slot];
}

int CreatureContainer::getRosterIndex(EntityHandle id) const {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return -1;
  }
  return roster_index[slot];
}

void CreatureContainer::schedule_destroy(std::uint32_t slot) {
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

void CreatureContainer::on_death(std::uint32_t slot, EntityHandle source_enemy) {
  (void)source_enemy;
  if (slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  schedule_destroy(slot);
}

bool CreatureContainer::applyDamage(EntityHandle id, float damage,
                                   EntityHandle source_enemy) {
  if (!engine_ || id == INVALID_ID || damage <= 0.0f) {
    return false;
  }
  if (!engine_is_handle_valid(engine_, id, getTypeId())) {
    return false;
  }

  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }
  if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
    return false;
  }

  float dmg = std::max(0.0f, damage);
  if (game_state_ && game_state_->hasWaveBuff(WaveBuffId::IronSkin)) {
    dmg *= std::max(0.0f, wave_shop::kIronSkinDamageTakenMultiplier);
  }

  hp[slot] = std::max(0.0f, hp[slot] - dmg);
  if (hp[slot] > 0.0f) {
    return false;
  }

  on_death(slot, source_enemy);
  return true;
}

void CreatureContainer::setAttackCooldown(EntityHandle id, float sec) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  attack_cooldown_sec[slot] = std::max(0.0f, sec);
}

void CreatureContainer::ensureAttackCooldownAtLeast(EntityHandle id, float sec) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return;
  }
  const float s = std::max(0.0f, sec);
  attack_cooldown_sec[slot] = std::max(attack_cooldown_sec[slot], s);
}

bool CreatureContainer::setWorldPosition(EntityHandle id, float x, float y) {
  if (!engine_) {
    return false;
  }
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }

  const int size_px = std::max(1, static_cast<int>(widths[slot]));
  const float max_x =
      std::max(0.0f, static_cast<float>(kWorldWidthPx - size_px));
  const float max_y =
      std::max(0.0f, static_cast<float>(kWorldHeightPx - size_px));
  const float nx = std::clamp(x, 0.0f, max_x);
  const float ny = std::clamp(y, 0.0f, max_y);

  move_with_grid(engine_, this, slot, nx, ny);

  move_recalc_remaining_sec_[slot] = movement_ai::kRecalcIntervalSec;
  move_segment_elapsed_sec_[slot] = 0.0f;
  move_segment_duration_sec_[slot] = movement_ai::kWaypointInterpSec;
  move_segment_start_x_[slot] = nx;
  move_segment_start_y_[slot] = ny;
  move_segment_end_x_[slot] = nx;
  move_segment_end_y_[slot] = ny;
  move_waypoint_index_[slot] = 0;
  if (slot < move_path_indices_.size()) {
    move_path_indices_[slot].clear();
  }
  if (state[slot] == CreatureState::Moving) {
    state[slot] = CreatureState::Idle;
  }

  return true;
}

bool CreatureContainer::moveToCell(EntityHandle id, int col, int row) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count)) {
    return false;
  }
  const int cols = kWorldWidthPx / kTileSizePx;
  const int rows = kWorldHeightPx / kTileSizePx;
  if (cols <= 0 || rows <= 0) {
    return false;
  }
  const int c = std::clamp(col, 0, cols - 1);
  const int r = std::clamp(row, 0, rows - 1);

  const int size_px = std::max(1, static_cast<int>(widths[slot]));
  const float cell_cx = static_cast<float>(c * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float cell_cy = static_cast<float>(r * kTileSizePx) +
                        static_cast<float>(kTileSizePx) * 0.5f;
  const float half = static_cast<float>(size_px) * 0.5f;

  const float max_x =
      std::max(0.0f, static_cast<float>(kWorldWidthPx - size_px));
  const float max_y =
      std::max(0.0f, static_cast<float>(kWorldHeightPx - size_px));

  const float x = std::clamp(cell_cx - half, 0.0f, max_x);
  const float y = std::clamp(cell_cy - half, 0.0f, max_y);
  return setWorldPosition(id, x, y);
}

void CreatureContainer::addKills(EntityHandle id, int delta) {
  const std::uint32_t slot = getSlot(id);
  if (slot == INVALID_ID || slot >= static_cast<std::uint32_t>(count) ||
      delta == 0) {
    return;
  }
  const int next = std::max(0, kills[slot] + delta);
  kills[slot] = next;

  bool resonant_growth_active = false;
  bool in_crystalis_aura = false;
  if (delta > 0 && game_state_ &&
      game_state_->isRelicEquipped(RelicId::ResonantGrowth)) {
    resonant_growth_active = true;

    const float self_half = static_cast<float>(widths[slot]) * 0.5f;
    const Vec2 self_center =
        make_vec2(x_positions[slot] + self_half, y_positions[slot] + self_half);

    for (std::uint32_t oslot = 0; oslot < static_cast<std::uint32_t>(count);
         ++oslot) {
      if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      if (character[oslot] != CharacterId::Crystalis) {
        continue;
      }
      const float radius = std::max(0.0f, attack_range_px[oslot]);
      if (radius <= 0.0f) {
        continue;
      }
      const float half = static_cast<float>(widths[oslot]) * 0.5f;
      const Vec2 center =
          make_vec2(x_positions[oslot] + half, y_positions[oslot] + half);
      if (length_sq(center - self_center) <= radius * radius) {
        in_crystalis_aura = true;
        break;
      }
    }
  }

  const int old_tier = std::max(1, tier[slot]);
  int tier_value = old_tier;
  int evolve_count = 0;
  for (;;) {
    int need = evolution::killsNeededForNextTier(tier_value);
    if (game_state_ && need > 0) {
      const float mult =
          std::clamp(game_state_->rapidGrowthKillThresholdMultiplier(), 0.01f, 1.0f);
      need = std::max(
          1, static_cast<int>(std::floor(static_cast<float>(need) * mult)));
    }

    int adj_need = need;
    if (resonant_growth_active && in_crystalis_aura && need > 0) {
      const float factor =
          std::clamp(1.0f - relics::kResonantGrowthEvolutionRateBonus, 0.0f, 1.0f);
      adj_need = std::max(
          1, static_cast<int>(std::floor(static_cast<float>(need) * factor)));
    }
    if (adj_need <= 0 || next < adj_need) {
      break;
    }
    if (tier_value >= std::numeric_limits<std::int32_t>::max()) {
      break;
    }
    tier_value += 1;
    evolve_count += 1;
  }
  if (evolve_count > 0) {
    tier[slot] = tier_value;
    recalc_stats_for_slot(*this, slot, tier_value);
    if (old_tier < 5 && tier_value >= 5) {
      signature_cooldown_sec[slot] = std::max(
          0.0f, signature_cooldown_for_character_sec(character[slot]));
    }
    state[slot] = CreatureState::Evolving;
    state_time_sec[slot] = evolution::kEvolutionAnimSec;

    if (game_state_) {
      const CharacterId cid = character[slot];
      const std::string_view stage = get_stage_name(cid, tier_value);
      std::string msg;
      msg.reserve(64);
      msg.append(to_string(cid));
      msg.append(" \xE2\x86\x92 ");
      msg.append(stage);
      msg.append("  TIER ");
      msg.append(std::to_string(tier_value));

      const float half = static_cast<float>(widths[slot]) * 0.5f;
      const float cx = x_positions[slot] + half;
      const float cy = y_positions[slot] - half * 0.25f;
      game_state_->floating_texts.push_back(
          FloatingText{cx, cy, evolution::kEvolutionFloatingTextSec, std::move(msg)});
      game_state_->screen_edge_glow_remaining_sec =
          std::max(game_state_->screen_edge_glow_remaining_sec,
                   evolution::kScreenEdgeGlowSec);
    }
  }

  if (game_state_) {
    const int ri = roster_index[slot];
    if (ri >= 0 && ri < static_cast<int>(game_state_->roster.size())) {
      game_state_->roster[static_cast<std::size_t>(ri)].kills = next;
      game_state_->roster[static_cast<std::size_t>(ri)].tier =
          std::max(1, tier[slot]);
      if (evolve_count > 0) {
        game_state_->evolutions_this_level += evolve_count;
      }
    }
  }
}

void CreatureContainer::recalcStatsForCharacter(CharacterId cid) {
  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }
    if (character[slot] != cid) {
      continue;
    }
    recalc_stats_for_slot(*this, slot, tier[slot]);
  }
}

void CreatureContainer::update(float delta_time) {
  const float dt = std::max(delta_time, 0.0f);
  const float pi = std::acos(-1.0f);

  if (!engine_ || !enemies_ || !projectiles_) {
    // Still tick basic timers for UI/animations.
    for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
      if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      attack_cooldown_sec[slot] =
          std::max(0.0f, attack_cooldown_sec[slot] - dt);
      state_time_sec[slot] = std::max(0.0f, state_time_sec[slot] - dt);
      if (state_time_sec[slot] <= 0.0f) {
        if (state[slot] == CreatureState::Evolving ||
            state[slot] == CreatureState::Attacking ||
            state[slot] == CreatureState::Merging) {
          state[slot] = CreatureState::Idle;
        }
      }
    }
    return;
  }

  auto creature_center = [&](std::uint32_t s) -> Vec2 {
    const float half = static_cast<float>(widths[s]) * 0.5f;
    return make_vec2(x_positions[s] + half, y_positions[s] + half);
  };

  auto enemy_center = [&](std::uint32_t eslot) -> Vec2 {
    const float half = static_cast<float>(enemies_->widths[eslot]) * 0.5f;
    return make_vec2(enemies_->x_positions[eslot] + half,
                     enemies_->y_positions[eslot] + half);
  };

  auto find_nearest_enemy = [&](const Vec2 &ccenter, float range,
                               EntityHandle &out_enemy,
                               Vec2 &out_center) -> bool {
    out_enemy = INVALID_ID;
    out_center = make_vec2(0.0f, 0.0f);
    if (range <= 0.0f) {
      return false;
    }

    const float query_r = range + static_cast<float>(kEnemyBaseSizePx) * 0.5f;
    const auto &refs = engine_->grid.queryCircle(ccenter.x, ccenter.y, query_r);
    float best_d2 = range * range;

    for (const EntityRef &ref : refs) {
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
      if ((enemies_->flags[eslot] &
           static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const Vec2 ecenter = enemy_center(eslot);
      const Vec2 d = ecenter - ccenter;
      const float d2 = length_sq(d);
      if (d2 <= best_d2) {
        best_d2 = d2;
        out_enemy = ref.index;
        out_center = ecenter;
      }
    }

    return out_enemy != INVALID_ID;
  };

  auto projectile_tex = [&](CharacterId cid) -> int {
    if (!projectile_textures_) {
      return -1;
    }
    const std::size_t idx = static_cast<std::size_t>(cid);
    return idx < projectile_textures_->size() ? (*projectile_textures_)[idx]
                                              : -1;
  };

  auto spawn_shot = [&](const Vec2 &start_center, const Vec2 &target_center,
                        CharacterId shooter_cid, EntityHandle source_id,
                        float dmg, int pierce, float splash) {
    const Vec2 dir = safe_normalize(target_center - start_center);
    (void)projectiles_->spawnProjectile(
        start_center.x - static_cast<float>(kProjectileSizePx) * 0.5f,
        start_center.y - static_cast<float>(kProjectileSizePx) * 0.5f,
        dir.x * kProjectileSpeedPxPerSec, dir.y * kProjectileSpeedPxPerSec, dmg,
        kProjectileDefaultLifetimeSec, pierce, splash, source_id,
        projectile_tex(shooter_cid));
  };

  const int tile_cols = kWorldWidthPx / kTileSizePx;
  const int tile_rows = kWorldHeightPx / kTileSizePx;
  const int tile_cell_count = std::max(0, tile_cols * tile_rows);

  static std::vector<std::int32_t> tile_occupant_slot{};
  if (static_cast<int>(tile_occupant_slot.size()) != tile_cell_count) {
    tile_occupant_slot.assign(static_cast<std::size_t>(tile_cell_count), -1);
  } else {
    std::fill(tile_occupant_slot.begin(), tile_occupant_slot.end(), -1);
  }

  if (tile_cell_count > 0 && tile_cols > 0 && tile_rows > 0) {
    for (std::uint32_t oslot = 0; oslot < static_cast<std::uint32_t>(count);
         ++oslot) {
      if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const float half = static_cast<float>(widths[oslot]) * 0.5f;
      const float cx = x_positions[oslot] + half;
      const float cy = y_positions[oslot] + half;
      const int col = std::clamp(static_cast<int>(cx / kTileSizePx), 0,
                                 std::max(0, tile_cols - 1));
      const int row = std::clamp(static_cast<int>(cy / kTileSizePx), 0,
                                 std::max(0, tile_rows - 1));
      const int idx = row * tile_cols + col;
      if (idx < 0 || idx >= tile_cell_count) {
        continue;
      }
      const std::int32_t cur = tile_occupant_slot[static_cast<std::size_t>(idx)];
      if (cur < 0) {
        tile_occupant_slot[static_cast<std::size_t>(idx)] =
            static_cast<std::int32_t>(oslot);
        continue;
      }
      const int cur_t = std::max(1, tier[static_cast<std::uint32_t>(cur)]);
      const int new_t = std::max(1, tier[oslot]);
      if (new_t > cur_t) {
        tile_occupant_slot[static_cast<std::size_t>(idx)] =
            static_cast<std::int32_t>(oslot);
      }
    }
  }

  Vec2 base_center = make_vec2(static_cast<float>(kWorldWidthPx) * 0.5f,
                               static_cast<float>(kWorldHeightPx) * 0.5f);
  if (base_ && base_id_ != INVALID_ID) {
    const std::uint32_t bslot = base_->getSlot(base_id_);
    if (bslot != INVALID_ID && bslot < static_cast<std::uint32_t>(base_->count)) {
      const float half = static_cast<float>(base_->widths[bslot]) * 0.5f;
      base_center =
          make_vec2(base_->x_positions[bslot] + half, base_->y_positions[bslot] + half);
    }
  }

  const bool pack_instinct_active =
      game_state_ && game_state_->isRelicEquipped(RelicId::PackInstinct);
  std::array<float, static_cast<std::size_t>(CharacterId::Count)> pack_rate_mult{};
  pack_rate_mult.fill(1.0f);
  if (pack_instinct_active) {
    std::array<int, static_cast<std::size_t>(CharacterId::Count)> type_count{};
    type_count.fill(0);
    for (std::uint32_t oslot = 0; oslot < static_cast<std::uint32_t>(count);
         ++oslot) {
      if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const std::size_t idx = static_cast<std::size_t>(character[oslot]);
      if (idx < type_count.size()) {
        type_count[idx] += 1;
      }
    }
    for (std::size_t i = 0; i < type_count.size(); ++i) {
      const int groups = std::max(0, type_count[i] / 3);
      if (groups > 0) {
        pack_rate_mult[i] =
            1.0f + relics::kPackInstinctAttackSpeedPer3SameType * static_cast<float>(groups);
      }
    }
  }

  const bool apex_hunger_active =
      game_state_ && game_state_->isRelicEquipped(RelicId::ApexHunger);
  int apex_best_roster_index = -1;
  int apex_best_kills = -1;
  if (apex_hunger_active) {
    for (std::uint32_t oslot = 0; oslot < static_cast<std::uint32_t>(count);
         ++oslot) {
      if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
        continue;
      }
      const int ri = roster_index[oslot];
      if (ri < 0) {
        continue;
      }
      const int k = kills[oslot];
      if (k > apex_best_kills) {
        apex_best_kills = k;
        apex_best_roster_index = ri;
      }
    }
  }

  for (std::uint32_t slot = 0; slot < static_cast<std::uint32_t>(count); ++slot) {
    if ((flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
      continue;
    }

    attack_cooldown_sec[slot] = std::max(0.0f, attack_cooldown_sec[slot] - dt);
    clone_remaining_sec[slot] = std::max(0.0f, clone_remaining_sec[slot] - dt);
    frenzy_remaining_sec[slot] = std::max(0.0f, frenzy_remaining_sec[slot] - dt);

    state_time_sec[slot] = std::max(0.0f, state_time_sec[slot] - dt);
    if (state_time_sec[slot] <= 0.0f) {
      if (state[slot] == CreatureState::Evolving ||
          state[slot] == CreatureState::Attacking ||
          state[slot] == CreatureState::Merging) {
        state[slot] = CreatureState::Idle;
      }
    }

    const CharacterId cid = character[slot];
    const int t = std::max(1, tier[slot]);
    const EntityHandle self_id = getStableId(slot);
    if (self_id == INVALID_ID) {
      continue;
    }

    if (textures_) {
      const std::size_t tidx = static_cast<std::size_t>(cid);
      if (tidx < textures_->size() && evolution::kVisualBandCount > 0) {
        const int band = std::clamp(evolution::visualBandIndexForTier(t), 0,
                                    evolution::kVisualBandCount - 1);
        texture_ids[slot] = static_cast<std::int16_t>(
            (*textures_)[tidx][static_cast<std::size_t>(band)]);
      }
    }

    float pulse_scale = 1.0f;
    if (state[slot] == CreatureState::Evolving &&
        evolution::kEvolutionAnimSec > 0.0f && state_time_sec[slot] > 0.0f) {
      const float progress = std::clamp(
          1.0f - (state_time_sec[slot] / evolution::kEvolutionAnimSec), 0.0f, 1.0f);
      pulse_scale = 1.0f + (evolution::kEvolutionPulseScale - 1.0f) *
                              std::sin(pi * progress);
    }
    const int desired_size = evolution::creatureSizePxForTier(t, pulse_scale);
    const int cur_size = std::max(1, static_cast<int>(widths[slot]));
    if (desired_size != cur_size) {
      const float cur_half = static_cast<float>(cur_size) * 0.5f;
      const float cx = x_positions[slot] + cur_half;
      const float cy = y_positions[slot] + cur_half;

      widths[slot] =
          static_cast<std::int16_t>(std::clamp(desired_size, 1, 0x7FFF));
      heights[slot] = widths[slot];

      const float new_half = static_cast<float>(desired_size) * 0.5f;
      float nx = cx - new_half;
      float ny = cy - new_half;
      const float max_x =
          std::max(0.0f, static_cast<float>(kWorldWidthPx - desired_size));
      const float max_y =
          std::max(0.0f, static_cast<float>(kWorldHeightPx - desired_size));
      nx = std::clamp(nx, 0.0f, max_x);
      ny = std::clamp(ny, 0.0f, max_y);
      move_with_grid(engine_, this, slot, nx, ny);
    }

    if (state[slot] == CreatureState::Moving) {
      if (slot >= move_path_indices_.size()) {
        state[slot] = CreatureState::Idle;
        move_waypoint_index_[slot] = 0;
      } else {
        auto &path = move_path_indices_[slot];
        std::uint16_t wp = move_waypoint_index_[slot];
        if (path.size() < 2 || wp == 0 || wp >= path.size()) {
          state[slot] = CreatureState::Idle;
          move_waypoint_index_[slot] = 0;
          path.clear();
          move_recalc_remaining_sec_[slot] = movement_ai::kRecalcIntervalSec;
        } else {
          move_segment_elapsed_sec_[slot] =
              std::max(0.0f, move_segment_elapsed_sec_[slot] + dt);
          const float dur = std::max(0.0001f, move_segment_duration_sec_[slot]);
          const float tprog =
              std::clamp(move_segment_elapsed_sec_[slot] / dur, 0.0f, 1.0f);
          const float nx = lerpf(move_segment_start_x_[slot],
                                 move_segment_end_x_[slot], tprog);
          const float ny = lerpf(move_segment_start_y_[slot],
                                 move_segment_end_y_[slot], tprog);
          move_with_grid(engine_, this, slot, nx, ny);

          if (tprog >= 1.0f) {
            move_segment_start_x_[slot] = move_segment_end_x_[slot];
            move_segment_start_y_[slot] = move_segment_end_y_[slot];
            move_segment_elapsed_sec_[slot] = 0.0f;

            const std::uint16_t next_wp = static_cast<std::uint16_t>(wp + 1u);
            move_waypoint_index_[slot] = next_wp;
            wp = next_wp;

            if (wp >= path.size()) {
              state[slot] = CreatureState::Idle;
              move_waypoint_index_[slot] = 0;
              path.clear();
              move_recalc_remaining_sec_[slot] = movement_ai::kRecalcIntervalSec;
            } else if (tile_cols > 0 && tile_rows > 0) {
              const int cell = static_cast<int>(path[wp]);
              const int col =
                  std::clamp(cell % tile_cols, 0, std::max(0, tile_cols - 1));
              const int row =
                  std::clamp(cell / tile_cols, 0, std::max(0, tile_rows - 1));
              const int idx = row * tile_cols + col;
              const std::int32_t occ =
                  (idx >= 0 && idx < tile_cell_count)
                      ? tile_occupant_slot[static_cast<std::size_t>(idx)]
                      : -1;
              if (occ != -1 && occ != static_cast<std::int32_t>(slot)) {
                state[slot] = CreatureState::Idle;
                move_waypoint_index_[slot] = 0;
                path.clear();
                move_recalc_remaining_sec_[slot] = 0.0f;
              } else {
                float ex = 0.0f;
                float ey = 0.0f;
                top_left_for_tile_cell(col, row, static_cast<int>(widths[slot]), ex,
                                       ey);
                move_segment_end_x_[slot] = ex;
                move_segment_end_y_[slot] = ey;
              }
            }
          }
        }
      }
    }

    float dmg_bonus = 0.0f;
    float rate_bonus = 0.0f;
    float range_bonus = 0.0f;
    compute_support_aura_bonuses(*this, slot, dmg_bonus, rate_bonus, range_bonus);

    float dmg = std::max(0.0f, attack_damage[slot]) * (1.0f + dmg_bonus);
    float rate = std::max(0.0f, attack_rate_per_sec[slot]) * (1.0f + rate_bonus);
    float range =
        std::max(0.0f, attack_range_px[slot]) * (1.0f + range_bonus);
    range = std::min(range, evolution::kRangeCapPx);

    if (game_state_ && game_state_->isRelicEquipped(RelicId::Bloodshard)) {
      dmg *= (1.0f + relics::kBloodshardDamagePerTier * static_cast<float>(t));
    }
    if (pack_instinct_active) {
      const std::size_t idx = static_cast<std::size_t>(cid);
      if (idx < pack_rate_mult.size()) {
        rate *= pack_rate_mult[idx];
      }
    }

    if (game_state_ && game_state_->hasWaveBuff(WaveBuffId::ApexHunter) &&
        roster_index[slot] >= 0 &&
        roster_index[slot] == game_state_->apex_hunter_roster_index) {
      dmg *= (1.0f + wave_shop::kApexHunterDamageBonus);
    }
    if (apex_hunger_active && roster_index[slot] >= 0 &&
        roster_index[slot] == apex_best_roster_index) {
      dmg *= (1.0f + relics::kApexHungerDamageBonus);
    }
    if (game_state_ && game_state_->hasWaveBuff(WaveBuffId::Surge)) {
      rate *= (1.0f + wave_shop::kSurgeAttackSpeedBonus);
    }

    const Vec2 ccenter = creature_center(slot);

    if (t >= 5) {
      signature_cooldown_sec[slot] =
          std::max(0.0f, signature_cooldown_sec[slot] - dt);
    }
    if (cid == CharacterId::Vex) {
      ability_cooldown_sec[slot] =
          std::max(0.0f, ability_cooldown_sec[slot] - dt);
    }

    // Mossling aura heal (Stage 7+).
    if (cid == CharacterId::Mossling && t >= 7) {
      const float ar = mossling_aura_radius_px(*this, t);
      if (ar > 0.0f) {
        const float r2 = ar * ar;
        for (std::uint32_t oslot = 0;
             oslot < static_cast<std::uint32_t>(count); ++oslot) {
          if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const Vec2 ocenter = creature_center(oslot);
          if (length_sq(ocenter - ccenter) > r2) {
            continue;
          }
          const float mh = std::max(0.0f, hp_max[oslot]);
          const float heal =
              characters::mossling::kAuraHealStage3HpPerSec * dt;
          if (heal > 0.0f) {
            hp[oslot] = std::clamp(hp[oslot] + heal, 0.0f, mh);
          }
        }
      }
    }

    if (state[slot] == CreatureState::Idle) {
      move_recalc_remaining_sec_[slot] =
          std::max(0.0f, move_recalc_remaining_sec_[slot] - dt);
      if (move_recalc_remaining_sec_[slot] <= 0.0f && path_grid_ &&
          tile_cols > 0 && tile_rows > 0 && tile_cell_count > 0 &&
          path_grid_->cols() == tile_cols && path_grid_->rows() == tile_rows) {
        move_recalc_remaining_sec_[slot] = movement_ai::kRecalcIntervalSec;

        const float near_r = movement_ai::kThreatRadiusNearPx;
        const float mid_r = movement_ai::kThreatRadiusMidPx;
        const float far_r = movement_ai::kThreatRadiusFarPx;
        const float near_r2 = near_r * near_r;
        const float mid_r2 = mid_r * mid_r;
        const float far_r2 = far_r * far_r;

        Vec2 sum_pos = make_vec2(0.0f, 0.0f);
        float sum_w = 0.0f;
        int enemy_seen = 0;

        const auto &threat_refs =
            engine_->grid.queryCircle(ccenter.x, ccenter.y, far_r);
        for (const EntityRef &ref : threat_refs) {
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
          if ((enemies_->flags[eslot] &
               static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const Vec2 ecenter = enemy_center(eslot);
          const float d2 = length_sq(ecenter - ccenter);
          if (d2 > far_r2) {
            continue;
          }
          const float w = (d2 <= near_r2)   ? movement_ai::kThreatWeightNear
                          : (d2 <= mid_r2) ? movement_ai::kThreatWeightMid
                                           : movement_ai::kThreatWeightFar;
          sum_pos = sum_pos + (ecenter * w);
          sum_w += w;
          ++enemy_seen;
        }

        if (enemy_seen > 0 && sum_w > 0.0f) {
          const Vec2 centroid = sum_pos * (1.0f / sum_w);

          Vec2 repulse_sum = make_vec2(0.0f, 0.0f);
          const float support_r = movement_ai::kSupportRepelRadiusPx;
          const float support_r2 = support_r * support_r;
          const auto &support_refs =
              engine_->grid.queryCircle(ccenter.x, ccenter.y, support_r);
          for (const EntityRef &ref : support_refs) {
            if (static_cast<int>(ref.type) != getTypeId()) {
              continue;
            }
            if (ref.index == self_id) {
              continue;
            }
            if (!engine_is_handle_valid(engine_, ref.index, getTypeId())) {
              continue;
            }
            const std::uint32_t oslot = getSlot(ref.index);
            if (oslot == INVALID_ID ||
                oslot >= static_cast<std::uint32_t>(count)) {
              continue;
            }
            if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const Vec2 ocenter = creature_center(oslot);
            const Vec2 d = ccenter - ocenter;
            const float d2 = length_sq(d);
            if (d2 <= 1e-6f || d2 > support_r2) {
              continue;
            }
            const float dist = std::sqrt(d2);
            const float scale = (support_r - dist) / support_r;
            repulse_sum = repulse_sum + (safe_normalize(d) * scale);
          }

          Vec2 repulse = make_vec2(0.0f, 0.0f);
          const float repulse_len = length(repulse_sum);
          if (repulse_len > 1e-6f) {
            const float strength = std::min(1.0f, repulse_len);
            repulse = safe_normalize(repulse_sum) *
                      (static_cast<float>(kTileSizePx) * strength);
          }

          const Vec2 desired = centroid + repulse;
          const int desired_col = std::clamp(static_cast<int>(desired.x / kTileSizePx),
                                             0, std::max(0, tile_cols - 1));
          const int desired_row = std::clamp(static_cast<int>(desired.y / kTileSizePx),
                                             0, std::max(0, tile_rows - 1));

          const float cr = static_cast<float>(widths[slot]) * 0.5f;
          const float base_avoid_r = kBaseRadiusPx + cr * 1.25f;
          const float base_avoid_r2 = base_avoid_r * base_avoid_r;

          auto cell_valid = [&](int col, int row) -> bool {
            if (col < 0 || row < 0 || col >= tile_cols || row >= tile_rows) {
              return false;
            }
            if (base_avoid_r2 > 0.0f &&
                tile_cell_in_base_zone(col, row, base_center.x, base_center.y,
                                       base_avoid_r2)) {
              return false;
            }
            if (path_grid_ && !path_grid_->isWalkable(col, row)) {
              return false;
            }
            const int idx = row * tile_cols + col;
            if (idx < 0 || idx >= tile_cell_count) {
              return false;
            }
            const std::int32_t occ =
                tile_occupant_slot[static_cast<std::size_t>(idx)];
            return (occ < 0) || (occ == static_cast<std::int32_t>(slot));
          };

          int goal_col = desired_col;
          int goal_row = desired_row;
          if (!cell_valid(goal_col, goal_row)) {
            const int max_r = static_cast<int>(std::ceil(
                movement_ai::kThreatRadiusFarPx / static_cast<float>(kTileSizePx)));
            bool found = false;
            for (int r = 1; !found && r <= max_r; ++r) {
              const int min_c = std::max(0, desired_col - r);
              const int max_c = std::min(std::max(0, tile_cols - 1), desired_col + r);
              const int min_r = std::max(0, desired_row - r);
              const int max_rv = std::min(std::max(0, tile_rows - 1), desired_row + r);
              for (int row = min_r; !found && row <= max_rv; ++row) {
                for (int col = min_c; !found && col <= max_c; ++col) {
                  if (r > 0 && col != min_c && col != max_c && row != min_r &&
                      row != max_rv) {
                    continue;
                  }
                  if (cell_valid(col, row)) {
                    goal_col = col;
                    goal_row = row;
                    found = true;
                  }
                }
              }
            }
            if (!found) {
              goal_col = desired_col;
              goal_row = desired_row;
            }
          }

          if (cell_valid(goal_col, goal_row)) {
            const float goal_cx = static_cast<float>(goal_col * kTileSizePx) +
                                  static_cast<float>(kTileSizePx) * 0.5f;
            const float goal_cy = static_cast<float>(goal_row * kTileSizePx) +
                                  static_cast<float>(kTileSizePx) * 0.5f;
            const float thresh = movement_ai::kDesiredMoveThresholdPx;
            if (length_sq(make_vec2(goal_cx, goal_cy) - ccenter) > thresh * thresh) {
              const int start_col =
                  std::clamp(static_cast<int>(ccenter.x / kTileSizePx), 0,
                             std::max(0, tile_cols - 1));
              const int start_row =
                  std::clamp(static_cast<int>(ccenter.y / kTileSizePx), 0,
                             std::max(0, tile_rows - 1));

              std::vector<PathGrid::Cell> cell_path{};
              cell_path.reserve(32);

              MovementBlockCtx ctx{};
              ctx.tile_occupant_slot = &tile_occupant_slot;
              ctx.cols = tile_cols;
              ctx.rows = tile_rows;
              ctx.self_slot = static_cast<std::int32_t>(slot);
              ctx.goal_idx = goal_row * tile_cols + goal_col;
              ctx.base_cx = base_center.x;
              ctx.base_cy = base_center.y;
              ctx.base_avoid_r2 = base_avoid_r2;

              const bool ok = path_grid_->findPath(
                  PathGrid::Cell{static_cast<std::int16_t>(start_col),
                                 static_cast<std::int16_t>(start_row)},
                  PathGrid::Cell{static_cast<std::int16_t>(goal_col),
                                 static_cast<std::int16_t>(goal_row)},
                  cell_path, &movement_extra_blocked, &ctx);

              if (ok && slot < move_path_indices_.size()) {
                auto &path = move_path_indices_[slot];
                path.clear();
                path.reserve(cell_path.size());
                for (const PathGrid::Cell &c : cell_path) {
                  const int idx =
                      static_cast<int>(c.row) * tile_cols + static_cast<int>(c.col);
                  if (idx < 0 || idx >= tile_cell_count) {
                    continue;
                  }
                  path.push_back(static_cast<std::uint16_t>(idx));
                }

                if (path.size() >= 2) {
                  state[slot] = CreatureState::Moving;
                  move_waypoint_index_[slot] = 1;
                  move_segment_elapsed_sec_[slot] = 0.0f;
                  move_segment_duration_sec_[slot] = movement_ai::kWaypointInterpSec;
                  move_segment_start_x_[slot] = x_positions[slot];
                  move_segment_start_y_[slot] = y_positions[slot];

                  const int cell = static_cast<int>(path[1]);
                  const int col =
                      std::clamp(cell % tile_cols, 0, std::max(0, tile_cols - 1));
                  const int row =
                      std::clamp(cell / tile_cols, 0, std::max(0, tile_rows - 1));
                  float ex = 0.0f;
                  float ey = 0.0f;
                  top_left_for_tile_cell(col, row, static_cast<int>(widths[slot]), ex,
                                         ey);
                  move_segment_end_x_[slot] = ex;
                  move_segment_end_y_[slot] = ey;
                }
              }
            }
          }
        }
      }
    }

    if (state[slot] == CreatureState::Moving) {
      continue;
    }

    auto fire_vex_ability = [&](int which, float strength) {
      if (self_id == INVALID_ID) {
        return;
      }
      if (which == 1) { // Teleport
        const int ox =
            static_cast<int>(xorshift32(rng_state[slot]) % 3u) - 1;
        const int oy =
            static_cast<int>(xorshift32(rng_state[slot]) % 3u) - 1;
        const float dx =
            static_cast<float>(ox) * static_cast<float>(GRID_CELL_SIZE);
        const float dy =
            static_cast<float>(oy) * static_cast<float>(GRID_CELL_SIZE);
        const float nx =
            std::clamp(x_positions[slot] + dx, 0.0f,
                       std::max(0.0f, static_cast<float>(kWorldWidthPx - widths[slot])));
        const float ny =
            std::clamp(y_positions[slot] + dy, 0.0f,
                       std::max(0.0f, static_cast<float>(kWorldHeightPx - heights[slot])));
        move_with_grid(engine_, this, slot, nx, ny);
        return;
      }

      if (which == 4) { // Clone
        const float dur =
            std::max(0.0f, characters::glitch::kSlowFieldDurationSec * strength);
        clone_remaining_sec[slot] = std::max(clone_remaining_sec[slot], dur);
        return;
      }

      if (which == 2) { // Slow pulse
        const float radius = characters::base_stats::kNullDrainRadiusPx;
        const float dur =
            std::max(0.0f, characters::glitch::kSlowFieldDurationSec * strength);
        const float mult = characters::glitch::kSlowFieldSpeedMultiplier;
        if (radius <= 0.0f || dur <= 0.0f) {
          return;
        }
        const float r2 = radius * radius;
        for (std::uint32_t eslot = 0;
             eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
          if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const Vec2 ecenter = enemy_center(eslot);
          if (length_sq(ecenter - ccenter) > r2) {
            continue;
          }
          enemies_->slow_multiplier[eslot] =
              std::min(enemies_->slow_multiplier[eslot], mult);
          enemies_->slow_time_sec[eslot] =
              std::max(enemies_->slow_time_sec[eslot], dur);
        }
        return;
      }

      EntityHandle target = INVALID_ID;
      Vec2 tcenter{};
      if (!find_nearest_enemy(ccenter, range, target, tcenter)) {
        return;
      }

      if (which == 0) { // Mini-explosion
        const float radius = characters::base_stats::kFlaraSplashRadiusPx;
        const float dd = std::max(0.0f, dmg * strength);
        if (radius <= 0.0f || dd <= 0.0f) {
          return;
        }
        const float r2 = radius * radius;
        for (std::uint32_t eslot = 0;
             eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
          if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const EntityHandle eid = enemies_->getStableId(eslot);
          if (eid == INVALID_ID) {
            continue;
          }
          const Vec2 ecenter = enemy_center(eslot);
          if (length_sq(ecenter - tcenter) > r2) {
            continue;
          }
          enemies_->applyDamage(eid, dd, self_id);
        }
        return;
      }

      if (which == 3) { // Lightning strike
        const float radius = characters::base_stats::kGlitchSlowFieldRadiusPx;
        const float dd = std::max(0.0f, dmg * strength);
        if (radius <= 0.0f || dd <= 0.0f) {
          return;
        }
        const float r2 = radius * radius;
        for (std::uint32_t eslot = 0;
             eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
          if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const EntityHandle eid = enemies_->getStableId(eslot);
          if (eid == INVALID_ID) {
            continue;
          }
          const Vec2 ecenter = enemy_center(eslot);
          if (length_sq(ecenter - tcenter) > r2) {
            continue;
          }
          enemies_->applyDamage(eid, dd, self_id);
        }
        return;
      }

    };

    // Vex random abilities tick.
    if (cid == CharacterId::Vex && ability_cooldown_sec[slot] <= 0.0f) {
      const int base_pool_size = (t >= 4) ? 5 : 3;
      int pool_size = base_pool_size;
      if (game_state_ && game_state_->isRelicEquipped(RelicId::ChaosSpark)) {
        const int lvl = std::max(1, game_state_->level_number);
        if (lvl > 30) {
          const int extra = std::max(0, lvl - 30) *
                            std::max(0, relics::kChaosSparkExtraOptionsPerLevelAbove30);
          if (extra > 0 && extra < 1000000) {
            pool_size = std::max(base_pool_size, base_pool_size + extra);
          }
        }
      }
      const float strength =
          (t >= 7) ? characters::vex::kStage3AbilityStrengthMultiplier : 1.0f;

      auto fire_roll = [&](int roll) {
        if (roll < base_pool_size) {
          fire_vex_ability(roll, strength);
          return;
        }

        const int a = static_cast<int>(
            xorshift32(rng_state[slot]) % static_cast<std::uint32_t>(base_pool_size));
        int b = a;
        if (base_pool_size > 1) {
          while (b == a) {
            b = static_cast<int>(
                xorshift32(rng_state[slot]) % static_cast<std::uint32_t>(base_pool_size));
          }
        }
        fire_vex_ability(a, strength);
        fire_vex_ability(b, strength);
      };

      const int first_roll =
          static_cast<int>(xorshift32(rng_state[slot]) % static_cast<std::uint32_t>(pool_size));
      fire_roll(first_roll);
      if (t >= characters::kEvolutionStage4MinTier) {
        int second_roll = first_roll;
        if (pool_size > 1) {
          while (second_roll == first_roll) {
            second_roll = static_cast<int>(
                xorshift32(rng_state[slot]) % static_cast<std::uint32_t>(pool_size));
          }
        }
        fire_roll(second_roll);
      }
      ability_cooldown_sec[slot] = characters::vex::kRandomAbilityIntervalSec;
    }

    // Signatures (Tier 5+).
    if (t >= 5 && signature_cooldown_sec[slot] <= 0.0f) {
      switch (cid) {
      case CharacterId::Brix: {
        EntityHandle target = INVALID_ID;
        Vec2 tcenter{};
        if (find_nearest_enemy(ccenter, range, target, tcenter)) {
          const Vec2 dir = safe_normalize(tcenter - ccenter);
          const float len = characters::brix::kSignatureLineLengthPx;
          const float width = characters::brix::kStage4SplashRadiusPx;
          const float width2 = width * width;
          for (std::uint32_t eslot = 0;
               eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
            if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const EntityHandle eid = enemies_->getStableId(eslot);
            if (eid == INVALID_ID) {
              continue;
            }
            const Vec2 ecenter = enemy_center(eslot);
            const Vec2 v = ecenter - ccenter;
            const float proj = dot(v, dir);
            if (proj < 0.0f || proj > len) {
              continue;
            }
            const float perp2 = std::max(0.0f, length_sq(v) - proj * proj);
            if (perp2 <= width2) {
              enemies_->displace(eid, dir.x * len, dir.y * len);
            }
          }
        }
        break;
      }
      case CharacterId::Flara: {
        const float r = characters::flara::kSignatureRadiusPx;
        const float dd =
            std::max(0.0f, dmg * characters::flara::kSignatureDamageMultiplier);
        if (r > 0.0f && dd > 0.0f) {
          const float r2 = r * r;
          for (std::uint32_t eslot = 0;
               eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
            if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const EntityHandle eid = enemies_->getStableId(eslot);
            if (eid == INVALID_ID) {
              continue;
            }
            const Vec2 ecenter = enemy_center(eslot);
            if (length_sq(ecenter - ccenter) > r2) {
              continue;
            }
            enemies_->applyDamage(eid, dd, self_id);
          }
        }
        break;
      }
      case CharacterId::Mossling: {
        const float r = mossling_aura_radius_px(*this, t);
        if (r > 0.0f) {
          const float r2 = r * r;
          for (std::uint32_t oslot = 0;
               oslot < static_cast<std::uint32_t>(count); ++oslot) {
            if ((flags[oslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const Vec2 ocenter = creature_center(oslot);
            if (length_sq(ocenter - ccenter) > r2) {
              continue;
            }
            attack_cooldown_sec[oslot] = 0.0f;
          }
        }
        break;
      }
      case CharacterId::Glitch: {
        const float r = characters::glitch::kSignatureFreezeRadiusPx;
        if (r > 0.0f) {
          const float r2 = r * r;
          for (std::uint32_t eslot = 0;
               eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
            if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
              continue;
            }
            const Vec2 ecenter = enemy_center(eslot);
            if (length_sq(ecenter - ccenter) > r2) {
              continue;
            }
            enemies_->frozen_time_sec[eslot] = std::max(
                enemies_->frozen_time_sec[eslot],
                characters::glitch::kSignatureFreezeDurationSec);
          }
        }
        break;
      }
      case CharacterId::Ironjaw: {
        frenzy_remaining_sec[slot] =
            std::max(frenzy_remaining_sec[slot],
                     characters::ironjaw::kSignatureFrenzyDurationSec);
        break;
      }
      case CharacterId::Wraith: {
        EntityHandle target = INVALID_ID;
        Vec2 tcenter{};
        if (find_nearest_enemy(ccenter, range, target, tcenter)) {
          enemies_->scheduleDelayedKill(
              target, characters::wraith::kSignatureMarkDurationSec, self_id);
        }
        break;
      }
      case CharacterId::Crystalis: {
        for (std::uint32_t eslot = 0;
             eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
          if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          const EntityHandle eid = enemies_->getStableId(eslot);
          if (eid == INVALID_ID) {
            continue;
          }
          enemies_->applyDamage(eid, dmg, self_id);
        }
        break;
      }
      case CharacterId::Vex: {
        const int pool_size = (t >= 4) ? 5 : 3;
        const float strength =
            (t >= 7) ? characters::vex::kStage3AbilityStrengthMultiplier : 1.0f;
        for (int a = 0; a < pool_size; ++a) {
          fire_vex_ability(a, strength);
        }
        break;
      }
      case CharacterId::Orin: {
        for (std::uint32_t eslot = 0;
             eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
          if ((enemies_->flags[eslot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
            continue;
          }
          enemies_->frozen_time_sec[eslot] = std::max(
              enemies_->frozen_time_sec[eslot],
              characters::orin::kSignatureFreezeDurationSec);
        }
        break;
      }
      case CharacterId::NullSeed: {
        EntityHandle target = INVALID_ID;
        Vec2 tcenter{};
        const float rr = characters::base_stats::kNullDrainRadiusPx;
        if (find_nearest_enemy(ccenter, rr, target, tcenter)) {
          const float ehp = std::max(0.0f, enemies_->getHpMax(target));
          enemies_->applyDamage(target, ehp + 1.0f, self_id);
          hp[slot] = std::clamp(hp[slot] + ehp, 0.0f, hp_max[slot]);
        }
        break;
      }
      case CharacterId::Count:
        break;
      }

      signature_cooldown_sec[slot] =
          std::max(0.0f, signature_cooldown_for_character_sec(cid));
    }

    if (attack_cooldown_sec[slot] > 0.0f) {
      continue;
    }

    if (cid == CharacterId::Mossling) {
      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    if (cid == CharacterId::Glitch) {
      Vec2 drop_center = ccenter;
      EntityHandle target = INVALID_ID;
      Vec2 tcenter{};
      if (find_nearest_enemy(ccenter, range, target, tcenter)) {
        drop_center = tcenter;
      }

      if (game_state_) {
        EffectZone z{};
        z.kind = EffectZoneKind::GlitchOrb;
        z.world_x = drop_center.x;
        z.world_y = drop_center.y;
        z.radius_px = characters::base_stats::kGlitchSlowFieldRadiusPx;
        z.age_sec = 0.0f;
        z.speed_multiplier = characters::glitch::kSlowFieldSpeedMultiplier;
        z.slow_duration_sec = characters::glitch::kSlowFieldDurationSec;
        z.damage_multiplier =
            (t >= 4) ? characters::glitch::kSlowFieldSpeedMultiplier : 1.0f;
        z.owner_creature = self_id;

        if (t >= 7) {
          z.detonate_after_sec = characters::glitch::kOrbDetonateAfterSec;
          z.lifetime_sec = z.detonate_after_sec;
          z.detonate_damage = dmg;
        } else {
          z.lifetime_sec = z.slow_duration_sec;
        }
        if (t >= characters::kEvolutionStage4MinTier) {
          z.chain_hops_remaining = 1;
        }

        game_state_->effect_zones.push_back(z);
      }

      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    if (cid == CharacterId::Ironjaw) {
      const bool frenzy = frenzy_remaining_sec[slot] > 0.0f;
      float charge_range = characters::base_stats::kIronjawChargeRangePx;
      if (frenzy) {
        charge_range = static_cast<float>(kWorldWidthPx + kWorldHeightPx);
        rate *= characters::ironjaw::kSignatureAttackSpeedMultiplier;
      }

      EntityHandle target = INVALID_ID;
      Vec2 tcenter{};
      if (!find_nearest_enemy(ccenter, charge_range, target, tcenter)) {
        attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
        continue;
      }

      const Vec2 dir = safe_normalize(tcenter - ccenter);
      const float knock = characters::base_stats::kIronjawChargeKnockbackPx;
      const float hit_radius =
          (t >= 7) ? knock : static_cast<float>(kEnemyBaseSizePx) * 0.5f;
      const int hit_cap =
          (t >= characters::kEvolutionStage4MinTier) ? 0x7FFFFFFF
          : (t >= 4) ? 3
                     : 1;

      const float ratio =
          characters::base_stats::kIronjawChargeDamage /
          std::max(0.001f, get_character_def(cid).base.base_damage);
      const float charge_dmg = std::max(0.0f, dmg * ratio);

      const Vec2 start = ccenter;
      const float dist = std::min(charge_range, length(tcenter - ccenter));
      const Vec2 end = start + dir * dist;

      const float self_half = static_cast<float>(widths[slot]) * 0.5f;
      const Vec2 next_pos = end - make_vec2(self_half, self_half);
      move_with_grid(engine_, this, slot, next_pos.x, next_pos.y);

      struct Hit final {
        EntityHandle id{INVALID_ID};
        float proj{0.0f};
      };
      std::array<Hit, 128> hits{};
      int hit_count = 0;
      const float rad2 = hit_radius * hit_radius;

      for (std::uint32_t eslot = 0;
           eslot < static_cast<std::uint32_t>(enemies_->count); ++eslot) {
        if ((enemies_->flags[eslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const EntityHandle eid = enemies_->getStableId(eslot);
        if (eid == INVALID_ID) {
          continue;
        }
        const Vec2 ecenter = enemy_center(eslot);
        const Vec2 v = ecenter - start;
        const float proj = dot(v, dir);
        if (proj < 0.0f || proj > dist) {
          continue;
        }
        const float perp2 = std::max(0.0f, length_sq(v) - proj * proj);
        if (perp2 > rad2) {
          continue;
        }
        if (hit_count < static_cast<int>(hits.size())) {
          hits[hit_count++] = Hit{eid, proj};
        }
      }

      std::sort(hits.begin(), hits.begin() + hit_count,
                [](const Hit &a, const Hit &b) { return a.proj < b.proj; });

      int applied = 0;
      for (int i = 0; i < hit_count; ++i) {
        if (applied >= hit_cap) {
          break;
        }
        if (hits[i].id == INVALID_ID) {
          continue;
        }
        enemies_->applyDamage(hits[i].id, charge_dmg, self_id);
        enemies_->displace(hits[i].id, dir.x * knock, dir.y * knock);
        applied++;
      }

      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    EntityHandle best = INVALID_ID;
    Vec2 bcenter{};
    if (!find_nearest_enemy(ccenter, range, best, bcenter)) {
      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    if (cid == CharacterId::Flara) {
      const int shots =
          (t >= characters::kEvolutionStage4MinTier)
              ? characters::flara::kStage4SimultaneousTargets
              : 1;
      const auto &refs = engine_->grid.queryCircle(ccenter.x, ccenter.y, range);

      std::array<EntityHandle, characters::flara::kStage4SimultaneousTargets>
          targets{};
      std::array<float, characters::flara::kStage4SimultaneousTargets> d2s{};
      int found = 0;
      for (int i = 0; i < shots; ++i) {
        targets[i] = INVALID_ID;
        d2s[i] = std::numeric_limits<float>::max();
      }

      for (const EntityRef &ref : refs) {
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
        if ((enemies_->flags[eslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const Vec2 ecenter = enemy_center(eslot);
        const float d2 = length_sq(ecenter - ccenter);
        for (int i = 0; i < shots; ++i) {
          if (d2 < d2s[i]) {
            for (int j = shots - 1; j > i; --j) {
              targets[j] = targets[j - 1];
              d2s[j] = d2s[j - 1];
            }
            targets[i] = ref.index;
            d2s[i] = d2;
            found = std::min(found + 1, shots);
            break;
          }
        }
      }

      for (int i = 0; i < found; ++i) {
        const std::uint32_t eslot = enemies_->getSlot(targets[i]);
        if (eslot == INVALID_ID ||
            eslot >= static_cast<std::uint32_t>(enemies_->count)) {
          continue;
        }
        spawn_shot(ccenter, enemy_center(eslot), cid, self_id, dmg, 0,
                   characters::base_stats::kFlaraSplashRadiusPx);
      }
      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    if (cid == CharacterId::Crystalis) {
      if (t >= characters::kEvolutionStage4MinTier) {
        const Vec2 dir = safe_normalize(bcenter - ccenter);
        (void)projectiles_->spawnProjectile(
            ccenter.x - static_cast<float>(kProjectileSizePx) * 0.5f,
            ccenter.y - static_cast<float>(kProjectileSizePx) * 0.5f,
            dir.x * kProjectileSpeedPxPerSec, dir.y * kProjectileSpeedPxPerSec,
            dmg, std::numeric_limits<float>::max(), 0, 0.0f, self_id,
            projectile_tex(cid), -1, range);
        attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
        continue;
      }
      const int shots = (t >= 7) ? characters::crystalis::kStage3RefractTargets
                      : (t >= 4) ? characters::crystalis::kStage2RefractTargets
                                 : 1;
      const auto &refs = engine_->grid.queryCircle(ccenter.x, ccenter.y, range);

      std::array<EntityHandle, 4> targets{};
      std::array<float, 4> d2s{};
      int found = 0;
      for (int i = 0; i < shots; ++i) {
        targets[i] = INVALID_ID;
        d2s[i] = std::numeric_limits<float>::max();
      }

      for (const EntityRef &ref : refs) {
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
        if ((enemies_->flags[eslot] &
             static_cast<std::uint8_t>(EntityFlag::VISIBLE)) == 0) {
          continue;
        }
        const Vec2 ecenter = enemy_center(eslot);
        const float d2 = length_sq(ecenter - ccenter);
        for (int i = 0; i < shots; ++i) {
          if (d2 < d2s[i]) {
            for (int j = shots - 1; j > i; --j) {
              targets[j] = targets[j - 1];
              d2s[j] = d2s[j - 1];
            }
            targets[i] = ref.index;
            d2s[i] = d2;
            found = std::min(found + 1, shots);
            break;
          }
        }
      }

      for (int i = 0; i < found; ++i) {
        const std::uint32_t eslot = enemies_->getSlot(targets[i]);
        if (eslot == INVALID_ID ||
            eslot >= static_cast<std::uint32_t>(enemies_->count)) {
          continue;
        }
        spawn_shot(ccenter, enemy_center(eslot), cid, self_id, dmg, 1, 0.0f);
      }

      attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
      continue;
    }

    if (cid == CharacterId::Brix) {
      int pierce = 0;
      float splash = 0.0f;
      if (t >= 7) {
        pierce = characters::brix::kPierceStage3;
      } else if (t >= 4) {
        pierce = characters::brix::kPierceStage2;
      }
      if (t >= characters::kEvolutionStage4MinTier) {
        splash = characters::brix::kStage4SplashRadiusPx;
      }
      spawn_shot(ccenter, bcenter, cid, self_id, dmg, pierce, splash);
    } else {
      spawn_shot(ccenter, bcenter, cid, self_id, dmg, 0, 0.0f);
      if (cid == CharacterId::Vex && clone_remaining_sec[slot] > 0.0f) {
        spawn_shot(ccenter, bcenter, cid, self_id, dmg, 0, 0.0f);
      }
    }

    attack_cooldown_sec[slot] = rate > 0.0f ? (1.0f / rate) : 0.0f;
  }
}

void CreatureContainer::swapSlots(std::uint32_t a, std::uint32_t b) {
  if (a == b) {
    return;
  }

  RenderableEntityContainer::swapSlots(a, b);
  std::swap(character[a], character[b]);
  std::swap(roster_index[a], roster_index[b]);
  std::swap(tier[a], tier[b]);
  std::swap(kills[a], kills[b]);
  std::swap(hp[a], hp[b]);
  std::swap(hp_max[a], hp_max[b]);
  std::swap(attack_damage[a], attack_damage[b]);
  std::swap(attack_range_px[a], attack_range_px[b]);
  std::swap(attack_rate_per_sec[a], attack_rate_per_sec[b]);
  std::swap(attack_cooldown_sec[a], attack_cooldown_sec[b]);
  std::swap(move_speed_px_per_sec[a], move_speed_px_per_sec[b]);
  std::swap(signature_cooldown_sec[a], signature_cooldown_sec[b]);
  std::swap(ability_cooldown_sec[a], ability_cooldown_sec[b]);
  std::swap(clone_remaining_sec[a], clone_remaining_sec[b]);
  std::swap(frenzy_remaining_sec[a], frenzy_remaining_sec[b]);
  std::swap(rng_state[a], rng_state[b]);
  std::swap(state[a], state[b]);
  std::swap(state_time_sec[a], state_time_sec[b]);
  std::swap(move_recalc_remaining_sec_[a], move_recalc_remaining_sec_[b]);
  std::swap(move_segment_elapsed_sec_[a], move_segment_elapsed_sec_[b]);
  std::swap(move_segment_duration_sec_[a], move_segment_duration_sec_[b]);
  std::swap(move_segment_start_x_[a], move_segment_start_x_[b]);
  std::swap(move_segment_start_y_[a], move_segment_start_y_[b]);
  std::swap(move_segment_end_x_[a], move_segment_end_x_[b]);
  std::swap(move_segment_end_y_[a], move_segment_end_y_[b]);
  std::swap(move_waypoint_index_[a], move_waypoint_index_[b]);
  if (a < move_path_indices_.size() && b < move_path_indices_.size()) {
    std::swap(move_path_indices_[a], move_path_indices_[b]);
  }
}

void CreatureContainer::resizeArrays(int new_capacity) {
  const int prev_capacity = capacity;
  RenderableEntityContainer::resizeArrays(new_capacity);
  if (capacity == prev_capacity) {
    return;
  }

  character.resize(capacity, count);
  roster_index.resize(capacity, count, -1);
  tier.resize(capacity, count, 1);
  kills.resize(capacity, count, 0);
  hp.resize(capacity, count, 0.0f);
  hp_max.resize(capacity, count, 0.0f);
  attack_damage.resize(capacity, count, 0.0f);
  attack_range_px.resize(capacity, count, 0.0f);
  attack_rate_per_sec.resize(capacity, count, 0.0f);
  attack_cooldown_sec.resize(capacity, count, 0.0f);
  move_speed_px_per_sec.resize(capacity, count, 0.0f);
  signature_cooldown_sec.resize(capacity, count, 0.0f);
  ability_cooldown_sec.resize(capacity, count, 0.0f);
  clone_remaining_sec.resize(capacity, count, 0.0f);
  frenzy_remaining_sec.resize(capacity, count, 0.0f);
  rng_state.resize(capacity, count, 0u);
  state.resize(capacity, count, CreatureState::Idle);
  state_time_sec.resize(capacity, count, 0.0f);

  move_recalc_remaining_sec_.resize(capacity, count, 0.0f);
  move_segment_elapsed_sec_.resize(capacity, count, 0.0f);
  move_segment_duration_sec_.resize(capacity, count, movement_ai::kWaypointInterpSec);
  move_segment_start_x_.resize(capacity, count, 0.0f);
  move_segment_start_y_.resize(capacity, count, 0.0f);
  move_segment_end_x_.resize(capacity, count, 0.0f);
  move_segment_end_y_.resize(capacity, count, 0.0f);
  move_waypoint_index_.resize(capacity, count, 0u);

  move_path_indices_.resize(static_cast<std::size_t>(std::max(0, capacity)));
}

} // namespace tower_swarm
