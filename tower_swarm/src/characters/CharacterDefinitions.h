#pragma once

#include "Constants.h"
#include "characters/CharacterId.h"

#include <array>
#include <cstdint>
#include <string_view>

namespace tower_swarm {

struct CharacterBaseStats final {
  float base_hp = 1.0f;
  float base_damage = 1.0f;
  float base_range_px = 1.0f;
  float base_attack_rate_per_sec = 1.0f;
  float base_move_speed_px_per_sec = 1.0f;

  float splash_radius_px = 0.0f;
  float aura_radius_px = 0.0f;
  float aura_attack_speed = 0.0f;
  float aura_damage = 0.0f;
  float slow_field_radius_px = 0.0f;
  float drain_radius_px = 0.0f;
};

struct CharacterDefinition final {
  CharacterId id{};
  std::string_view name{};
  std::string_view lore{};
  CharacterRole role{};
  Rarity rarity{};
  CharacterBaseStats base{};

  std::array<std::string_view, 3> stage_names{};
  std::array<std::string_view, 3> stage_abilities{};
  std::string_view signature_name{};
  std::string_view signature_desc{};

  int unlock_level = 1;
  int unlock_shards = 0;
};

inline const CharacterDefinition &get_character_def(CharacterId id) {
  using namespace std::literals;

  static const CharacterDefinition kBrix{
      CharacterId::Brix,
      "Brix"sv,
      "Formed from the rubble of fallen walls, Brix remembers every battle."sv,
      CharacterRole::Shooter,
      Rarity::Common,
      CharacterBaseStats{characters::base_stats::kBrixBaseHp,
                         characters::base_stats::kBrixBaseDamage,
                         characters::base_stats::kBrixBaseRangePx,
                         characters::base_stats::kBrixBaseAttackRatePerSec,
                         characters::base_stats::kBrixBaseMoveSpeedPxPerSec},
      {"Rockshot"sv, "Stoneclaw"sv, "Titan Lord"sv},
      {"Shots pierce 1 enemy."sv, "Pierce 3, +20% range."sv,
       "Shots detonate on impact (mini-splash)."sv},
      "Avalanche"sv,
      "Every 15s, fires a massive boulder that knocks back enemies in a line."sv,
      1,
      0};

  static const CharacterDefinition kFlara{
      CharacterId::Flara,
      "Flara"sv,
      "She doesn't aim. She just burns everything in the way."sv,
      CharacterRole::Splasher,
      Rarity::Common,
      CharacterBaseStats{characters::base_stats::kFlaraBaseHp,
                         characters::base_stats::kFlaraBaseDamage,
                         characters::base_stats::kFlaraBaseRangePx,
                         characters::base_stats::kFlaraBaseAttackRatePerSec,
                         characters::base_stats::kFlaraBaseMoveSpeedPxPerSec,
                         characters::base_stats::kFlaraSplashRadiusPx},
      {"Emberkin"sv, "Blazeling"sv, "Inferno God"sv},
      {"Burning ground: lingering AoE."sv, "Burning ground lasts longer + slows."sv,
       "3 simultaneous blast targets."sv},
      "Conflagration"sv,
      "Every 20s, erupts into a firestorm dealing 5× damage in a large radius."sv,
      1,
      0};

  static const CharacterDefinition kMossling{
      CharacterId::Mossling,
      "Mossling"sv,
      "It doesn't attack. It makes sure everything around it does — better."sv,
      CharacterRole::Support,
      Rarity::Common,
      CharacterBaseStats{characters::base_stats::kMosslingBaseHp,
                         characters::base_stats::kMosslingBaseDamage,
                         characters::base_stats::kMosslingBaseRangePx,
                         characters::base_stats::kMosslingBaseAttackRatePerSec,
                         characters::base_stats::kMosslingBaseMoveSpeedPxPerSec,
                         0.0f,
                         characters::mossling::kAuraRadiusStage1Px,
                         characters::mossling::kAuraAttackSpeedStage1,
                         0.0f},
      {"Verdant"sv, "Grovekeeper"sv, "World Root"sv},
      {"Aura: +10% atk speed, +8% damage."sv, "Aura heals nearby creatures."sv,
       "Aura radius expands; also slows nearby enemies."sv},
      "Overgrowth"sv,
      "Every 25s, pulses and resets attack cooldowns of nearby creatures."sv,
      1,
      0};

  static const CharacterDefinition kGlitch{
      CharacterId::Glitch,
      "Glitch"sv,
      "It wasn't created. It escaped."sv,
      CharacterRole::Trapper,
      Rarity::Rare,
      CharacterBaseStats{characters::base_stats::kGlitchBaseHp,
                         characters::base_stats::kGlitchBaseDamage,
                         characters::base_stats::kGlitchBaseRangePx,
                         characters::base_stats::kGlitchBaseAttackRatePerSec,
                         characters::base_stats::kGlitchBaseMoveSpeedPxPerSec,
                         0.0f,
                         0.0f,
                         0.0f,
                         0.0f,
                         characters::base_stats::kGlitchSlowFieldRadiusPx},
      {"Nether Pulse"sv, "Signal Rend"sv, "Void Matrix"sv},
      {"Orbs also reduce enemy damage output."sv,
       "Orbs detonate after a delay, dealing burst damage."sv,
       "Orbs chain to adjacent enemies when they detonate."sv},
      "System Crash"sv,
      "Every 18s, freezes all enemies in range for 2.5s."sv,
      unlocks::kGlitchShopLevel,
      armory::kCharacterGlitchShardCost};

  static const CharacterDefinition kIronjaw{
      CharacterId::Ironjaw,
      "Ironjaw"sv,
      "It doesn't wait for enemies to come to it."sv,
      CharacterRole::Charger,
      Rarity::Rare,
      CharacterBaseStats{characters::base_stats::kIronjawBaseHp,
                         characters::base_stats::kIronjawBaseDamage,
                         characters::base_stats::kIronjawBaseRangePx,
                         characters::base_stats::kIronjawBaseAttackRatePerSec,
                         characters::base_stats::kIronjawBaseMoveSpeedPxPerSec},
      {"Ruststorm"sv, "Iron Colossus"sv, "Steel Leviathan"sv},
      {"Charge hits up to 3 enemies."sv, "Charge leaves shockwave trail."sv,
       "Charge becomes a rampage, hits all in its path."sv},
      "Override"sv,
      "Every 22s, enters a 4s frenzy: 3× attack speed and unlimited movement."sv,
      unlocks::kIronjawShopLevel,
      armory::kCharacterIronjawShardCost};

  static const CharacterDefinition kWraith{
      CharacterId::Wraith,
      "Wraith"sv,
      "By the time they see the shot, it's already over."sv,
      CharacterRole::Sniper,
      Rarity::Rare,
      CharacterBaseStats{characters::base_stats::kWraithBaseHp,
                         characters::base_stats::kWraithBaseDamage,
                         characters::base_stats::kWraithBaseRangePx,
                         characters::base_stats::kWraithBaseAttackRatePerSec,
                         characters::base_stats::kWraithBaseMoveSpeedPxPerSec},
      {"Darkshot"sv, "Phantom Arbiter"sv, "Reaper"sv},
      {"Arrows ignore 30% armor."sv, "Instakill enemies below 15% HP."sv,
       "Kills chain a bolt to the nearest enemy."sv},
      "Death Mark"sv,
      "Every 30s, marks one enemy; it dies after 4s regardless of HP."sv,
      unlocks::kWraithShopLevel,
      armory::kCharacterWraithShardCost};

  static const CharacterDefinition kCrystalis{
      CharacterId::Crystalis,
      "Crystalis"sv,
      "It doesn't know if it's a weapon or a temple. Maybe both."sv,
      CharacterRole::Hybrid,
      Rarity::Epic,
      CharacterBaseStats{characters::base_stats::kCrystalisBaseHp,
                         characters::base_stats::kCrystalisBaseDamage,
                         characters::base_stats::kCrystalisBaseRangePx,
                         characters::base_stats::kCrystalisBaseAttackRatePerSec,
                         characters::base_stats::kCrystalisBaseMoveSpeedPxPerSec},
      {"Prism Guard"sv, "Resonance Core"sv, "Cosmic Array"sv},
      {"Beams refract to hit 2 targets."sv, "Beams refract to hit 4 targets."sv,
       "Beams bounce between enemies until they die."sv},
      "Prismatic Nova"sv,
      "Every 20s, fires a 360° burst that hits all enemies on screen."sv,
      unlocks::kCrystalisShopLevel,
      armory::kCharacterCrystalisShardCost};

  static const CharacterDefinition kVex{
      CharacterId::Vex,
      "Vex"sv,
      "Unpredictable. Unreliable. Unstoppable."sv,
      CharacterRole::Chaos,
      Rarity::Epic,
      CharacterBaseStats{characters::base_stats::kVexBaseHp,
                         characters::base_stats::kVexBaseDamage,
                         characters::base_stats::kVexBaseRangePx,
                         characters::base_stats::kVexBaseAttackRatePerSec,
                         characters::base_stats::kVexBaseMoveSpeedPxPerSec},
      {"Malice"sv, "Dread Aura"sv, "Void Sovereign"sv},
      {"Random ability pool grows."sv, "Random abilities are 2× stronger."sv,
       "Abilities chain: each triggers the next."sv},
      "Entropy Storm"sv,
      "Every 25s, releases an uncontrolled wave of all random abilities."sv,
      unlocks::kVexShopLevel,
      armory::kCharacterVexShardCost};

  static const CharacterDefinition kOrin{
      CharacterId::Orin,
      "Orin"sv,
      "It predates the levels. It predates the swarm. It simply endures."sv,
      CharacterRole::Titan,
      Rarity::Legendary,
      CharacterBaseStats{characters::base_stats::kOrinBaseHp,
                         characters::base_stats::kOrinBaseDamage,
                         characters::base_stats::kOrinBaseRangePx,
                         characters::base_stats::kOrinBaseAttackRatePerSec,
                         characters::base_stats::kOrinBaseMoveSpeedPxPerSec},
      {"Orin (Awakened)"sv, "Orin (Ascendant)"sv, "Orin (Ascendant)"sv},
      {"Passive base shield improves; emits damage aura."sv,
       "Base shield reaches 25%; resets one creature HP when it would die."sv,
       "—"sv},
      "Temporal Ward"sv,
      "Every 60s, freezes all enemies on screen for 5s while creatures attack."sv,
      unlocks::kOrinUnlockLevel,
      armory::kCharacterOrinShardCost};

  static const CharacterDefinition kNull{
      CharacterId::NullSeed,
      "Null"sv,
      "It doesn't belong here. That's why it's perfect."sv,
      CharacterRole::Nullifier,
      Rarity::Legendary,
      CharacterBaseStats{characters::base_stats::kNullBaseHp,
                         characters::base_stats::kNullBaseDamage,
                         characters::base_stats::kNullBaseRangePx,
                         characters::base_stats::kNullBaseAttackRatePerSec,
                         characters::base_stats::kNullBaseMoveSpeedPxPerSec,
                         0.0f,
                         0.0f,
                         0.0f,
                         0.0f,
                         0.0f,
                         characters::base_stats::kNullDrainRadiusPx},
      {"Null (Expanding)"sv, "Null (Complete)"sv, "Null (Complete)"sv},
      {"Drains more enemy damage and speed."sv,
       "Enemies in range deal 0 damage; their kills credit Null."sv,
       "—"sv},
      "Consumption"sv,
      "Every 45s, absorbs the nearest enemy and gains HP equal to its max HP."sv,
      unlocks::kNullUnlockLevel,
      armory::kCharacterNullShardCost};

  switch (id) {
  case CharacterId::Brix:
    return kBrix;
  case CharacterId::Flara:
    return kFlara;
  case CharacterId::Mossling:
    return kMossling;
  case CharacterId::Glitch:
    return kGlitch;
  case CharacterId::Ironjaw:
    return kIronjaw;
  case CharacterId::Wraith:
    return kWraith;
  case CharacterId::Crystalis:
    return kCrystalis;
  case CharacterId::Vex:
    return kVex;
  case CharacterId::Orin:
    return kOrin;
  case CharacterId::NullSeed:
    return kNull;
  case CharacterId::Count:
    break;
  }
  return kBrix;
}

inline std::string_view get_stage_name(CharacterId id, int tier) {
  const CharacterDefinition &def = get_character_def(id);
  const int stage1_max =
      (id == CharacterId::Orin || id == CharacterId::NullSeed) ? 5
                                                               : characters::kEvolutionStage1MaxTier;
  const int stage2_max =
      (id == CharacterId::Orin || id == CharacterId::NullSeed) ? 9
                                                               : characters::kEvolutionStage2MaxTier;
  const int stage4_min = characters::kEvolutionStage4MinTier;

  if (tier >= stage4_min) {
    return def.stage_names[2];
  }
  if (tier > stage2_max) {
    return def.stage_names[1];
  }
  if (tier > stage1_max) {
    return def.stage_names[0];
  }
  return def.name;
}

} // namespace tower_swarm
