#pragma once

#include "Constants.h"
#include "characters/CharacterId.h"
#include "shop/RelicId.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tower_swarm {

enum class UpgradeNode : std::uint8_t {
  Strike = 0,
  Vitality = 1,
  Reach = 2,
  Tempo = 3,
  Signature = 4,
  Count = 5
};

enum class MasteryId : std::uint8_t {
  EchoFoundation = 0,
  NexusVault = 1,
  RapidGrowth = 2,
  KineticSwarm = 3,
  SynthesisMastery = 4,
  IronResolve = 5,
  VoidAppetite = 6,
  ShardEye = 7,
  Count = 8
};

enum class WaveBuffId : std::uint8_t {
  Surge = 0,
  Fortify = 1,
  FrenziedBlood = 2,
  SlowTide = 3,
  Foresight = 4,
  Mend = 5,
  WildSeed = 6,
  EchoStrike = 7,
  EssenceCache = 8,
  IronSkin = 9,
  ApexHunter = 10,
  VoidPulse = 11,
  Count = 12,
};

struct ActiveWaveBuff final {
  WaveBuffId id{WaveBuffId::Surge};
  std::int32_t remaining_waves{0};
};

struct FloatingText final {
  float world_x{0.0f};
  float world_y{0.0f};
  float remaining_sec{0.0f};
  std::string text{};
};

enum class EffectZoneKind : std::uint8_t {
  BurningGround = 0,
  GlitchOrb = 1,
};

struct EffectZone final {
  EffectZoneKind kind{EffectZoneKind::BurningGround};
  float world_x{0.0f};
  float world_y{0.0f};
  float radius_px{0.0f};

  float age_sec{0.0f};
  float lifetime_sec{0.0f};

  // Continuous effects (applied while active).
  float damage_per_sec{0.0f};
  float speed_multiplier{1.0f};
  float damage_multiplier{1.0f};

  // Timed burst (used by Glitch orbs).
  float slow_duration_sec{0.0f};
  float detonate_after_sec{0.0f};
  float detonate_damage{0.0f};
  bool detonated{false};
  std::int32_t chain_hops_remaining{0};

  std::uint32_t owner_creature{0xFFFFFFFFu};
};

struct RosterEntry final {
  CharacterId character{CharacterId::Brix};
  std::int32_t tier{1};
  std::int32_t kills{0};
  std::int32_t seed_cost_essence{0};
  std::array<std::uint8_t, static_cast<std::size_t>(UpgradeNode::Count)>
      upgrades{};
};

struct GameState final {
  // Persistent (save) fields (Phase 3+).
  std::int32_t max_level_reached{1};
  std::vector<std::uint8_t> stars_per_level{};
  std::int32_t essence{0};
  std::int32_t base_hp{level::kBaseHp};
  std::int32_t shards{0};
  std::int32_t player_level{1};
  std::int32_t player_xp{0};

  std::int32_t lifetime_levels_completed{0};
  std::int32_t lifetime_stars_earned{0};
  std::int32_t lifetime_bosses_killed{0};
  std::int32_t lifetime_merges{0};

  std::array<std::uint8_t, static_cast<std::size_t>(CharacterId::Count)>
      unlocked_characters{};
  std::array<std::uint8_t, static_cast<std::size_t>(MasteryId::Count)>
      mastery_ranks{};

  std::vector<RosterEntry> roster{};
  std::array<std::uint8_t, static_cast<std::size_t>(RelicId::Count)>
      relic_unlocked{};
  std::array<RelicId, relics::kSlotCount> equipped_relics{
      RelicId::None, RelicId::None, RelicId::None};

  // Runtime (per-level) fields.
  std::int32_t level_number{1};
  std::int32_t wave_index{0};
  std::int32_t wave_count{0};
  bool is_elite{false};
  float level_time_sec{0.0f};
  std::int32_t enemies_killed_this_level{0};
  std::int32_t essence_earned_this_level{0};
  std::int32_t merges_this_level{0};
  std::int32_t evolutions_this_level{0};
  std::int32_t next_level_base_hp_target{std::numeric_limits<std::int32_t>::min()};
  bool any_enemy_reached_base_this_level{false};
  bool eternal_echo_used_this_level{false};
  std::int32_t the_quiet_bonus_stars_cosmetic{0};

  std::vector<ActiveWaveBuff> active_wave_buffs{};
  std::int32_t void_pulse_kill_counter{0};
  std::int32_t apex_hunter_roster_index{-1};

  std::vector<FloatingText> floating_texts{};
  float screen_edge_glow_remaining_sec{0.0f};
  std::vector<EffectZone> effect_zones{};

  bool isRelicUnlocked(RelicId id) const {
    const std::size_t idx = static_cast<std::size_t>(id);
    if (id == RelicId::None || idx >= relic_unlocked.size()) {
      return false;
    }
    return relic_unlocked[idx] != 0;
  }

  bool isRelicEquipped(RelicId id) const {
    if (id == RelicId::None) {
      return false;
    }
    for (const RelicId cur : equipped_relics) {
      if (cur == id) {
        return true;
      }
    }
    return false;
  }

  bool hasWaveBuff(WaveBuffId id) const {
    for (const auto &b : active_wave_buffs) {
      if (b.id == id && b.remaining_waves > 0) {
        return true;
      }
    }
    return false;
  }

  void addOrRefreshWaveBuff(WaveBuffId id, std::int32_t duration_waves) {
    const std::int32_t waves = std::max<std::int32_t>(0, duration_waves);
    if (waves <= 0) {
      return;
    }
    for (auto &b : active_wave_buffs) {
      if (b.id == id) {
        b.remaining_waves = std::max(b.remaining_waves, waves);
        return;
      }
    }
    active_wave_buffs.push_back(ActiveWaveBuff{id, waves});
  }

  void onWaveStarted() {
    void_pulse_kill_counter = 0;
    apex_hunter_roster_index = -1;
  }

  void onWaveEnded() {
    for (auto &b : active_wave_buffs) {
      b.remaining_waves = std::max<std::int32_t>(0, b.remaining_waves - 1);
    }
    active_wave_buffs.erase(
        std::remove_if(active_wave_buffs.begin(), active_wave_buffs.end(),
                       [](const ActiveWaveBuff &b) {
                         return b.remaining_waves <= 0;
                       }),
        active_wave_buffs.end());
  }

  void clearWaveBuffs() {
    active_wave_buffs.clear();
    void_pulse_kill_counter = 0;
    apex_hunter_roster_index = -1;
  }

  void resetToNewProfile() {
    max_level_reached = 1;
    stars_per_level.clear();
    essence = 0;
    base_hp = level::kBaseHp;
    shards = 0;
    player_level = 1;
    player_xp = 0;
    lifetime_levels_completed = 0;
    lifetime_stars_earned = 0;
    lifetime_bosses_killed = 0;
    lifetime_merges = 0;
    unlocked_characters.fill(0);
    mastery_ranks.fill(0);
    roster.clear();
    relic_unlocked.fill(0);
    equipped_relics.fill(RelicId::None);

    RosterEntry brix{};
    brix.character = CharacterId::Brix;
    brix.tier = 1;
    brix.kills = 0;
    brix.seed_cost_essence = 0;

    RosterEntry flara{};
    flara.character = CharacterId::Flara;
    flara.tier = 1;
    flara.kills = 0;
    flara.seed_cost_essence = 0;

    RosterEntry moss{};
    moss.character = CharacterId::Mossling;
    moss.tier = 1;
    moss.kills = 0;
    moss.seed_cost_essence = 0;

    roster.push_back(brix);
    roster.push_back(flara);
    roster.push_back(moss);

    clearWaveBuffs();
    effect_zones.clear();

    unlocked_characters[static_cast<std::size_t>(CharacterId::Brix)] = 1;
    unlocked_characters[static_cast<std::size_t>(CharacterId::Flara)] = 1;
    unlocked_characters[static_cast<std::size_t>(CharacterId::Mossling)] = 1;

    // Relics (start unlocked).
    relic_unlocked[static_cast<std::size_t>(RelicId::IronCore)] = 1;
    relic_unlocked[static_cast<std::size_t>(RelicId::Bloodshard)] = 1;

    recomputeMetaProgression();
  }

  static int masteryMaxRanks(MasteryId id) {
    switch (id) {
    case MasteryId::EchoFoundation:
      return armory::kEchoFoundationRanks;
    case MasteryId::NexusVault:
      return armory::kNexusVaultRanks;
    case MasteryId::RapidGrowth:
      return armory::kRapidGrowthRanks;
    case MasteryId::KineticSwarm:
      return armory::kKineticSwarmRanks;
    case MasteryId::SynthesisMastery:
      return armory::kSynthesisMasteryRanks;
    case MasteryId::IronResolve:
      return armory::kIronResolveRanks;
    case MasteryId::VoidAppetite:
      return armory::kVoidAppetiteRanks;
    case MasteryId::ShardEye:
      return armory::kShardEyeRanks;
    case MasteryId::Count:
      break;
    }
    return 0;
  }

  static int masteryNextRankCost(MasteryId id, int current_rank) {
    const int next = std::clamp(current_rank + 1, 1, 3);
    switch (id) {
    case MasteryId::EchoFoundation:
      return (next == 1)   ? armory::kEchoFoundationCostR1
             : (next == 2) ? armory::kEchoFoundationCostR2
                           : armory::kEchoFoundationCostR3;
    case MasteryId::NexusVault:
      return (next == 1)   ? armory::kNexusVaultCostR1
             : (next == 2) ? armory::kNexusVaultCostR2
                           : armory::kNexusVaultCostR3;
    case MasteryId::RapidGrowth:
      return (next == 1)   ? armory::kRapidGrowthCostR1
             : (next == 2) ? armory::kRapidGrowthCostR2
                           : armory::kRapidGrowthCostR3;
    case MasteryId::KineticSwarm:
      return (next == 1)   ? armory::kKineticSwarmCostR1
             : (next == 2) ? armory::kKineticSwarmCostR2
                           : armory::kKineticSwarmCostR3;
    case MasteryId::SynthesisMastery:
      return (next == 1)   ? armory::kSynthesisMasteryCostR1
             : (next == 2) ? armory::kSynthesisMasteryCostR2
                           : armory::kSynthesisMasteryCostR3;
    case MasteryId::IronResolve:
      return (next == 1)   ? armory::kIronResolveCostR1
             : (next == 2) ? armory::kIronResolveCostR2
                           : armory::kIronResolveCostR3;
    case MasteryId::VoidAppetite:
      return (next == 1)   ? armory::kVoidAppetiteCostR1
             : (next == 2) ? armory::kVoidAppetiteCostR2
                           : armory::kVoidAppetiteCostR3;
    case MasteryId::ShardEye:
      return (next == 1) ? armory::kShardEyeCostR1 : armory::kShardEyeCostR2;
    case MasteryId::Count:
      break;
    }
    return 0;
  }

  int masteryRank(MasteryId id) const {
    const std::size_t idx = static_cast<std::size_t>(id);
    if (id == MasteryId::Count || idx >= mastery_ranks.size()) {
      return 0;
    }
    const int r = static_cast<int>(mastery_ranks[idx]);
    return std::clamp(r, 0, masteryMaxRanks(id));
  }

  int echoFoundationEssenceBonus() const {
    return masteryRank(MasteryId::EchoFoundation) *
           armory::kEchoFoundationStartEssencePerRank;
  }

  int nexusVaultStartHpBonus() const {
    return masteryRank(MasteryId::NexusVault) *
           armory::kNexusVaultStartHpPerRank;
  }

  float rapidGrowthKillThresholdMultiplier() const {
    const int r = masteryRank(MasteryId::RapidGrowth);
    const float red =
        static_cast<float>(r) * armory::kRapidGrowthKillThresholdReductionPerRank;
    return std::clamp(1.0f - red, 0.10f, 1.0f);
  }

  float kineticSwarmMoveSpeedMultiplier() const {
    const int r = masteryRank(MasteryId::KineticSwarm);
    const float add = static_cast<float>(r) * armory::kKineticSwarmMoveSpeedBonusPerRank;
    return std::max(0.0f, 1.0f + add);
  }

  float synthesisMergeCooldownSec() const {
    const int r = masteryRank(MasteryId::SynthesisMastery);
    const float red =
        static_cast<float>(r) * armory::kSynthesisMergeCooldownReductionSecPerRank;
    return std::max(0.0f, merge::kCooldownSec - red);
  }

  float ironResolveHpMultiplier() const {
    const int r = masteryRank(MasteryId::IronResolve);
    const int over = std::max(0, player_level - meta::kPlayerLevelUnlockBrutalMode);
    const float add = static_cast<float>(over) *
                      static_cast<float>(r) *
                      armory::kIronResolveHpBonusPerLevelAbove20PerRank;
    return std::max(0.0f, 1.0f + add);
  }

  float voidAppetiteEssenceDropMultiplier() const {
    const int r = masteryRank(MasteryId::VoidAppetite);
    const float add =
        static_cast<float>(r) * armory::kVoidAppetiteEssenceDropBonusPerRank;
    return std::max(0.0f, 1.0f + add);
  }

  int shardEyeFirstTimeCompleteBonusShards() const {
    return masteryRank(MasteryId::ShardEye) *
           armory::kShardEyeBonusShardsPerRank;
  }

  bool isCharacterUnlocked(CharacterId id) const {
    const std::size_t idx = static_cast<std::size_t>(id);
    if (id == CharacterId::Count || idx >= unlocked_characters.size()) {
      return false;
    }
    return unlocked_characters[idx] != 0;
  }

  void unlockCharacter(CharacterId id) {
    const std::size_t idx = static_cast<std::size_t>(id);
    if (id == CharacterId::Count || idx >= unlocked_characters.size()) {
      return;
    }
    unlocked_characters[idx] = 1;
  }

  void sanitizeCharacterUnlocks() {
    unlockCharacter(CharacterId::Brix);
    unlockCharacter(CharacterId::Flara);
    unlockCharacter(CharacterId::Mossling);
    for (const RosterEntry &re : roster) {
      unlockCharacter(re.character);
    }
  }

  void sanitizeMasteries() {
    for (std::size_t i = 0; i < mastery_ranks.size(); ++i) {
      const auto id = static_cast<MasteryId>(static_cast<std::uint8_t>(i));
      const int max_r = masteryMaxRanks(id);
      mastery_ranks[i] = static_cast<std::uint8_t>(
          std::clamp<int>(mastery_ranks[i], 0, std::max(0, max_r)));
    }
  }

  void sanitizeMetaProgression() {
    lifetime_levels_completed = std::max(0, lifetime_levels_completed);
    lifetime_stars_earned = std::max(0, lifetime_stars_earned);
    lifetime_bosses_killed = std::max(0, lifetime_bosses_killed);
    lifetime_merges = std::max(0, lifetime_merges);

    if (lifetime_levels_completed == 0 && !stars_per_level.empty()) {
      std::int32_t levels = 0;
      std::int32_t stars = 0;
      for (const std::uint8_t s : stars_per_level) {
        if (s > 0) {
          levels += 1;
        }
        stars += static_cast<std::int32_t>(std::clamp<int>(s, 0, 3));
      }
      lifetime_levels_completed = std::max(lifetime_levels_completed, levels);
      lifetime_stars_earned = std::max(lifetime_stars_earned, stars);
    }

    recomputeMetaProgression();
  }

  void recomputeMetaProgression() {
    const std::int64_t xp =
        static_cast<std::int64_t>(lifetime_levels_completed) *
            static_cast<std::int64_t>(meta::kXpPerLevelCompleted) +
        static_cast<std::int64_t>(lifetime_stars_earned) *
            static_cast<std::int64_t>(meta::kXpPerStar) +
        static_cast<std::int64_t>(lifetime_bosses_killed) *
            static_cast<std::int64_t>(meta::kXpPerBossKilled) +
        static_cast<std::int64_t>(lifetime_merges) *
            static_cast<std::int64_t>(meta::kXpPerMerge);
    const std::int64_t clamped =
        std::clamp<std::int64_t>(xp, 0, static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()));
    player_xp = static_cast<std::int32_t>(clamped);
    const int per = std::max(1, meta::kXpPerPlayerLevel);
    player_level = std::max(1, 1 + (player_xp / per));
  }
};

} // namespace tower_swarm
