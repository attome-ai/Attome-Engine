#include "shop/RelicSystem.h"

#include "Constants.h"
#include "levels/GameState.h"

#include <algorithm>
#include <array>

namespace tower_swarm {
namespace {

constexpr std::array<RelicDef, RelicSystem::kRelicCount> kDefs = {{
    {RelicId::IronCore, "Iron Core", "All creatures +10% max HP", 0, true},
    {RelicId::Bloodshard, "Bloodshard",
     "+3% damage per tier of the attacking creature", 0, true},
    {RelicId::EssenceMagnet, "Essence Magnet",
     "+15% essence from all drops", relic_unlocks::kEssenceMagnetShardCost, false},
    {RelicId::MergersGift, "Merger's Gift",
     "Merged creatures inherit 40% progress toward next evolution threshold",
     relic_unlocks::kMergersGiftShardCost, false},
    {RelicId::WarpedTime, "Warped Time",
     "Between-wave grace timer +3 seconds", relic_unlocks::kWarpedTimeShardCost,
     false},
    {RelicId::PackInstinct, "Pack Instinct",
     "+8% attack speed per 3 same-type creatures on field",
     relic_unlocks::kPackInstinctShardCost, false},
    {RelicId::EruptionCore, "Eruption Core",
     "Splasher explosions leave burning ground for 3 seconds",
     relic_unlocks::kEruptionCoreShardCost, false},
    {RelicId::ChainStrike, "Chain Strike",
     "Charger kills release a 60px shockwave",
     relic_unlocks::kChainStrikeShardCost, false},
    {RelicId::VoidLens, "Void Lens",
     "Sniper creatures reveal enemy HP bars at 2x range",
     relic_unlocks::kVoidLensShardCost, false},
    {RelicId::LivingWall, "Living Wall",
     "Walls gain +20 HP per completed wave", relic_unlocks::kLivingWallShardCost,
     false},
    {RelicId::ApexHunger, "Apex Hunger",
     "Creature with most kills deals +20% bonus damage",
     relic_unlocks::kApexHungerShardCost, false},
    {RelicId::TwinPulse, "Twin Pulse",
     "Support aura radius +40px", relic_unlocks::kTwinPulseShardCost, false},
    {RelicId::ColdBloom, "Cold Bloom",
     "Trapper slow fields also reduce enemy HP regen",
     relic_unlocks::kColdBloomShardCost, false},
    {RelicId::ResonantGrowth, "Resonant Growth",
     "Crystalis aura boosts creature evolution rate +10%",
     relic_unlocks::kResonantGrowthShardCost, false},
    {RelicId::ChaosSpark, "Chaos Spark",
     "Vex's random ability pool gains +1 option per level above 30",
     relic_unlocks::kChaosSparkShardCost, false},
    {RelicId::EternalEcho, "Eternal Echo",
     "Once per level, the first time your base would reach 0 HP, it stays at 1 HP",
     relic_unlocks::kEternalEchoShardCost, false},
    {RelicId::RecursiveMerge, "Recursive Merge",
     "After a merge, 10% chance to trigger a free second merge on the new creature",
     relic_unlocks::kRecursiveMergeShardCost, false},
    {RelicId::ShardHunger, "Shard Hunger",
     "Earn +1 bonus Shard per 100 kills in a single level",
     relic_unlocks::kShardHungerShardCost, false},
    {RelicId::DeathBloom, "Death Bloom",
     "When a creature would die in combat, it explodes for 150px damage burst",
     relic_unlocks::kDeathBloomShardCost, false},
    {RelicId::TheQuiet, "The Quiet",
     "If no enemies reach the base in a level, earn +3 bonus stars (cosmetic)",
     relic_unlocks::kTheQuietShardCost, false},
}};

constexpr std::size_t clamp_index(RelicId id) {
  const std::size_t idx = static_cast<std::size_t>(id);
  return idx < kDefs.size() ? idx : 0;
}

void ensure_start_unlocked(GameState &io_state) {
  for (const RelicDef &d : kDefs) {
    const std::size_t idx = static_cast<std::size_t>(d.id);
    if (idx >= io_state.relic_unlocked.size()) {
      continue;
    }
    if (d.start_unlocked) {
      io_state.relic_unlocked[idx] = 1;
    }
  }
}

} // namespace

const RelicDef &RelicSystem::def(RelicId id) { return kDefs[clamp_index(id)]; }

int RelicSystem::unlockedSlotCount(int player_level) {
  const int lvl = std::max(1, player_level);
  int slots = 1;
  if (lvl >= relics::kSlot2UnlockPlayerLevel) {
    slots = 2;
  }
  if (lvl >= relics::kSlot3UnlockPlayerLevel) {
    slots = 3;
  }
  return std::clamp(slots, 1, relics::kSlotCount);
}

bool RelicSystem::isSlotUnlocked(int slot_index, int player_level) {
  return slot_index >= 0 && slot_index < unlockedSlotCount(player_level);
}

void RelicSystem::sanitizePersistent(GameState &io_state) {
  ensure_start_unlocked(io_state);

  for (std::size_t i = 0; i < io_state.equipped_relics.size(); ++i) {
    const int slot = static_cast<int>(i);
    if (!isSlotUnlocked(slot, io_state.player_level)) {
      io_state.equipped_relics[i] = RelicId::None;
      continue;
    }
    RelicId id = io_state.equipped_relics[i];
    const std::size_t idx = static_cast<std::size_t>(id);
    if (id == RelicId::None || idx >= io_state.relic_unlocked.size() ||
        io_state.relic_unlocked[idx] == 0) {
      io_state.equipped_relics[i] = RelicId::None;
    }
  }

  // Enforce uniqueness: keep earliest slot.
  for (std::size_t a = 0; a < io_state.equipped_relics.size(); ++a) {
    const RelicId id = io_state.equipped_relics[a];
    if (id == RelicId::None) {
      continue;
    }
    for (std::size_t b = a + 1; b < io_state.equipped_relics.size(); ++b) {
      if (io_state.equipped_relics[b] == id) {
        io_state.equipped_relics[b] = RelicId::None;
      }
    }
  }
}

void RelicSystem::apply_all(GameState &io_state) {
  sanitizePersistent(io_state);

  io_state.any_enemy_reached_base_this_level = false;
  io_state.eternal_echo_used_this_level = false;
  io_state.the_quiet_bonus_stars_cosmetic = 0;
}

} // namespace tower_swarm

