#pragma once

#include "levels/GameState.h"

#include <limits>
#include <cstdint>
#include <string>
#include <string_view>

namespace tower_swarm {

struct PersistentSnapshot final {
  std::int32_t max_level_reached{1};
  std::vector<std::uint8_t> stars_per_level{};
  std::int32_t essence{0};
  std::int32_t base_hp{100};
  std::int32_t next_level_base_hp_target{std::numeric_limits<std::int32_t>::min()};
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
};

class SaveState final {
public:
  static constexpr const char *kStorageKey = "tower_swarm_save_v1";

  static bool load(GameState &io_state);
  static bool save(const GameState &state);

  static PersistentSnapshot snapshotPersistent(const GameState &state);
  static void restorePersistent(GameState &io_state,
                                const PersistentSnapshot &snapshot);

  static std::string toJson(const GameState &state);
  static bool fromJson(std::string_view json, GameState &io_state);
};

} // namespace tower_swarm
