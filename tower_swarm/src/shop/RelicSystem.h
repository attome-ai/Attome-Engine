#pragma once

#include "shop/RelicId.h"

#include <array>
#include <cstdint>

namespace tower_swarm {

struct GameState;

struct RelicDef final {
  RelicId id{RelicId::None};
  const char *name{""};
  const char *effect{""};
  std::int32_t shard_cost{0};
  bool start_unlocked{false};
};

class RelicSystem final {
public:
  static constexpr std::size_t kRelicCount =
      static_cast<std::size_t>(RelicId::Count);

  static const RelicDef &def(RelicId id);

  static int unlockedSlotCount(int player_level);
  static bool isSlotUnlocked(int slot_index, int player_level);

  static void sanitizePersistent(GameState &io_state);
  static void apply_all(GameState &io_state);
};

} // namespace tower_swarm

