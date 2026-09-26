#pragma once

// Gathering nodes (data/resources.json): the blocks players can harvest in
// the overworld (trees, rocks). The overworld itself is not editable. A
// harvested node depletes (tree -> stump, rock -> depleted rock) and the
// server regrows it after `respawnSeconds`, RuneScape-style.

#include "ItemDefs.h"

#include <algorithm>
#include <string_view>
#include <vector>

namespace ao {

struct ResourceDef {
  std::string_view name;       // "oak_tree"
  std::string_view display;    // "Oak tree"
  BlockId block = 0;           // harvested block (oak_log, copper_ore)
  BlockId depleted = 0;        // left behind (stump, depleted_rock)
  Skill skill = Skill::Mining;
  uint8_t level = 1;           // skill level required
  WeaponType tool = WeaponType::None; // tool that must be in hand
  ItemId item = 0;             // given per successful harvest
  float xp = 0.0f;             // per harvest
  float depleteChance = 1.0f;  // chance per harvest that the node depletes
  bool fells = false;          // trees: the whole tree (logs + leaves) falls
  float respawnSeconds = 30.0f;
  float gatherSeconds = 1.5f;  // time per harvest at the required level
};

const std::vector<ResourceDef> &resourceDefs();
// The node a block belongs to; nullptr when the block can't be harvested.
const ResourceDef *resourceForBlock(BlockId block);

// Seconds per harvest: 1% faster per level above the requirement, down to
// half the base time. Shared so the client's progress bar matches the server.
inline float gatherTime(const ResourceDef &r, uint32_t level) {
  const float over = float(level > r.level ? level - r.level : 0u);
  return r.gatherSeconds * std::max(0.5f, 1.0f - over * 0.01f);
}

} // namespace ao
