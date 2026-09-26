#pragma once

// Private to shared/defs: the mutable tables behind the read-only accessors
// (itemDef, monsterDef, mapRegions, ...), filled by GameData.cpp.

#include "ItemDefs.h"
#include "MapDefs.h"
#include "NpcDefs.h"
#include "ResourceDefs.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace ao::defs_detail {

// Stable storage for definition strings (string_view / const char* point here).
const std::string &intern(std::string s);

std::vector<ItemDef> &itemTable();
std::unordered_map<BlockId, ItemId> &blockDropTable();
std::vector<MonsterDef> &npcTable();
std::vector<MapRegion> &regionTable();
std::vector<MapSpawner> &spawnerTable();
std::vector<ResourceDef> &resourceTable();
std::string &mapNameStorage();

} // namespace ao::defs_detail
