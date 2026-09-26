#pragma once

// Loads the game's content definitions from a data directory:
//   items.json, npcs.json, maps/overworld.json
// Call once at startup (client, server, bots) before using itemDef(),
// monsterDef() or mapRegions(). Idempotent: later calls return the first
// result. Fails with a readable error (file, id, field) on bad data.

#include <string>

namespace ao {

bool loadGameData(const std::string &dataDir, std::string *error = nullptr);
bool gameDataLoaded();

} // namespace ao
