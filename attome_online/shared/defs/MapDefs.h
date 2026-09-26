#pragma once

// Map definition (data/maps/overworld.json): named regions of the procedural
// world with their own spawn tables, level range and world-map tint.

#include "../CoreTypes.h"

#include <string_view>
#include <vector>

namespace ao {

struct MapRegion {
  std::string_view name;     // "Sunny Meadows"
  std::string_view levels;   // "1-5" (display only)
  bool hasShape = false;     // false = matches everywhere (fallback region)
  double centerX = 0.0, centerZ = 0.0;
  double minRadius = 0.0, maxRadius = 0.0;
  uint32_t colour = 0xFF808080u; // bytes R,G,B,A (world map tint)
  struct Spawn { uint8_t npc; float weight; };
  std::vector<Spawn> spawns;
  float totalWeight = 0.0f;
};

// A fixed NPC spawn point (RuneScape-style): keeps `count` of one NPC alive
// within `radius` of (x, z). A killed NPC comes back after `respawnSeconds`;
// NPCs never wander or chase further than `leash` from the spawner.
struct MapSpawner {
  uint8_t npc = 0;
  double x = 0.0, z = 0.0;
  float radius = 6.0f;
  uint8_t count = 1;
  float respawnSeconds = 20.0f;
  float leash = 20.0f;
};

std::string_view mapName();
const std::vector<MapSpawner> &mapSpawners();
const std::vector<MapRegion> &mapRegions();
// The first region containing (x, z); the fallback region if none. Never null
// once data is loaded (nullptr only if the map has no regions at all).
const MapRegion *regionAt(double x, double z);
// Picks an NPC id from a region's spawn table with r01 in [0, 1); -1 if none.
int pickSpawn(const MapRegion &region, float r01);

} // namespace ao
