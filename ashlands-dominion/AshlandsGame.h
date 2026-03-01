#ifndef ASHLANDS_GAME_H
#define ASHLANDS_GAME_H

#include "../engine/ATMEngine.h"
#include "AshlandsStructures.h"
#include <SDL3/SDL.h>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

enum class Faction { Ironborn, Thornkin, Ashwalkers, None };

struct FactionInfo {
  Faction faction;
};

#include "AshlandsMilitary.h"

class TileContainer : public RenderableEntityContainer {
public:
  TileContainer(int typeId, uint8_t defaultLayer, int initialCapacity)
      : RenderableEntityContainer(typeId, defaultLayer, initialCapacity) {}

  void update(float dt) override {}
};

// ============================================================================
// ENGINE 1 - WORLD STATE & MAP GENERATION
// ============================================================================

struct HexCoord {
  int q, r, s; // s = -q - r

  HexCoord(int q = 0, int r = 0, int s = 0) : q(q), r(r), s(s) {}

  bool operator==(const HexCoord &other) const {
    return q == other.q && r == other.r && s == other.s;
  }

  int distance(const HexCoord &other) const {
    return (std::abs(q - other.q) + std::abs(r - other.r) +
            std::abs(s - other.s)) /
           2;
  }

  int distanceToOrigin() const {
    return (std::abs(q) + std::abs(r) + std::abs(s)) / 2;
  }
};

// Hash function for HexCoord to use in unordered_map
struct HexCoordHash {
  std::size_t operator()(const HexCoord &h) const {
    std::size_t h1 = std::hash<int>()(h.q);
    std::size_t h2 = std::hash<int>()(h.r);
    return h1 ^ (h2 << 1);
  }
};

enum class TileType { Wasteland, Settlement, Rift, Ruin, WraithBastion };

struct ResourceLayout {
  int timber, stone, ore, ember;
};

struct HexTile {
  HexCoord coord;
  TileType type;
  ResourceLayout layout;
  std::string owner_id; // UUID of player/bot, or empty

  // Runtime entity handle for rendering
  EntityHandle render_entity = INVALID_ID;
};

// Map configuration
constexpr int WORLD_RADIUS = 80;
constexpr float HEX_SIZE = 16.0f; // Graphical size of hex

// Convert hex to pixel
inline void hex_to_pixel(const HexCoord &h, float &x, float &y) {
  x = HEX_SIZE * std::sqrt(3.0f) * (h.q + h.r / 2.0f);
  y = HEX_SIZE * 3.0f / 2.0f * h.r;
}

// Convert pixel to hex (pointy topped)
inline HexCoord pixel_to_hex(float x, float y) {
  float q = (std::sqrt(3.0f) / 3.0f * x - 1.0f / 3.0f * y) / HEX_SIZE;
  float r = (2.0f / 3.0f * y) / HEX_SIZE;
  float s = -q - r;

  int rx = std::round(q);
  int ry = std::round(r);
  int rz = std::round(s);

  float x_diff = std::abs(rx - q);
  float y_diff = std::abs(ry - r);
  float z_diff = std::abs(rz - s);

  if (x_diff > y_diff && x_diff > z_diff) {
    rx = -ry - rz;
  } else if (y_diff > z_diff) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }
  return HexCoord(rx, ry, rz);
}

class WorldMap {
public:
  std::unordered_map<HexCoord, HexTile, HexCoordHash> tiles;

  void generateMap();
  std::vector<HexCoord> getTilesWithinRadius(const HexCoord &center,
                                             int radius);
  ResourceLayout generateResourceLayout(int distance);
};

class SettlementContainer : public RenderableEntityContainer {
public:
  DynamicArray<std::array<uint8_t, 18>>
      harvest_types; // 0:Timber, 1:Stone, 2:Ore, 3:Ember
  DynamicArray<std::array<uint8_t, 18>> harvest_levels;
  DynamicArray<std::array<uint8_t, 32>> structure_levels;
  DynamicArray<double> timber;
  DynamicArray<double> stone;
  DynamicArray<double> ore;
  DynamicArray<double> ember;

  DynamicArray<Faction> factions;
  DynamicArray<uint32_t> owners;
  DynamicArray<int> populations;
  DynamicArray<int> loyalties;
  DynamicArray<bool> is_strongholds;
  DynamicArray<std::vector<BuildJob>> build_queues;

  DynamicArray<TroopCount> garrisoned_troops;

  struct RecruitJob {
    UnitType type;
    int amount;
    float time_remaining_per_unit;
  };
  DynamicArray<std::vector<RecruitJob>> recruit_queues;

  SettlementContainer(int typeId, uint8_t defaultLayer, int initialCapacity);

  bool checkPrerequisites(uint32_t slot, StructureType type);
  bool queueStructure(EntityHandle id, StructureType type);
  bool queueRecruit(EntityHandle id, UnitType type, int amount);

  void update(float dt) override;
  EntityHandle createSettlement(float x, float y, Faction faction,
                                uint32_t owner, bool is_stronghold,
                                ResourceLayout layout);

protected:
  void swapSlots(uint32_t a, uint32_t b) override;
  void resizeArrays(int newCapacity) override;
};

class UnitContainer : public RenderableEntityContainer {
public:
  DynamicArray<Faction> factions;
  DynamicArray<uint32_t> owners;
  DynamicArray<EntityHandle> origin_settlement;
  DynamicArray<EntityHandle> target_settlement;
  DynamicArray<MovementType> move_types;
  DynamicArray<TroopCount> troops;
  DynamicArray<float> travel_progress;
  DynamicArray<float> total_travel_time;

  UnitContainer(int typeId, uint8_t defaultLayer, int initialCapacity);
  void update(float dt) override;

  EntityHandle dispatchArmy(EntityHandle origin, EntityHandle target,
                            MovementType mType, TroopCount counts,
                            Faction faction, uint32_t owner, float travel_time);

  void resolveBattle(uint32_t attacker_unit_slot,
                     class SettlementContainer *settlements);

protected:
  void swapSlots(uint32_t a, uint32_t b) override;
  void resizeArrays(int newCapacity) override;
};

// ============================================================================
// GAME MANAGER
// ============================================================================

class AshlandsGame {
public:
  Engine *engine;
  WorldMap world;

  SettlementContainer *settlements;
  UnitContainer *units;

  int hex_texture_id;
  int rift_texture_id;
  int ruin_texture_id;
  int bastion_texture_id;
  int settlement_texture_id;

  AshlandsGame(Engine *engine);
  void initialize();
  void update(float dt);
  void renderTiles();
};

#endif // ASHLANDS_GAME_H
