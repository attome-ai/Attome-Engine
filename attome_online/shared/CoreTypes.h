#pragma once

// Core game types shared by server, client and bots (docs/GAME_DESIGN.md):
// ids, simulation rate, entity kinds, skills and input buttons. Content
// definitions (items, NPCs, maps) live in shared/defs and are loaded from
// data/*.json; include shared/GameTypes.h to get everything.

#include "../../engine/model/Character.h"
#include "../../engine/voxel/VoxelTypes.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <string_view>

namespace ao {

using atm::voxel::BlockId;
using atm::voxel::BlockPos;

using EntityId = uint32_t;               // 0 = none; server-assigned, not reused within a session
inline constexpr EntityId kNoEntity = 0;
using ItemId = uint16_t;                 // 0 = empty
using Tick = uint32_t;

inline constexpr int kSimHz = 30;        // GAME_DESIGN §25 (tunable at runtime)
inline constexpr int kSnapshotHz = 20;
inline constexpr float kSimDt = 1.0f / kSimHz;

enum class EntityKind : uint8_t { Player = 0, Monster = 1, DroppedItem = 2, Projectile = 3 };

// ---------------------------------------------------------------------------
// Skills (GAME_DESIGN §9): 21 skills, no cap, RS curve to 99 then linear.
// ---------------------------------------------------------------------------
enum class Skill : uint8_t {
  Attack, Strength, Defence, Ranged, Magic, Hitpoints, Prayer,
  Mining, Woodcutting, Fishing, Hunter, Farming,
  Smithing, Crafting, Fletching, Cooking, Herblore,
  Construction, Agility, Slayer, Taming,
  Count
};
inline constexpr int kSkillCount = int(Skill::Count);
std::string_view skillName(Skill s);

uint64_t xpForLevel(uint32_t level);  // level 1 = 0 XP; 99 = 13,034,431; +1,228,825 per level after
uint32_t levelForXp(uint64_t xp);     // no cap
uint32_t combatLevel(const std::array<uint32_t, kSkillCount> &levels); // levels capped at 99 inside

// ---------------------------------------------------------------------------
// Input buttons (bitmask sent every tick)
// ---------------------------------------------------------------------------
namespace button {
inline constexpr uint16_t Jump = 1u << 0;
inline constexpr uint16_t Dash = 1u << 1;
inline constexpr uint16_t Primary = 1u << 2;    // attack / break block
inline constexpr uint16_t Secondary = 1u << 3;  // place block / ability
inline constexpr uint16_t Sprint = 1u << 4;
inline constexpr uint16_t Glide = 1u << 5;      // hold jump in air (derived client side)
} // namespace button

} // namespace ao
