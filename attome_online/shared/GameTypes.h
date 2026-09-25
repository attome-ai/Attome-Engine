#pragma once

// Game-level types shared by server, client and bots (docs/GAME_DESIGN.md).

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
// Items (demo set; data-driven later)
// ---------------------------------------------------------------------------
enum class ItemKind : uint8_t { None, Block, Weapon, Tool, Armour, Resource, Food };
enum class WeaponType : uint8_t { None = 0, Sword = 1, Bow = 2, Staff = 3, Pickaxe = 4 };

// Seconds between attacks per weapon. Shared so the client never swings
// faster than the server accepts (a mismatch = animations with no damage).
inline constexpr float weaponCooldown(WeaponType w) {
  switch (w) {
  case WeaponType::Sword: return 0.5f;
  case WeaponType::Bow: return 0.8f;
  case WeaponType::Staff: return 0.9f;
  case WeaponType::Pickaxe: return 0.7f;
  default: return 0.6f;
  }
}

struct ItemDef {
  std::string_view name;
  ItemKind kind = ItemKind::None;
  WeaponType weapon = WeaponType::None;
  atm::model::EquipSlot slot = atm::model::EquipSlot::MainHand;
  BlockId placesBlock = 0;        // Block items
  uint16_t damage = 0;            // Weapons
  uint16_t armour = 0;            // Armour
  uint8_t levelReq = 1;           // in the item's skill
  Skill skill = Skill::Attack;    // skill used / required
  uint16_t healAmount = 0;        // Food
  const char *piece = nullptr;    // ModelLibrary piece name for equipment
  uint16_t maxStack = 1;
};

const ItemDef &itemDef(ItemId id);
ItemId findItem(std::string_view name);
uint16_t itemCount();

// Item ids of the demo table (Content.cpp). Order is part of the protocol.
namespace items {
inline constexpr ItemId Empty = 0,
    // placeable blocks
    Stone = 1, Dirt = 2, Grass = 3, Sand = 4, OakLog = 5, OakLeaves = 6, Planks = 7,
    Brick = 8, Glass = 9, Lamp = 10, Snow = 11,
    // resources
    CopperOre = 12, IronOre = 13, GoldOre = 14, CrystalShard = 15, RawMeat = 16,
    CookedMeat = 17, WolfPelt = 18, SlimeGel = 19, Coins = 20,
    // weapons and tools
    WoodenSword = 21, IronSword = 22, CrystalSword = 23, Bow = 24, Arrows = 25, Staff = 26,
    Pickaxe = 27, WoodenShield = 28,
    // armour
    LeatherCap = 29, IronHelm = 30, CrystalCrown = 31, LeatherTunic = 32, IronChestplate = 33,
    MageRobe = 34, LeatherGloves = 35, IronGauntlets = 36, LeatherPants = 37, IronGreaves = 38,
    Boots = 39, RedCape = 40, GliderWings = 41;
inline constexpr ItemId Count = 42;
} // namespace items

// Item dropped when a block is broken (0 = nothing; e.g. grass -> dirt,
// copper ore block -> copper ore, crystal -> crystal shard).
ItemId blockDropItem(BlockId block);

inline constexpr int kInventorySlots = 28;   // RuneScape inventory size
struct ItemStack { ItemId item = 0; uint16_t count = 0; };

// ---------------------------------------------------------------------------
// Monsters (demo set)
// ---------------------------------------------------------------------------
struct MonsterDef {
  std::string_view name;
  uint16_t maxHp;
  uint16_t damage;
  float speed;          // blocks/s
  float aggroRange;     // blocks
  float attackRange;
  float attackCooldown; // s
  uint32_t xp;          // combat XP per kill (shared by damage)
  const char *model;    // model library name
  // Loot: common per-player roll + one rare roll per kill (GAME_DESIGN §11)
  struct Drop { ItemId item; uint16_t min, max; float chance; };
  std::array<Drop, 4> common;
  Drop rare;
};
const MonsterDef &monsterDef(uint8_t type);
uint8_t monsterTypeCount();
namespace monsters {
inline constexpr uint8_t Slime = 0, Wolf = 1, Golem = 2, Count = 3;
} // namespace monsters

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
