#pragma once

// Item definitions, loaded from data/items.json (see the _doc block there).
// Item ids are part of the network protocol and of saved inventories.

#include "../CoreTypes.h"

#include <string_view>

namespace ao {

enum class ItemKind : uint8_t { None, Block, Weapon, Tool, Armour, Resource, Food };
enum class WeaponType : uint8_t { None = 0, Sword = 1, Bow = 2, Staff = 3, Pickaxe = 4 };
// Drop-label / tooltip colour tier (RuneLite-style ground items).
enum class Rarity : uint8_t { Common, Uncommon, Rare, Epic, Legendary };

// Special Equip.equipSlot values (real slots are 0..kEquipSlotCount-1):
// select the held hotbar slot (inventorySlot = 0..8), or eat/use an item.
inline constexpr uint8_t kEquipSelectHotbar = 15;
inline constexpr uint8_t kEquipConsume = 14;

// The item a player attacks / mines with: the selected hotbar slot when it
// holds a weapon or tool, otherwise the equipped main hand.
inline constexpr bool isHoldable(ItemKind k) { return k == ItemKind::Weapon || k == ItemKind::Tool; }

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

// Strings point into storage owned by the item table (valid for the process).
struct ItemDef {
  std::string_view name;          // internal id name ("leather_tunic")
  std::string_view display;       // shown to players ("Leather Tunic")
  std::string_view examine;       // one-line description
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
  uint32_t value = 0;             // coins (drop labels, shops)
  Rarity rarity = Rarity::Common;
};

const ItemDef &itemDef(ItemId id);  // id 0 / unknown = the empty item
ItemId findItem(std::string_view name);
uint16_t itemCount();               // table size (max id + 1)

// Well-known item ids referenced by code (starting kit, ammo, block drops).
// Checked against data/items.json at load: the names must match.
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
} // namespace items

// Item dropped when a block is broken (0 = nothing), from items.json
// "blockDrops" (e.g. grass -> dirt, crystal -> crystal shard).
ItemId blockDropItem(BlockId block);

inline constexpr int kInventorySlots = 28;   // RuneScape inventory size
struct ItemStack { ItemId item = 0; uint16_t count = 0; };

} // namespace ao
