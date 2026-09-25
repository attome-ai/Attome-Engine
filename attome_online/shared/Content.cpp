// Demo content tables: items, monsters, block drops (data-driven later).
// Item ids are fixed by ao::items (GameTypes.h); piece names match
// atm::model::ModelLibrary::buildDefaults(); monster models match
// ModelLibrary creature names.

#include "GameTypes.h"

#include "../../engine/voxel/BlockRegistry.h"

#include <vector>

namespace ao {

namespace {

using atm::model::EquipSlot;
namespace vb = atm::voxel::blocks;

ItemDef blockItem(std::string_view name, BlockId block) {
  ItemDef d;
  d.name = name;
  d.kind = ItemKind::Block;
  d.placesBlock = block;
  d.skill = Skill::Construction;
  d.maxStack = 100;
  return d;
}

ItemDef resource(std::string_view name, uint16_t maxStack = 100, Skill skill = Skill::Mining) {
  ItemDef d;
  d.name = name;
  d.kind = ItemKind::Resource;
  d.skill = skill;
  d.maxStack = maxStack;
  return d;
}

ItemDef weapon(std::string_view name, WeaponType type, uint16_t damage, Skill skill,
               uint8_t levelReq, const char *piece) {
  ItemDef d;
  d.name = name;
  d.kind = type == WeaponType::Pickaxe ? ItemKind::Tool : ItemKind::Weapon;
  d.weapon = type;
  d.slot = EquipSlot::MainHand;
  d.damage = damage;
  d.skill = skill;
  d.levelReq = levelReq;
  d.piece = piece;
  return d;
}

ItemDef armour(std::string_view name, EquipSlot slot, uint16_t armourValue, uint8_t levelReq,
               const char *piece, Skill skill = Skill::Defence) {
  ItemDef d;
  d.name = name;
  d.kind = ItemKind::Armour;
  d.slot = slot;
  d.armour = armourValue;
  d.levelReq = levelReq;
  d.skill = skill;
  d.piece = piece;
  return d;
}

ItemDef food(std::string_view name, uint16_t heal) {
  ItemDef d;
  d.name = name;
  d.kind = ItemKind::Food;
  d.healAmount = heal;
  d.skill = Skill::Cooking;
  d.maxStack = 50;
  return d;
}

const std::vector<ItemDef> &itemTable() {
  static const std::vector<ItemDef> table = [] {
    std::vector<ItemDef> t(items::Count);
    t[items::Empty] = ItemDef{};
    t[items::Stone] = blockItem("stone", vb::Stone);
    t[items::Dirt] = blockItem("dirt", vb::Dirt);
    t[items::Grass] = blockItem("grass", vb::Grass);
    t[items::Sand] = blockItem("sand", vb::Sand);
    t[items::OakLog] = blockItem("oak_log", vb::OakLog);
    t[items::OakLog].skill = Skill::Woodcutting;
    t[items::OakLeaves] = blockItem("oak_leaves", vb::OakLeaves);
    t[items::Planks] = blockItem("planks", vb::Planks);
    t[items::Brick] = blockItem("brick", vb::Brick);
    t[items::Glass] = blockItem("glass", vb::Glass);
    t[items::Lamp] = blockItem("lamp", vb::Lamp);
    t[items::Snow] = blockItem("snow", vb::Snow);

    t[items::CopperOre] = resource("copper_ore");
    t[items::IronOre] = resource("iron_ore");
    t[items::GoldOre] = resource("gold_ore");
    t[items::CrystalShard] = resource("crystal_shard");
    t[items::RawMeat] = resource("raw_meat", 50, Skill::Cooking);
    t[items::CookedMeat] = food("cooked_meat", 20);
    t[items::WolfPelt] = resource("wolf_pelt", 50, Skill::Crafting);
    t[items::SlimeGel] = resource("slime_gel", 100, Skill::Herblore);
    t[items::Coins] = resource("coins", 60000, Skill::Attack);

    t[items::WoodenSword] = weapon("wooden_sword", WeaponType::Sword, 4, Skill::Attack, 1, "wooden_sword");
    t[items::IronSword] = weapon("iron_sword", WeaponType::Sword, 8, Skill::Attack, 10, "iron_sword");
    t[items::CrystalSword] = weapon("crystal_sword", WeaponType::Sword, 14, Skill::Attack, 30, "crystal_sword");
    t[items::Bow] = weapon("bow", WeaponType::Bow, 6, Skill::Ranged, 1, "bow");
    t[items::Arrows] = resource("arrows", 1000, Skill::Ranged);
    t[items::Staff] = weapon("staff", WeaponType::Staff, 7, Skill::Magic, 1, "staff");
    t[items::Pickaxe] = weapon("pickaxe", WeaponType::Pickaxe, 2, Skill::Mining, 1, "pickaxe");
    t[items::WoodenShield] = armour("wooden_shield", EquipSlot::OffHand, 3, 1, "wooden_shield");

    t[items::LeatherCap] = armour("leather_cap", EquipSlot::Head, 2, 1, "leather_cap");
    t[items::IronHelm] = armour("iron_helm", EquipSlot::Head, 5, 10, "iron_helm");
    t[items::CrystalCrown] = armour("crystal_crown", EquipSlot::Head, 8, 30, "crystal_crown");
    t[items::LeatherTunic] = armour("leather_tunic", EquipSlot::Torso, 4, 1, "leather_tunic");
    t[items::IronChestplate] = armour("iron_chestplate", EquipSlot::Torso, 10, 10, "iron_chestplate");
    t[items::MageRobe] = armour("mage_robe", EquipSlot::Torso, 3, 5, "mage_robe", Skill::Magic);
    t[items::LeatherGloves] = armour("leather_gloves", EquipSlot::Hands, 1, 1, "leather_gloves");
    t[items::IronGauntlets] = armour("iron_gauntlets", EquipSlot::Hands, 3, 10, "iron_gauntlets");
    t[items::LeatherPants] = armour("leather_pants", EquipSlot::Legs, 3, 1, "leather_pants");
    t[items::IronGreaves] = armour("iron_greaves", EquipSlot::Legs, 7, 10, "iron_greaves");
    t[items::Boots] = armour("boots", EquipSlot::Feet, 2, 1, "boots");
    t[items::RedCape] = armour("red_cape", EquipSlot::Back, 1, 1, "red_cape");
    t[items::GliderWings] = armour("glider_wings", EquipSlot::Back, 0, 1, "glider_wings", Skill::Agility);
    return t;
  }();
  return table;
}

// Monster table (GAME_DESIGN §11: per-player common rolls + one rare roll).
using Drop = MonsterDef::Drop;
constexpr Drop kNoDrop{0, 0, 0, 0.0f};

const MonsterDef kMonsters[monsters::Count] = {
    {"slime", 30, 3, 2.5f, 8.0f, 1.6f, 1.2f, 40, "slime",
     {Drop{items::SlimeGel, 1, 3, 1.0f}, Drop{items::Coins, 1, 5, 0.6f},
      Drop{items::CopperOre, 1, 1, 0.15f}, kNoDrop},
     Drop{items::LeatherCap, 1, 1, 0.02f}},
    {"wolf", 80, 8, 6.0f, 14.0f, 1.8f, 1.0f, 120, "wolf",
     {Drop{items::RawMeat, 1, 2, 1.0f}, Drop{items::WolfPelt, 1, 1, 0.5f},
      Drop{items::Coins, 5, 15, 0.7f}, Drop{items::Arrows, 3, 8, 0.2f}},
     Drop{items::RedCape, 1, 1, 0.03f}},
    {"golem", 300, 20, 3.0f, 12.0f, 2.6f, 2.0f, 600, "golem",
     {Drop{items::IronOre, 2, 5, 1.0f}, Drop{items::CrystalShard, 1, 3, 0.6f},
      Drop{items::Coins, 20, 50, 1.0f}, Drop{items::GoldOre, 1, 2, 0.3f}},
     Drop{items::CrystalSword, 1, 1, 0.05f}},
};

} // namespace

const ItemDef &itemDef(ItemId id) {
  const auto &t = itemTable();
  return t[id < t.size() ? id : 0];
}

ItemId findItem(std::string_view name) {
  const auto &t = itemTable();
  for (size_t i = 1; i < t.size(); ++i)
    if (t[i].name == name)
      return ItemId(i);
  return 0;
}

uint16_t itemCount() { return uint16_t(itemTable().size()); }

ItemId blockDropItem(BlockId block) {
  switch (block) {
  case vb::Stone: return items::Stone;
  case vb::Dirt: return items::Dirt;
  case vb::Grass: return items::Dirt;
  case vb::Sand: return items::Sand;
  case vb::OakLog: return items::OakLog;
  case vb::OakLeaves: return items::OakLeaves;
  case vb::CopperOre: return items::CopperOre;
  case vb::IronOre: return items::IronOre;
  case vb::GoldOre: return items::GoldOre;
  case vb::Crystal: return items::CrystalShard;
  case vb::Planks: return items::Planks;
  case vb::Brick: return items::Brick;
  case vb::Glass: return items::Glass;
  case vb::Lamp: return items::Lamp;
  case vb::Snow: return items::Snow;
  default: return 0; // air, water, bedrock, unknown
  }
}

const MonsterDef &monsterDef(uint8_t type) {
  return kMonsters[type < monsters::Count ? type : 0];
}

uint8_t monsterTypeCount() { return monsters::Count; }

} // namespace ao
