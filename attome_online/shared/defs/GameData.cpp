// Loads data/items.json, data/npcs.json and data/maps/overworld.json into the
// definition tables (DefsInternal.h). Strict: unknown enum names, duplicate
// ids, unknown item / NPC references and renamed well-known ids are errors,
// reported as "<file>: <entry>: <problem>".

#include "GameData.h"

#include "DefsInternal.h"

#include "../../../engine/ATMJson.h"
#include "../../../engine/voxel/BlockRegistry.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace ao {

namespace {

using atm::Json;
using atm::model::EquipSlot;
using namespace defs_detail;

struct Ctx {
  std::string file;
  std::string *error;
  bool fail(const std::string &where, const std::string &what) {
    if (error)
      *error = file + ": " + where + ": " + what;
    return false;
  }
};

std::string lower(std::string_view s) {
  std::string out(s);
  for (char &c : out)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string str(const Json &o, const char *key, std::string fallback = {}) {
  const Json *v = o.find(key);
  return v && v->isString() ? v->asString() : fallback;
}
double num(const Json &o, const char *key, double fallback) {
  const Json *v = o.find(key);
  return v ? v->asNumber(fallback) : fallback;
}
uint32_t colour(const Json &o, const char *key, uint32_t fallback) {
  const Json *v = o.find(key);
  if (!v || !v->isArray() || v->size() < 3)
    return fallback;
  auto ch = [&](size_t i) { return uint32_t(std::clamp(v->asArray()[i].asNumber(0), 0.0, 255.0)); };
  return ch(0) | (ch(1) << 8) | (ch(2) << 16) | 0xFF000000u;
}

bool parseSkill(const std::string &s, Skill &out) {
  for (int i = 0; i < kSkillCount; ++i)
    if (lower(skillName(Skill(i))) == s) {
      out = Skill(i);
      return true;
    }
  return false;
}

bool readFile(const std::string &path, Json &out, std::string *error) {
  std::string err;
  if (!Json::parseFile(path, out, &err)) {
    if (error)
      *error = path + ": " + (err.empty() ? "cannot read file" : err);
    return false;
  }
  return true;
}

// --- items ---------------------------------------------------------------------

const atm::voxel::BlockRegistry &defaultBlocks() {
  static const atm::voxel::BlockRegistry blocks = [] {
    atm::voxel::BlockRegistry r;
    r.registerDefaults();
    return r;
  }();
  return blocks;
}

bool loadItems(const Json &root, Ctx &c) {
  const atm::voxel::BlockRegistry &blocks = defaultBlocks();
  const Json *list = root.find("items");
  if (!list || !list->isArray())
    return c.fail("root", "missing \"items\" array");

  std::vector<ItemDef> table(1); // id 0 = empty
  for (const Json &e : list->asArray()) {
    const int id = int(num(e, "id", -1));
    const std::string name = str(e, "name");
    const std::string where = "item " + std::to_string(id) + " '" + name + "'";
    if (id <= 0 || id > 60000) return c.fail(where, "id must be 1..60000");
    if (name.empty()) return c.fail(where, "missing name");
    if (size_t(id) >= table.size()) table.resize(size_t(id) + 1);
    if (!table[size_t(id)].name.empty()) return c.fail(where, "duplicate id");

    ItemDef d;
    d.name = intern(name);
    d.display = intern(str(e, "display", name));
    d.examine = intern(str(e, "examine"));
    const std::string kind = lower(str(e, "kind"));
    if (kind == "block") d.kind = ItemKind::Block;
    else if (kind == "resource") d.kind = ItemKind::Resource;
    else if (kind == "weapon") d.kind = ItemKind::Weapon;
    else if (kind == "tool") d.kind = ItemKind::Tool;
    else if (kind == "armour" || kind == "armor") d.kind = ItemKind::Armour;
    else if (kind == "food") d.kind = ItemKind::Food;
    else if (kind == "prop") d.kind = ItemKind::Prop;
    else return c.fail(where, "unknown kind '" + kind + "'");

    const std::string skill = lower(str(e, "skill", "attack"));
    if (!parseSkill(skill, d.skill)) return c.fail(where, "unknown skill '" + skill + "'");
    d.levelReq = uint8_t(std::clamp(num(e, "level", 1), 1.0, 255.0));
    d.maxStack = uint16_t(std::clamp(num(e, "stack", 1), 1.0, 65535.0));
    d.value = uint32_t(std::max(0.0, num(e, "value", 0)));
    d.damage = uint16_t(std::clamp(num(e, "damage", 0), 0.0, 65535.0));
    d.armour = uint16_t(std::clamp(num(e, "armour", 0), 0.0, 65535.0));
    d.healAmount = uint16_t(std::clamp(num(e, "heal", 0), 0.0, 65535.0));

    const std::string rarity = lower(str(e, "rarity", "common"));
    if (rarity == "common") d.rarity = Rarity::Common;
    else if (rarity == "uncommon") d.rarity = Rarity::Uncommon;
    else if (rarity == "rare") d.rarity = Rarity::Rare;
    else if (rarity == "epic") d.rarity = Rarity::Epic;
    else if (rarity == "legendary") d.rarity = Rarity::Legendary;
    else return c.fail(where, "unknown rarity '" + rarity + "'");

    if (d.kind == ItemKind::Block) {
      const std::string b = str(e, "block");
      d.placesBlock = blocks.find(b);
      if (d.placesBlock == atm::voxel::kAir) return c.fail(where, "unknown block '" + b + "'");
    }
    if (d.kind == ItemKind::Weapon || d.kind == ItemKind::Tool) {
      const std::string w = lower(str(e, "weapon"));
      if (w == "sword") d.weapon = WeaponType::Sword;
      else if (w == "bow") d.weapon = WeaponType::Bow;
      else if (w == "staff") d.weapon = WeaponType::Staff;
      else if (w == "pickaxe") d.weapon = WeaponType::Pickaxe;
      else if (w == "axe") d.weapon = WeaponType::Axe;
      else return c.fail(where, "unknown weapon type '" + w + "'");
      d.slot = EquipSlot::MainHand;
    }
    if (d.kind == ItemKind::Armour) {
      static const char *kSlots[] = {"head", "torso", "hands", "legs", "feet", "back", "mainhand", "offhand"};
      const std::string s = lower(str(e, "slot"));
      int si = -1;
      for (int i = 0; i < atm::model::kEquipSlotCount; ++i)
        if (s == kSlots[i]) si = i;
      if (si < 0) return c.fail(where, "unknown slot '" + s + "'");
      d.slot = EquipSlot(si);
    }
    if (d.kind == ItemKind::Prop) {
      const std::string pr = str(e, "prop", name);
      if (pr.empty()) return c.fail(where, "missing prop");
      d.prop = intern("prop_" + pr).c_str();
    }
    const std::string piece = str(e, "piece");
    if (!piece.empty()) d.piece = intern(piece).c_str();
    table[size_t(id)] = d;
  }

  // Well-known ids used by code must keep their names.
  static const std::pair<ItemId, const char *> kKnown[] = {
      {items::Stone, "stone"}, {items::Dirt, "dirt"}, {items::Sand, "sand"}, {items::OakLog, "oak_log"},
      {items::CopperOre, "copper_ore"}, {items::IronOre, "iron_ore"}, {items::GoldOre, "gold_ore"},
      {items::CrystalShard, "crystal_shard"}, {items::RawMeat, "raw_meat"}, {items::CookedMeat, "cooked_meat"},
      {items::Coins, "coins"}, {items::WoodenSword, "wooden_sword"}, {items::Bow, "bow"},
      {items::Arrows, "arrows"}, {items::Staff, "staff"}, {items::Pickaxe, "pickaxe"},
      {items::LeatherTunic, "leather_tunic"}, {items::GliderWings, "glider_wings"}, {items::Axe, "axe"}};
  for (const auto &[id, nm] : kKnown)
    if (id >= table.size() || table[id].name != nm)
      return c.fail("item " + std::to_string(id), std::string("must be '") + nm + "' (referenced by code)");

  std::unordered_map<BlockId, ItemId> drops;
  if (const Json *bd = root.find("blockDrops"); bd && bd->isObject()) {
    for (const auto &[blockName, itemName] : bd->asObject()) {
      const BlockId b = blocks.find(blockName);
      if (b == atm::voxel::kAir) return c.fail("blockDrops", "unknown block '" + blockName + "'");
      ItemId it = 0;
      for (size_t i = 1; i < table.size(); ++i)
        if (table[i].name == itemName.asString()) it = ItemId(i);
      if (!it) return c.fail("blockDrops", "unknown item '" + itemName.asString() + "'");
      drops[b] = it;
    }
  }
  itemTable() = std::move(table);
  blockDropTable() = std::move(drops);
  return true;
}

bool readDrop(const Json &e, MonsterDef::Drop &d, Ctx &c, const std::string &where) {
  const std::string item = str(e, "item");
  d.item = findItem(item);
  if (!d.item) return c.fail(where, "unknown item '" + item + "'");
  d.min = uint16_t(std::clamp(num(e, "min", 1), 1.0, 65535.0));
  d.max = uint16_t(std::clamp(num(e, "max", d.min), double(d.min), 65535.0));
  d.chance = float(std::clamp(num(e, "chance", 1.0), 0.0, 1.0));
  return true;
}

// --- NPCs ------------------------------------------------------------------------

bool loadNpcs(const Json &root, Ctx &c) {
  const Json *list = root.find("npcs");
  if (!list || !list->isArray())
    return c.fail("root", "missing \"npcs\" array");
  std::vector<MonsterDef> table;
  std::vector<bool> seen;
  for (const Json &e : list->asArray()) {
    const int id = int(num(e, "id", -1));
    const std::string name = str(e, "name");
    const std::string where = "npc " + std::to_string(id) + " '" + name + "'";
    if (id < 0 || id > 250) return c.fail(where, "id must be 0..250");
    if (name.empty()) return c.fail(where, "missing name");
    if (size_t(id) >= table.size()) {
      table.resize(size_t(id) + 1);
      seen.resize(size_t(id) + 1, false);
    }
    if (seen[size_t(id)]) return c.fail(where, "duplicate id");
    seen[size_t(id)] = true;

    MonsterDef d;
    d.name = intern(name);
    d.display = intern(str(e, "display", name));
    d.level = uint16_t(std::clamp(num(e, "level", 1), 1.0, 9999.0));
    d.maxHp = uint16_t(std::clamp(num(e, "hp", 1), 1.0, 65535.0));
    d.damage = uint16_t(std::clamp(num(e, "damage", 0), 0.0, 65535.0));
    d.speed = float(num(e, "speed", 3.0));
    d.aggroRange = float(num(e, "aggro", 10.0));
    d.attackRange = float(num(e, "attackRange", 1.5));
    d.attackCooldown = float(num(e, "attackCooldown", 1.0));
    d.xp = uint32_t(std::max(0.0, num(e, "xp", 0)));
    d.model = intern(str(e, "model", name)).c_str();
    if (const Json *h = e.find("hit"); h && h->isObject()) {
      d.hit.radius = float(num(*h, "radius", 0.8));
      d.hit.bottom = float(num(*h, "bottom", 0.8));
      d.hit.top = float(std::max(num(*h, "top", 0.8), double(d.hit.bottom)));
    }
    if (const Json *drops = e.find("drops"); drops && drops->isArray())
      for (const Json &de : drops->asArray()) {
        MonsterDef::Drop dr;
        if (!readDrop(de, dr, c, where + " drop")) return false;
        d.common.push_back(dr);
      }
    if (const Json *rare = e.find("rare"); rare && rare->isObject())
      if (!readDrop(*rare, d.rare, c, where + " rare")) return false;
    table[size_t(id)] = std::move(d);
  }
  for (size_t i = 0; i < seen.size(); ++i)
    if (!seen[i]) return c.fail("npc " + std::to_string(i), "ids must be contiguous from 0");
  static const std::pair<uint8_t, const char *> kKnown[] = {
      {monsters::Slime, "slime"}, {monsters::Wolf, "wolf"}, {monsters::Golem, "golem"}};
  for (const auto &[id, nm] : kKnown)
    if (id >= table.size() || table[id].name != nm)
      return c.fail("npc " + std::to_string(id), std::string("must be '") + nm + "' (referenced by code)");
  npcTable() = std::move(table);
  return true;
}

// --- map -------------------------------------------------------------------------

bool loadMap(const Json &root, Ctx &c) {
  const Json *list = root.find("regions");
  if (!list || !list->isArray() || list->size() == 0)
    return c.fail("root", "missing \"regions\" array");
  std::vector<MapRegion> regions;
  for (const Json &e : list->asArray()) {
    MapRegion r;
    const std::string name = str(e, "name", "Unnamed");
    const std::string where = "region '" + name + "'";
    r.name = intern(name);
    r.levels = intern(str(e, "levels"));
    r.colour = colour(e, "colour", 0xFF808080u);
    if (const Json *cen = e.find("center"); cen && cen->isArray() && cen->size() >= 2) {
      r.hasShape = true;
      r.centerX = cen->asArray()[0].asNumber(0);
      r.centerZ = cen->asArray()[1].asNumber(0);
      r.minRadius = num(e, "minRadius", 0);
      r.maxRadius = num(e, "maxRadius", 1e9);
    }
    if (const Json *sp = e.find("spawns"); sp && sp->isArray())
      for (const Json &s : sp->asArray()) {
        const std::string npc = str(s, "npc");
        const int id = findNpc(npc);
        if (id < 0) return c.fail(where, "unknown npc '" + npc + "'");
        const float w = float(std::max(0.0, num(s, "weight", 1)));
        r.spawns.push_back({uint8_t(id), w});
        r.totalWeight += w;
      }
    regions.push_back(std::move(r));
  }
  std::vector<MapSpawner> spawners;
  if (const Json *sp = root.find("spawners"); sp && sp->isArray())
    for (const Json &s : sp->asArray()) {
      MapSpawner m;
      const std::string npc = str(s, "npc");
      const std::string where = "spawner " + std::to_string(spawners.size()) + " '" + npc + "'";
      const int id = findNpc(npc);
      if (id < 0) return c.fail(where, "unknown npc '" + npc + "'");
      const Json *pos = s.find("pos");
      if (!pos || !pos->isArray() || pos->size() < 2) return c.fail(where, "missing \"pos\": [x, z]");
      m.npc = uint8_t(id);
      m.x = pos->asArray()[0].asNumber(0);
      m.z = pos->asArray()[1].asNumber(0);
      m.radius = float(std::clamp(num(s, "radius", 6), 0.0, 64.0));
      m.count = uint8_t(std::clamp(num(s, "count", 1), 1.0, 32.0));
      m.respawnSeconds = float(std::clamp(num(s, "respawn", 20), 1.0, 3600.0));
      m.leash = float(std::clamp(num(s, "leash", m.radius + 14.0), double(m.radius) + 2.0, 128.0));
      spawners.push_back(m);
    }
  spawnerTable() = std::move(spawners);
  regionTable() = std::move(regions);
  mapNameStorage() = str(root, "name", "World");
  return true;
}

// --- gathering nodes ---------------------------------------------------------------

bool loadResources(const Json &root, Ctx &c) {
  const atm::voxel::BlockRegistry &blocks = defaultBlocks();
  const Json *list = root.find("resources");
  if (!list || !list->isArray())
    return c.fail("root", "missing \"resources\" array");
  std::vector<ResourceDef> table;
  for (const Json &e : list->asArray()) {
    ResourceDef r;
    const std::string name = str(e, "name");
    const std::string where = "resource '" + name + "'";
    if (name.empty()) return c.fail(where, "missing name");
    r.name = intern(name);
    r.display = intern(str(e, "display", name));
    const std::string block = str(e, "block"), depleted = str(e, "depleted");
    r.block = blocks.find(block);
    if (r.block == atm::voxel::kAir) return c.fail(where, "unknown block '" + block + "'");
    r.depleted = blocks.find(depleted);
    if (r.depleted == atm::voxel::kAir) return c.fail(where, "unknown depleted block '" + depleted + "'");
    for (const ResourceDef &o : table)
      if (o.block == r.block) return c.fail(where, "block '" + block + "' already used by '" + std::string(o.name) + "'");
    const std::string skill = lower(str(e, "skill", "mining"));
    if (!parseSkill(skill, r.skill)) return c.fail(where, "unknown skill '" + skill + "'");
    r.level = uint8_t(std::clamp(num(e, "level", 1), 1.0, 99.0));
    const std::string tool = lower(str(e, "tool"));
    if (tool == "axe") r.tool = WeaponType::Axe;
    else if (tool == "pickaxe") r.tool = WeaponType::Pickaxe;
    else if (tool == "none" || tool.empty()) r.tool = WeaponType::None;
    else return c.fail(where, "unknown tool '" + tool + "'");
    const std::string item = str(e, "item");
    r.item = findItem(item);
    if (!r.item) return c.fail(where, "unknown item '" + item + "'");
    r.xp = float(std::max(0.0, num(e, "xp", 0)));
    r.depleteChance = float(std::clamp(num(e, "depleteChance", 1.0), 0.0, 1.0));
    r.fells = e.find("fells") && e.find("fells")->asBool(false);
    r.respawnSeconds = float(std::clamp(num(e, "respawn", 30), 1.0, 3600.0));
    r.gatherSeconds = float(std::clamp(num(e, "gather", 1.5), 0.2, 30.0));
    table.push_back(r);
  }
  resourceTable() = std::move(table);
  return true;
}

bool g_loaded = false;
bool g_ok = false;
std::string g_error;

} // namespace

bool loadGameData(const std::string &dataDir, std::string *error) {
  if (g_loaded) {
    if (error && !g_ok) *error = g_error;
    return g_ok;
  }
  g_loaded = true;
  auto run = [&]() -> bool {
    const std::string dir = dataDir.empty() || dataDir.back() == '/' || dataDir.back() == '\\' ? dataDir
                                                                                               : dataDir + "/";
    Json items, npcs, map;
    if (!readFile(dir + "items.json", items, &g_error)) return false;
    Ctx ci{dir + "items.json", &g_error};
    if (!loadItems(items, ci)) return false; // NPCs and maps reference items
    if (!readFile(dir + "npcs.json", npcs, &g_error)) return false;
    Ctx cn{dir + "npcs.json", &g_error};
    if (!loadNpcs(npcs, cn)) return false;   // maps reference NPCs
    if (!readFile(dir + "maps/overworld.json", map, &g_error)) return false;
    Ctx cm{dir + "maps/overworld.json", &g_error};
    if (!loadMap(map, cm)) return false;
    Json res;
    if (!readFile(dir + "resources.json", res, &g_error)) return false;
    Ctx cr{dir + "resources.json", &g_error};
    return loadResources(res, cr);
  };
  g_ok = run();
  if (error && !g_ok) *error = g_error;
  return g_ok;
}

bool gameDataLoaded() { return g_loaded && g_ok; }

} // namespace ao
