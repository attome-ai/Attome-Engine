// Content data (data/items.json, npcs.json, maps/overworld.json): loads the
// same files the game ships with and checks them. Also loads them for every
// other test in this executable (static initialiser below), exactly like the
// client / server do at startup.

#include "atm_test.h"

#include "../shared/GameTypes.h"

#include "../../engine/ATMConfig.h"

#include <cmath>
#include <set>
#include <string>

using namespace ao;

namespace {

// Runs before main() (and so before any test): the tests use itemDef() /
// monsterDef() / mapRegions(), which are empty until the data is loaded.
std::string g_dataError;
const bool g_dataLoaded = loadGameData(atm::resolve_path("data"), &g_dataError);

} // namespace

ATM_TEST(data_loads_without_errors) {
  if (!g_dataLoaded)
    std::printf("    data error: %s\n", g_dataError.c_str());
  ATM_REQUIRE(g_dataLoaded);
  ATM_CHECK(gameDataLoaded());
}

ATM_TEST(data_items_are_complete) {
  ATM_REQUIRE(g_dataLoaded);
  std::set<std::string> names;
  for (ItemId i = 1; i < itemCount(); ++i) {
    const ItemDef &d = itemDef(i);
    if (d.name.empty())
      continue; // gaps in the id space are allowed
    ATM_CHECK(names.insert(std::string(d.name)).second); // unique names
    ATM_CHECK(!d.display.empty());
    ATM_CHECK(d.maxStack >= 1);
    ATM_CHECK(int(d.rarity) <= int(Rarity::Legendary));
    if (d.kind == ItemKind::Weapon || d.kind == ItemKind::Tool)
      ATM_CHECK(d.weapon != WeaponType::None && d.piece != nullptr);
    if (d.kind == ItemKind::Armour)
      ATM_CHECK(d.piece != nullptr);
    if (d.kind == ItemKind::Food)
      ATM_CHECK(d.healAmount > 0);
  }
  // Well-known ids used by code (starting kit, ammo) keep their names.
  ATM_CHECK(itemDef(items::WoodenSword).name == "wooden_sword");
  ATM_CHECK(itemDef(items::Arrows).name == "arrows");
  ATM_CHECK(itemDef(items::Coins).name == "coins");
  ATM_CHECK(itemDef(0).name.empty()); // id 0 = empty item
}

ATM_TEST(data_npcs_are_sane) {
  ATM_REQUIRE(g_dataLoaded);
  ATM_REQUIRE(monsterTypeCount() >= 3);
  for (uint8_t m = 0; m < monsterTypeCount(); ++m) {
    const MonsterDef &d = monsterDef(m);
    ATM_CHECK(!d.name.empty() && !d.display.empty());
    ATM_CHECK(d.level >= 1 && d.maxHp > 0);
    ATM_CHECK(d.hit.radius > 0.1f && d.hit.radius < 5.0f);
    ATM_CHECK(d.hit.top >= d.hit.bottom && d.hit.bottom >= 0.0f);
    ATM_CHECK(d.attackRange > 0.0f && d.attackCooldown > 0.0f);
    for (const auto &drop : d.common) {
      ATM_CHECK(drop.item != 0 && !itemDef(drop.item).name.empty());
      ATM_CHECK(drop.chance > 0.0f && drop.chance <= 1.0f && drop.min <= drop.max);
    }
    ATM_CHECK_EQ(findNpc(d.name), int(m));
  }
  ATM_CHECK_EQ(findNpc("no_such_npc"), -1);
  // The golem is the big one: its hit capsule must be taller than the slime's.
  ATM_CHECK(monsterDef(monsters::Golem).hit.top > monsterDef(monsters::Slime).hit.top);
}

ATM_TEST(data_map_regions) {
  ATM_REQUIRE(g_dataLoaded);
  ATM_REQUIRE(!mapRegions().empty());
  ATM_CHECK(!mapName().empty());
  // Rings around the spawn (see overworld.json).
  const MapRegion *spawn = regionAt(0.5, 0.5);
  ATM_REQUIRE(spawn != nullptr);
  ATM_CHECK(spawn->name == "Sunny Meadows");
  ATM_CHECK(regionAt(200.0, 0.0)->name == "Howling Woods");
  ATM_CHECK(regionAt(0.0, -500.0)->name == "Golem Highlands");
  ATM_CHECK(regionAt(5000.0, 5000.0)->name == "The Wilds"); // fallback region
  // Every region can spawn something, and only real NPCs.
  for (const MapRegion &r : mapRegions()) {
    ATM_CHECK(!r.spawns.empty() && r.totalWeight > 0.0f);
    for (float t = 0.0f; t < 1.0f; t += 0.05f) {
      const int npc = pickSpawn(r, t);
      ATM_CHECK(npc >= 0 && npc < int(monsterTypeCount()));
    }
  }
}

ATM_TEST(data_hit_capsule_math) {
  ATM_REQUIRE(g_dataLoaded);
  const MonsterHitShape s = monsterHitShape(monsters::Golem);
  // A point level with the capsule axis, `radius` away sideways: on the edge.
  const double mid = 0.5 * (s.bottom + s.top);
  ATM_CHECK_NEAR(monsterHitDistance2(monsters::Golem, 0, 0, 0, s.radius, mid, 0), double(s.radius) * s.radius, 1e-6);
  // Above the top: distance measured from the top of the axis.
  double axisY = 0.0;
  const double d2 = monsterHitDistance2(monsters::Golem, 0, 0, 0, 0, s.top + 2.0, 0, &axisY);
  ATM_CHECK_NEAR(axisY, double(s.top), 1e-9);
  ATM_CHECK_NEAR(d2, 4.0, 1e-6);
  // Feet at y = 10: everything shifts with the feet.
  ATM_CHECK_NEAR(monsterHitDistance2(monsters::Golem, 5, 10, 5, 5, 10 + mid, 5), 0.0, 1e-9);
}

ATM_TEST(data_weapon_rules) {
  for (WeaponType w : {WeaponType::None, WeaponType::Sword, WeaponType::Bow, WeaponType::Staff, WeaponType::Pickaxe})
    ATM_CHECK(weaponCooldown(w) > 0.1f && weaponCooldown(w) < 5.0f);
  ATM_CHECK(isHoldable(ItemKind::Weapon) && isHoldable(ItemKind::Tool));
  ATM_CHECK(!isHoldable(ItemKind::Armour) && !isHoldable(ItemKind::Food) && !isHoldable(ItemKind::Block));
}
