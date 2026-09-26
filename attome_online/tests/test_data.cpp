// Content data (data/items.json, npcs.json, maps/overworld.json, resources.json): loads the
// same files the game ships with and checks them. Also loads them for every
// other test in this executable (static initialiser below), exactly like the
// client / server do at startup.

#include "atm_test.h"

#include "../shared/GameTypes.h"
#include "../shared/world/Town.h"

#include "../../engine/ATMConfig.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/model/Character.h"
#include "../../engine/model/VoxelCodec.h"
#include "../../engine/model/IconRender.h"

#include <algorithm>
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
  // Rings around the home town (see overworld.json).
  const MapRegion *spawn = regionAt(0.5, 0.5);
  ATM_REQUIRE(spawn != nullptr);
  ATM_CHECK(spawn->name == "Brightwater");
  ATM_CHECK(spawn->spawns.empty()); // the town is safe
  ATM_CHECK(regionAt(100.0, 0.0)->name == "Sunny Meadows");
  ATM_CHECK(regionAt(200.0, 0.0)->name == "Howling Woods");
  ATM_CHECK(regionAt(0.0, -500.0)->name == "Golem Highlands");
  ATM_CHECK(regionAt(5000.0, 5000.0)->name == "The Wilds"); // fallback region
  // The wilderness spawns something (ambient spawning), and only real NPCs.
  ATM_CHECK(!regionAt(5000.0, 5000.0)->spawns.empty());
  for (const MapRegion &r : mapRegions()) {
    if (r.spawns.empty()) continue;
    ATM_CHECK(r.totalWeight > 0.0f);
    for (float t = 0.0f; t < 1.0f; t += 0.05f) {
      const int npc = pickSpawn(r, t);
      ATM_CHECK(npc >= 0 && npc < int(monsterTypeCount()));
    }
  }
}

ATM_TEST(data_spawners) {
  ATM_REQUIRE(g_dataLoaded);
  ATM_REQUIRE(!mapSpawners().empty());
  for (const MapSpawner &s : mapSpawners()) {
    ATM_CHECK(s.npc < monsterTypeCount());
    ATM_CHECK(s.count >= 1 && s.respawnSeconds >= 1.0f);
    ATM_CHECK(s.leash > s.radius);
    // No spawner (or its wander area) inside the safe home town.
    const MapRegion *r = regionAt(s.x, s.z);
    ATM_CHECK(r != nullptr && !(r->name == "Brightwater"));
    ATM_CHECK(std::max(std::fabs(s.x), std::fabs(s.z)) - s.leash > 64.0);
  }
}

ATM_TEST(data_resources) {
  ATM_REQUIRE(g_dataLoaded);
  ATM_REQUIRE(!resourceDefs().empty());
  for (const ResourceDef &r : resourceDefs()) {
    ATM_CHECK(r.block != 0 && r.depleted != 0 && r.block != r.depleted);
    ATM_CHECK(resourceForBlock(r.block) == &r);
    ATM_CHECK(resourceForBlock(r.depleted) == nullptr); // depleted nodes can't be harvested
    ATM_CHECK(r.item != 0 && !itemDef(r.item).name.empty());
    ATM_CHECK(r.tool == WeaponType::Axe || r.tool == WeaponType::Pickaxe);
    ATM_CHECK(r.respawnSeconds >= 1.0f && r.depleteChance > 0.0f);
    ATM_CHECK(r.level >= 1 && r.level <= 99);
    // Faster with levels, never below half the base time.
    ATM_CHECK(gatherTime(r, 99) <= gatherTime(r, r.level));
    ATM_CHECK(gatherTime(r, 99) >= r.gatherSeconds * 0.5f - 1e-6f);
  }
  // Trees and ores are gatherable; ordinary terrain is not.
  ATM_CHECK(resourceForBlock(atm::voxel::blocks::OakLog) != nullptr);
  ATM_CHECK(resourceForBlock(atm::voxel::blocks::CopperOre) != nullptr);
  ATM_CHECK(resourceForBlock(atm::voxel::blocks::Stone) == nullptr);
  ATM_CHECK(resourceForBlock(atm::voxel::blocks::Grass) == nullptr);
  ATM_CHECK(resourceForBlock(atm::voxel::blocks::Cobblestone) == nullptr);
  ATM_CHECK(itemDef(items::Axe).weapon == WeaponType::Axe);
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
  for (WeaponType w : {WeaponType::None, WeaponType::Sword, WeaponType::Bow, WeaponType::Staff, WeaponType::Pickaxe,
                       WeaponType::Axe})
    ATM_CHECK(weaponCooldown(w) > 0.1f && weaponCooldown(w) < 5.0f);
  ATM_CHECK(isHoldable(ItemKind::Weapon) && isHoldable(ItemKind::Tool));
  ATM_CHECK(!isHoldable(ItemKind::Armour) && !isHoldable(ItemKind::Food) && !isHoldable(ItemKind::Block));
}

ATM_TEST(home_town_is_flat_and_deterministic) {
  namespace vx = atm::voxel;
  const vx::WorldGenerator gen(12345);
  const world::TownLayout &town = world::homeTown(gen);
  ATM_CHECK(town.groundY > 64); // above the sea
  // The spawn spot on the square is open ground; the castle towers above it;
  // natural terrain far outside.
  ATM_CHECK_EQ(world::groundHeight(gen, 0, 10), town.groundY);
  ATM_CHECK(world::groundHeight(gen, 0, -42) > town.groundY + 30);
  ATM_CHECK_EQ(world::groundHeight(gen, 500, 500), gen.surfaceHeight(500, 500));
  ATM_CHECK(world::inTown(town, 0.5, 0.5) && !world::inTown(town, 200.0, 0.0));
  // The chunk under the square: plaza paving, open air above the spawn spot,
  // and the same result every time.
  const vx::ChunkCoord c{0, town.groundY / vx::kChunkSize, 0};
  vx::Chunk a, b;
  gen.generate(c, a);
  gen.generate(c, b);
  world::stampTown(town, gen, c, a);
  world::stampTown(town, gen, c, b);
  const int ly = town.groundY - c.y * vx::kChunkSize;
  const vx::BlockId paving = a.get(5, ly, 10);
  ATM_CHECK_EQ(paving, vx::blocks::Barrier); // the square is fine voxels on an invisible floor
  for (int y = ly + 1; y < std::min(ly + 4, vx::kChunkSize); ++y) ATM_CHECK_EQ(a.get(0, y, 10), vx::kAir);
  for (int z = 0; z < vx::kChunkSize; ++z)
    for (int y = 0; y < vx::kChunkSize; ++y)
      for (int x = 0; x < vx::kChunkSize; ++x)
        if (a.get(x, y, z) != b.get(x, y, z)) {
          ATM_CHECK(false);
          return;
        }
  // Players spawn on the square, above the ground.
  double sx = 0, sy = 0, sz = 0;
  world::townSpawnPoint(town, sx, sy, sz);
  ATM_CHECK(world::inTown(town, sx, sz) && sy > town.groundY);
  // The town uses the new building set and has decoration props.
  ATM_CHECK(world::townProps(town).size() > 50);
}

ATM_TEST(props_have_models) {
  ATM_REQUIRE(g_dataLoaded);
  atm::model::ModelLibrary lib;
  lib.buildDefaults();
  // Every prop item and every prop the town places has a voxel model that
  // is small (16 voxels per block, at most 2 blocks on a side).
  int propItems = 0;
  for (ItemId i = 1; i < itemCount(); ++i) {
    const ItemDef &d = itemDef(i);
    if (d.kind != ItemKind::Prop) continue;
    ++propItems;
    ATM_REQUIRE(d.prop != nullptr);
    const int part = lib.findPart(d.prop);
    ATM_CHECK(part >= 0);
    if (part >= 0) {
      const auto &p = lib.parts()[size_t(part)];
      ATM_CHECK(p.sx <= 32 && p.sy <= 32 && p.sz <= 32);
    }
  }
  ATM_CHECK(propItems >= 10);
  const atm::voxel::WorldGenerator gen(12345);
  for (const world::TownProp &p : world::townProps(world::homeTown(gen)))
    ATM_CHECK(lib.findPart(std::string("prop_") + p.name) >= 0);
}

ATM_TEST(town_structures_are_fine_voxels) {
  const atm::voxel::WorldGenerator gen(12345);
  const world::TownLayout &town = world::homeTown(gen);
  const auto &structures = world::townStructures(town);
  ATM_REQUIRE(structures.size() > 30); // square, fountain, castle, houses, walls, market
  size_t voxels = 0, filled = 0, encoded = 0;
  for (const world::MicroModel &m : structures) {
    ATM_CHECK(m.sx % world::kMicro == 0 && m.sy % world::kMicro == 0 && m.sz % world::kMicro == 0);
    ATM_CHECK(m.filled() > 0);
    voxels += m.v.size();
    filled += m.filled();
    // Every structure survives the compact codec unchanged.
    const auto bytes = atm::model::encodeVoxels(m.sx, m.sy, m.sz, m.v, world::microPalette());
    encoded += bytes.size();
    atm::model::EncodedVoxels back;
    ATM_REQUIRE(atm::model::decodeVoxels(bytes, back));
    ATM_CHECK(back.voxels == m.v);
  }
  std::printf("    town: %zu structures, %.1f M voxels (%.1f M filled), %.0f KB as AVX1 (generated from code: 0 KB on disk)\n",
              structures.size(), voxels / 1e6, filled / 1e6, encoded / 1024.0);
  ATM_CHECK(encoded < voxels / 8); // several times smaller than raw, even with noisy stone textures
  // Buildings collide (Barrier in their walls) and doorways are open: the
  // house at x 30..38, z -20..-11 has its door on the south wall (z = -11).
  namespace vx = atm::voxel;
  const int gy = town.groundY + 2; // the first free block above the plinth
  const vx::ChunkCoord c{1, gy >> 5, -1};
  vx::Chunk ch;
  gen.generate(c, ch);
  world::stampTown(town, gen, c, ch);
  ATM_CHECK_EQ(ch.get(34 - 32, gy & 31, -11 + 32), vx::kAir);          // doorway
  ATM_CHECK_EQ(ch.get(37 - 32, gy & 31, -11 + 32), vx::blocks::Barrier); // wall beside it
}

ATM_TEST(voxel_codec_rejects_bad_data) {
  const std::vector<uint8_t> vox = {0, 0, 1, 1, 1, 2, 0, 0};
  const std::vector<uint32_t> pal = {0, 0xFF0000FFu, 0xFF00FF00u};
  auto bytes = atm::model::encodeVoxels(2, 2, 2, vox, pal);
  atm::model::EncodedVoxels out;
  ATM_REQUIRE(atm::model::decodeVoxels(bytes, out));
  ATM_CHECK(out.voxels == vox && out.palette == pal && out.sx == 2);
  auto truncated = bytes;
  truncated.pop_back();
  ATM_CHECK(!atm::model::decodeVoxels(truncated, out));
  auto badMagic = bytes;
  badMagic[0] = 'X';
  ATM_CHECK(!atm::model::decodeVoxels(badMagic, out));
}

ATM_TEST(item_icons_render_from_models) {
  ATM_REQUIRE(g_dataLoaded);
  atm::model::ModelLibrary lib;
  lib.buildDefaults();
  // Every equipment / prop / material item with a model gets a visible icon.
  int rendered = 0;
  for (ItemId i = 1; i < itemCount(); ++i) {
    const ItemDef &d = itemDef(i);
    int part = -1;
    if (d.kind == ItemKind::Prop && d.prop) part = lib.findPart(d.prop);
    if (part < 0) part = lib.findPart("item_" + std::string(d.name));
    if (part < 0 && d.piece) {
      const auto &pc = lib.piece(lib.findPiece(d.piece));
      part = pc.socketPart;
      for (int16_t p : pc.boneParts)
        if (part < 0 && p >= 0) part = p;
    }
    if (part < 0) continue;
    std::vector<uint32_t> px;
    atm::model::renderVoxelIcon(lib.parts()[size_t(part)], 64, px);
    size_t opaque = 0;
    for (uint32_t c : px) opaque += (c >> 24) > 128 ? 1 : 0;
    if (opaque < 200) std::printf("    icon '%s' (part %s) nearly empty: %zu px\n", std::string(d.name).c_str(),
                                  lib.parts()[size_t(part)].name.c_str(), opaque);
    ATM_CHECK(opaque >= 200);
    ++rendered;
  }
  ATM_CHECK(rendered >= 40);
}
