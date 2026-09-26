// Procedural decoration props (Trove-style decor): small voxel models placed
// in the world at 16 voxels per block, smaller than a block. Parts are named
// "prop_<name>" (items.json "prop"), pivot at the bottom centre, front
// facing -Z. The home town uses them now; homes / clan plots will let players
// place them later.

#include "Character.h"
#include "ModelContentDetail.h"

#include <cmath>
#include <tuple>
#include <string>
#include <utility>

namespace atm::model {

using namespace content_detail;

namespace {

// Pivot at the bottom centre of an sx x sy x sz part.
Painter prop(const char *name, int sx, int sy, int sz) {
  return Painter(std::string("prop_") + name, sx, sy, sz, {sx * 0.5f, 0.0f, sz * 0.5f});
}

// Round-ish column (disc per layer) centred in the part.
void disc(Painter &p, float cx, float cz, float r, int y0, int y1, int idx) {
  for (int y = y0; y < y1; ++y)
    for (int z = 0; z < p.part.sz; ++z)
      for (int x = 0; x < p.part.sx; ++x) {
        const float dx = x + 0.5f - cx, dz = z + 0.5f - cz;
        if (dx * dx + dz * dz <= r * r)
          p.set(x, y, z, idx);
      }
}

void ball(Painter &p, float cx, float cy, float cz, float r, int idx) {
  for (int y = 0; y < p.part.sy; ++y)
    for (int z = 0; z < p.part.sz; ++z)
      for (int x = 0; x < p.part.sx; ++x) {
        const float dx = x + 0.5f - cx, dy = y + 0.5f - cy, dz = z + 0.5f - cz;
        if (dx * dx + dy * dy + dz * dz <= r * r)
          p.set(x, y, z, idx);
      }
}

const uint32_t kWood = rgba(170, 118, 66), kWoodD = rgba(118, 78, 44), kWoodL = rgba(214, 170, 110),
               kIron = rgba(70, 72, 84), kIronL = rgba(120, 124, 138), kGold = rgba(250, 200, 70),
               kFlame = rgba(255, 200, 80), kFlameHot = rgba(255, 250, 200), kLeaf = rgba(92, 196, 72),
               kLeafD = rgba(64, 150, 56), kPot = rgba(196, 104, 70), kPotD = rgba(160, 80, 54);

void add(ModelLibrary &lib, Painter &p) { lib.addPart(std::move(p.part)); }

} // namespace

void buildProps(ModelLibrary &lib) {
  { // Torch: stick with a glowing flame (about half a block tall).
    Painter p = prop("torch", 4, 11, 4);
    const int stick = p.c(kWoodD), band = p.c(kIron), flame = p.g(kFlame), hot = p.g(kFlameHot);
    p.box(1, 0, 1, 3, 7, 3, stick);
    p.box(1, 6, 1, 3, 7, 3, band);
    p.box(0, 7, 0, 4, 10, 4, flame);
    p.box(1, 8, 1, 3, 11, 3, hot);
    add(lib, p);
  }
  { // Lantern: iron frame, warm glowing core, gold cap and ring.
    Painter p = prop("lantern", 8, 12, 8);
    const int iron = p.c(kIron), gold = p.c(kGold), glow = p.g(kFlame), hot = p.g(kFlameHot);
    p.box(0, 0, 0, 8, 1, 8, iron);                            // base
    p.box(1, 1, 1, 7, 8, 7, glow);                            // glass
    p.box(3, 3, 3, 5, 6, 5, hot);
    for (int x : {0, 7})
      for (int z : {0, 7})
        p.box(x, 1, z, x + 1, 8, z + 1, iron);               // corner posts
    p.box(0, 8, 0, 8, 9, 8, iron);
    p.box(1, 9, 1, 7, 10, 7, gold);                           // cap
    p.box(3, 10, 3, 5, 12, 5, iron);                          // ring
    add(lib, p);
  }
  { // Street lamp: tall iron post with a lantern head (two blocks tall).
    Painter p = prop("street_lamp", 10, 32, 10);
    const int iron = p.c(kIron), ironL = p.c(kIronL), gold = p.c(kGold), glow = p.g(kFlame),
              hot = p.g(kFlameHot);
    p.box(2, 0, 2, 8, 2, 8, iron);                            // foot
    p.box(3, 2, 3, 7, 3, 7, ironL);
    p.box(4, 3, 4, 6, 22, 6, iron);                           // post
    p.box(3, 12, 3, 7, 13, 7, gold);                          // collar
    p.box(1, 22, 1, 9, 23, 9, iron);                          // head
    p.box(2, 23, 2, 8, 29, 8, glow);
    p.box(4, 24, 4, 6, 28, 6, hot);
    for (int x : {1, 8})
      for (int z : {1, 8})
        p.box(x, 23, z, x + 1, 29, z + 1, iron);
    p.box(1, 29, 1, 9, 30, 9, iron);
    p.box(3, 30, 3, 7, 31, 7, gold);
    p.box(4, 31, 4, 6, 32, 6, gold);
    add(lib, p);
  }
  { // Barrel: bulging wooden staves with two iron hoops.
    Painter p = prop("barrel", 12, 14, 12);
    const int wood = p.c(kWood), dark = p.c(kWoodD), hoop = p.c(kIron), top = p.c(kWoodL);
    for (int y = 0; y < 14; ++y) {
      const float bulge = 4.8f + 1.2f * std::sin(float(y) / 13.0f * 3.14159f);
      disc(p, 6.0f, 6.0f, bulge, y, y + 1, (y % 2) ? wood : dark);
    }
    p.speckle(0, 0, 0, 12, 14, 12, dark, 0.25f, 7u);
    for (int y : {2, 11}) disc(p, 6.0f, 6.0f, 6.0f, y, y + 1, hoop);
    disc(p, 6.0f, 6.0f, 4.6f, 13, 14, top);
    add(lib, p);
  }
  { // Small crate: planks with a dark frame.
    Painter p = prop("small_crate", 10, 10, 10);
    const int wood = p.c(kWoodL), frame = p.c(kWoodD);
    p.box(0, 0, 0, 10, 10, 10, wood);
    p.speckle(0, 0, 0, 10, 10, 10, p.c(kWood), 0.3f, 3u);
    for (int a : {0, 9}) {
      p.box(a, 0, 0, a + 1, 10, 1, frame), p.box(a, 0, 9, a + 1, 10, 10, frame);
      p.box(0, a, 0, 10, a + 1, 1, frame), p.box(0, a, 9, 10, a + 1, 10, frame);
      p.box(0, a, 0, 1, a + 1, 10, frame), p.box(9, a, 0, 10, a + 1, 10, frame);
    }
    add(lib, p);
  }
  { // Bench: long seat on two legs (spans ~2 blocks).
    Painter p = prop("bench", 28, 9, 8);
    const int seat = p.c(kWood), dark = p.c(kWoodD), iron = p.c(kIron);
    p.box(0, 5, 0, 28, 7, 8, seat);
    p.speckle(0, 5, 0, 28, 7, 8, dark, 0.2f, 11u);
    for (int x : {2, 23}) p.box(x, 0, 1, x + 3, 5, 7, iron);
    p.box(0, 7, 6, 28, 9, 8, dark);                           // low back rail
    add(lib, p);
  }
  { // Flower pot: clay pot with a burst of flowers.
    Painter p = prop("flower_pot", 8, 12, 8);
    const int pot = p.c(kPot), potD = p.c(kPotD), leaf = p.c(kLeaf), red = p.c(rgba(236, 70, 96)),
              yel = p.c(rgba(252, 214, 72)), soil = p.c(rgba(90, 60, 40));
    p.box(1, 0, 1, 7, 5, 7, pot);
    p.box(0, 4, 0, 8, 6, 8, potD);
    p.box(1, 5, 1, 7, 6, 7, soil);
    ball(p, 4.0f, 8.0f, 4.0f, 3.3f, leaf);
    for (auto [x, y, z, c] : {std::tuple{2, 9, 2, red}, {5, 10, 3, yel}, {3, 11, 5, red}, {6, 8, 6, yel}, {1, 8, 5, red}})
      p.set(x, y, z, c);
    add(lib, p);
  }
  { // Fruit basket: wicker basket of apples and pears (market goods).
    Painter p = prop("fruit_basket", 12, 8, 10);
    const int wicker = p.c(rgba(196, 150, 86)), wickD = p.c(rgba(150, 108, 60)), apple = p.c(rgba(220, 50, 50)),
              pear = p.c(rgba(170, 210, 70)), orange = p.c(rgba(250, 150, 40));
    p.box(0, 0, 0, 12, 5, 10, wicker);
    p.speckle(0, 0, 0, 12, 5, 10, wickD, 0.4f, 5u);
    p.box(1, 1, 1, 11, 5, 9, 0);
    for (int i = 0; i < 12; ++i) {
      const int x = 1 + (i * 5) % 9, z = 1 + (i * 3) % 7, c = i % 3 == 0 ? apple : i % 3 == 1 ? pear : orange;
      p.box(x, 4, z, x + 2, 6 + (i % 2), z + 2, c);
    }
    add(lib, p);
  }
  { // Sign post: post with a hanging board.
    Painter p = prop("sign_post", 14, 24, 4);
    const int post = p.c(kWoodD), board = p.c(kWoodL), trim = p.c(kGold), iron = p.c(kIron);
    p.box(1, 0, 1, 3, 24, 3, post);
    p.box(1, 22, 1, 14, 23, 3, post);                         // arm
    p.box(12, 20, 1, 13, 22, 3, iron);
    p.box(4, 12, 1, 14, 20, 3, board);
    p.box(4, 12, 1, 14, 13, 3, trim), p.box(4, 19, 1, 14, 20, 3, trim);
    add(lib, p);
  }
  { // Table: plank top on four legs.
    Painter p = prop("table", 16, 12, 16);
    const int top = p.c(kWoodL), leg = p.c(kWoodD);
    p.box(0, 10, 0, 16, 12, 16, top);
    p.speckle(0, 10, 0, 16, 12, 16, p.c(kWood), 0.25f, 9u);
    for (int x : {1, 13})
      for (int z : {1, 13}) p.box(x, 0, z, x + 2, 10, z + 2, leg);
    add(lib, p);
  }
  { // Chair.
    Painter p = prop("chair", 10, 16, 10);
    const int wood = p.c(kWood), dark = p.c(kWoodD);
    p.box(0, 6, 0, 10, 8, 10, wood);
    for (int x : {0, 8})
      for (int z : {0, 8}) p.box(x, 0, z, x + 2, 6, z + 2, dark);
    p.box(0, 8, 8, 10, 16, 10, dark);                         // back
    p.box(2, 10, 8, 8, 14, 10, wood);
    add(lib, p);
  }
  { // Potted bush: round clipped bush in a stone planter.
    Painter p = prop("potted_bush", 12, 18, 12);
    const int stone = p.c(rgba(214, 208, 194)), stoneD = p.c(rgba(176, 170, 160)), leaf = p.c(kLeaf),
              leafD = p.c(kLeafD);
    p.box(1, 0, 1, 11, 5, 11, stone);
    p.box(0, 4, 0, 12, 6, 12, stoneD);
    ball(p, 6.0f, 11.0f, 6.0f, 6.0f, leaf);
    p.speckle(0, 6, 0, 12, 18, 12, leafD, 0.3f, 13u);
    add(lib, p);
  }
  { // Fence segment: pickets on two rails, one block wide.
    Painter p = prop("fence", 16, 14, 4);
    const int wood = p.c(kWoodL), dark = p.c(kWoodD);
    for (int x = 0; x < 16; x += 3) p.box(x, 0, 1, x + 2, 13 + (x % 2), 3, wood);
    p.box(0, 3, 0, 16, 5, 1, dark), p.box(0, 9, 0, 16, 11, 1, dark);
    add(lib, p);
  }
  { // Anvil.
    Painter p = prop("anvil", 14, 10, 8);
    const int iron = p.c(kIron), ironL = p.c(kIronL), wood = p.c(kWoodD);
    p.box(3, 0, 1, 11, 3, 7, wood);                           // stump base
    p.box(5, 3, 2, 9, 6, 6, iron);
    p.box(1, 6, 1, 13, 9, 7, iron);
    p.box(0, 7, 2, 2, 9, 6, iron);                            // horn
    p.box(1, 9, 1, 13, 10, 7, ironL);
    add(lib, p);
  }
  { // Candle cluster.
    Painter p = prop("candles", 8, 8, 8);
    const int wax = p.c(rgba(246, 238, 220)), dish = p.c(kGold), flame = p.g(kFlame);
    p.box(0, 0, 0, 8, 1, 8, dish);
    p.box(1, 1, 1, 3, 6, 3, wax), p.set(1, 6, 1, flame), p.set(2, 6, 2, flame);
    p.box(4, 1, 2, 6, 4, 4, wax), p.set(5, 4, 3, flame);
    p.box(2, 1, 5, 4, 5, 7, wax), p.set(3, 5, 6, flame);
    add(lib, p);
  }
  { // Garden bush: round shrub with a few flowers (bigger than a block wide).
    Painter p = prop("bush", 20, 14, 20);
    const int leaf = p.c(kLeaf), leafD = p.c(kLeafD), flower = p.c(rgba(250, 240, 250)),
              pink = p.c(rgba(240, 120, 180));
    ball(p, 10.0f, 5.0f, 10.0f, 8.5f, leaf);
    ball(p, 6.0f, 8.0f, 12.0f, 5.0f, leaf);
    p.speckle(0, 0, 0, 20, 14, 20, leafD, 0.3f, 17u);
    p.speckle(0, 6, 0, 20, 14, 20, flower, 0.04f, 19u);
    p.speckle(0, 6, 0, 20, 14, 20, pink, 0.04f, 23u);
    add(lib, p);
  }
  { // Hanging shop sign with a gold coin (market stalls).
    Painter p = prop("coin_sign", 12, 12, 2);
    const int board = p.c(kWoodD), gold = p.c(kGold), goldD = p.c(rgba(200, 150, 40)), iron = p.c(kIron);
    p.box(5, 10, 0, 7, 12, 2, iron);
    p.box(0, 0, 0, 12, 10, 2, board);
    for (int y = 1; y < 9; ++y)
      for (int x = 1; x < 11; ++x) {
        const float dx = x + 0.5f - 6.0f, dy = y + 0.5f - 5.0f;
        if (dx * dx + dy * dy <= 12.0f) p.box(x, y, 0, x + 1, y + 1, 1, (dx * dx + dy * dy <= 4.0f) ? goldD : gold);
      }
    add(lib, p);
  }
}

// Models for items that aren't equipment or props (ores, food, materials):
// parts "item_<name>". Used for inventory icons and dropped items.
void buildItemModels(ModelLibrary &lib) {
  auto item = [](const char *name, int sx, int sy, int sz) {
    return Painter(std::string("item_") + name, sx, sy, sz, {sx * 0.5f, 0.0f, sz * 0.5f});
  };
  auto ore = [&](const char *name, uint32_t fleck, uint32_t fleckL, bool glowFleck) {
    Painter p = item(name, 10, 8, 9);
    const int rock = p.c(rgba(128, 130, 140)), rockD = p.c(rgba(96, 98, 108)), rockL = p.c(rgba(160, 162, 172));
    const int f = glowFleck ? p.g(fleck) : p.c(fleck), fl = glowFleck ? p.g(fleckL) : p.c(fleckL);
    for (int y = 0; y < 8; ++y)
      for (int z = 0; z < 9; ++z)
        for (int x = 0; x < 10; ++x) {
          const float dx = x + 0.5f - 5.0f, dy = (y + 0.5f - 3.6f) * 1.2f, dz = z + 0.5f - 4.5f;
          if (dx * dx + dy * dy + dz * dz <= 4.6f * 4.6f) p.set(x, y, z, rock);
        }
    p.speckle(0, 0, 0, 10, 8, 9, rockD, 0.3f, 3u);
    p.speckle(0, 4, 0, 10, 8, 9, rockL, 0.2f, 5u);
    p.speckle(0, 0, 0, 10, 8, 9, f, 0.22f, 7u);
    p.speckle(0, 3, 0, 10, 8, 9, fl, 0.08f, 9u);
    add(lib, p);
  };
  ore("copper_ore", rgba(222, 124, 64), rgba(255, 176, 110), false);
  ore("iron_ore", rgba(196, 150, 128), rgba(236, 206, 186), false);
  ore("gold_ore", rgba(252, 204, 60), rgba(255, 240, 150), false);
  { // Crystal shard: faceted glowing prism.
    Painter p = item("crystal_shard", 7, 14, 7);
    const int c = p.g(rgba(150, 110, 255)), cl = p.g(rgba(220, 200, 255));
    for (int y = 0; y < 14; ++y) {
      const float r = y < 4 ? 1.5f + y * 0.6f : 3.6f * (1.0f - float(y - 4) / 10.0f) + 0.4f;
      for (int z = 0; z < 7; ++z)
        for (int x = 0; x < 7; ++x)
          if (std::abs(x + 0.5f - 3.5f) + std::abs(z + 0.5f - 3.5f) <= r) p.set(x, y, z, (x + y) % 4 == 0 ? cl : c);
    }
    add(lib, p);
  }
  auto meat = [&](const char *name, uint32_t flesh, uint32_t fleshD, uint32_t fat) {
    Painter p = item(name, 12, 6, 9);
    const int m = p.c(flesh), md = p.c(fleshD), f = p.c(fat), bone = p.c(rgba(240, 232, 214));
    for (int y = 0; y < 6; ++y)
      for (int z = 0; z < 9; ++z)
        for (int x = 0; x < 9; ++x) {
          const float dx = (x + 0.5f - 4.5f) / 4.5f, dy = (y + 0.5f - 2.8f) / 3.0f, dz = (z + 0.5f - 4.5f) / 4.2f;
          if (dx * dx + dy * dy + dz * dz <= 1.0f) p.set(x, y, z, m);
        }
    p.speckle(0, 0, 0, 9, 6, 9, md, 0.3f, 13u);
    p.speckle(0, 3, 0, 9, 6, 9, f, 0.15f, 15u);
    p.box(8, 2, 4, 12, 4, 5, bone);
    p.box(10, 1, 3, 12, 5, 6, bone);
    add(lib, p);
  };
  meat("raw_meat", rgba(214, 70, 70), rgba(170, 44, 50), rgba(250, 210, 200));
  meat("cooked_meat", rgba(150, 82, 44), rgba(110, 56, 30), rgba(200, 130, 70));
  { // Wolf pelt: folded grey fur.
    Painter p = item("wolf_pelt", 14, 4, 11);
    const int fur = p.c(rgba(140, 142, 150)), furD = p.c(rgba(100, 102, 112)), furL = p.c(rgba(196, 198, 204));
    p.box(0, 0, 1, 14, 2, 10, fur);
    p.box(2, 2, 2, 12, 4, 9, fur);
    p.box(0, 0, 0, 3, 1, 1, furD), p.box(11, 0, 10, 14, 1, 11, furD);
    p.speckle(0, 0, 0, 14, 4, 11, furD, 0.3f, 17u);
    p.speckle(0, 2, 0, 14, 4, 11, furL, 0.2f, 19u);
    add(lib, p);
  }
  { // Slime gel: translucent-looking green blob with highlights.
    Painter p = item("slime_gel", 9, 7, 9);
    const int g = p.c(rgba(110, 220, 90)), gd = p.c(rgba(70, 170, 60)), gl = p.c(rgba(200, 255, 170));
    for (int y = 0; y < 7; ++y)
      for (int z = 0; z < 9; ++z)
        for (int x = 0; x < 9; ++x) {
          const float dx = x + 0.5f - 4.5f, dy = (y + 0.5f) * 1.3f - 2.0f, dz = z + 0.5f - 4.5f;
          if (dx * dx + dy * dy + dz * dz <= 18.0f) p.set(x, y, z, y < 2 ? gd : g);
        }
    p.set(3, 5, 3, gl), p.set(2, 4, 3, gl), p.set(3, 4, 2, gl);
    add(lib, p);
  }
  { // Coins: a small stack and a few loose coins.
    Painter p = item("coins", 12, 7, 10);
    const int gold = p.c(rgba(252, 204, 60)), goldD = p.c(rgba(200, 150, 40)), goldL = p.c(rgba(255, 240, 150));
    for (int y = 0; y < 6; ++y) disc(p, 5.0f, 5.0f, 3.2f, y, y + 1, y % 2 ? goldD : gold);
    disc(p, 5.0f, 5.0f, 2.0f, 6, 7, goldL);
    disc(p, 9.5f, 2.5f, 2.2f, 0, 1, gold);
    disc(p, 9.0f, 7.5f, 2.2f, 0, 1, goldD);
    add(lib, p);
  }
  { // Arrows: a bundle of three with fletching.
    Painter p = item("arrows", 6, 5, 18);
    const int shaft = p.c(rgba(176, 130, 80)), tip = p.c(rgba(190, 196, 210)), fl = p.c(rgba(240, 240, 240)),
              flr = p.c(rgba(220, 60, 56));
    for (int k = 0; k < 3; ++k) {
      const int x = 1 + k * 2 - (k == 2 ? 3 : 0), y = k == 2 ? 3 : 1;
      p.box(x, y, 2, x + 1, y + 1, 16, shaft);
      p.box(x, y, 0, x + 1, y + 1, 2, tip);
      p.box(x - 1, y, 14, x + 2, y + 1, 18, k == 1 ? flr : fl);
    }
    add(lib, p);
  }
  { // Oak log: bark with a ringed cut end.
    Painter p = item("oak_log", 8, 8, 14);
    const int bark = p.c(rgba(116, 82, 50)), barkD = p.c(rgba(86, 60, 36)), wood = p.c(rgba(214, 176, 116)),
              ring = p.c(rgba(176, 136, 84));
    for (int z = 0; z < 14; ++z)
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
          const float dx = x + 0.5f - 4.0f, dy = y + 0.5f - 4.0f, r2 = dx * dx + dy * dy;
          if (r2 > 16.0f) continue;
          const bool end = z == 0 || z == 13;
          p.set(x, y, z, end ? (int(std::sqrt(r2)) % 2 ? ring : wood) : bark);
        }
    p.speckle(0, 0, 1, 8, 8, 13, barkD, 0.3f, 21u);
    add(lib, p);
  }
}

} // namespace atm::model
