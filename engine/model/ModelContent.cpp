// Procedural demo content for ModelLibrary: Trove-style chunky voxel body
// parts (2 body shapes, 4 skin tones as palette variants), hair, equipment
// pieces and monster models. Everything is generated in code so the demo
// needs no art files; MagicaVoxel imports can replace parts later.
//
// Part space: voxel (0,0,0) is the part's min corner; `pivot` is the joint
// (in voxels) that sits on the bone. Voxel z = 0 is the FRONT (the model
// faces -Z) and x grows toward the character's right (+X).

#include "Character.h"
#include "ModelContentDetail.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace atm::model {

using namespace content_detail;

namespace {

// ---------------------------------------------------------------------------
// Colours
// ---------------------------------------------------------------------------

// Body palette layout (shared by every body part so skin variants line up).
enum BodyCol : uint8_t {
  kSkin = 1, kSkinShade, kEyeWhite, kEyeDark, kBlush, kShirt, kShirtDark,
  kPants, kPantsDark, kShoe, kSole, kBelt, kBuckle, kMouth, kBodyColCount
};

std::vector<uint32_t> bodyPalette(uint32_t skin, uint32_t skinShade) {
  std::vector<uint32_t> p(kBodyColCount, 0);
  p[kSkin] = skin;
  p[kSkinShade] = skinShade;
  p[kEyeWhite] = rgba(250, 250, 255);
  p[kEyeDark] = rgba(30, 28, 44);
  p[kBlush] = rgba(240, 130, 130);
  p[kShirt] = rgba(70, 150, 230);
  p[kShirtDark] = rgba(40, 95, 170);
  p[kPants] = rgba(92, 72, 58);
  p[kPantsDark] = rgba(66, 50, 40);
  p[kShoe] = rgba(120, 78, 44);
  p[kSole] = rgba(58, 40, 28);
  p[kBelt] = rgba(80, 52, 30);
  p[kBuckle] = rgba(250, 205, 70);
  p[kMouth] = rgba(150, 70, 70);
  return p;
}

struct SkinTone { uint32_t skin, shade; };
const SkinTone kSkinTones[ModelLibrary::kSkinToneCount] = {
    {rgba(255, 213, 170), rgba(232, 180, 140)},
    {rgba(230, 170, 120), rgba(200, 140, 96)},
    {rgba(172, 112, 72), rgba(142, 88, 56)},
    {rgba(110, 70, 46), rgba(86, 54, 36)},
};

// Adds skin tone variants 1..3 to a body part.
void addSkinVariants(VoxelPart &p) {
  const std::vector<uint32_t> base = p.palette;
  for (int t = 1; t < ModelLibrary::kSkinToneCount; ++t) {
    std::vector<uint32_t> v = base;
    v[kSkin] = kSkinTones[t].skin;
    v[kSkinShade] = kSkinTones[t].shade;
    p.variantPalettes.insert(p.variantPalettes.end(), v.begin(), v.end());
  }
}

Painter bodyPainter(const std::string &name, int sx, int sy, int sz, glm::vec3 pivot) {
  Painter p(name, sx, sy, sz, pivot);
  p.part.palette = bodyPalette(kSkinTones[0].skin, kSkinTones[0].shade);
  return p;
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

void buildBody(ModelLibrary &lib) {
  for (uint8_t shape = 0; shape < ModelLibrary::kBodyShapeCount; ++shape) {
    const bool broad = shape == 1;
    const std::string sfx = broad ? "_broad" : "_slim";
    auto add = [&](Bone bone, Painter &p) {
      addSkinVariants(p.part);
      lib.setBodyPart(bone, shape, lib.addPart(std::move(p.part)));
    };

    { // Head 9x9x9 (same for both shapes): big eyes, blush, tiny mouth.
      Painter p = bodyPainter("body_head" + sfx, 9, 9, 9, {4.5f, 0.0f, 4.5f});
      p.box(0, 0, 0, 9, 9, 9, kSkin);
      p.box(0, 0, 0, 9, 1, 9, kSkinShade);           // jaw shadow row
      p.box(1, 3, 0, 3, 5, 1, kEyeWhite);            // left eye (x small = left)
      p.box(6, 3, 0, 8, 5, 1, kEyeWhite);
      p.set(2, 3, 0, kEyeDark);
      p.set(2, 4, 0, kEyeDark);
      p.set(6, 3, 0, kEyeDark);
      p.set(6, 4, 0, kEyeDark);
      p.set(1, 2, 0, kBlush);
      p.set(7, 2, 0, kBlush);
      p.box(4, 1, 0, 5, 2, 1, kMouth);
      p.box(0, 4, 3, 1, 6, 5, kSkinShade);           // ears
      p.box(8, 4, 3, 9, 6, 5, kSkinShade);
      add(Bone::Head, p);
    }
    { // Torso: shirt with collar and a darker hem.
      const int w = broad ? 10 : 9, d = broad ? 6 : 5;
      Painter p = bodyPainter("body_torso" + sfx, w, 6, d, {w * 0.5f, 0.0f, d * 0.5f});
      p.box(0, 0, 0, w, 6, d, kShirt);
      p.box(0, 0, 0, w, 1, d, kShirtDark);
      p.box(w / 2 - 1, 4, 0, w / 2 + 1 + (w & 1), 6, 1, kSkin); // collar V
      p.box(w / 2, 3, 0, w / 2 + (w & 1), 4, 1, kShirtDark);
      add(Bone::Torso, p);
    }
    { // Pelvis: belt with buckle over trousers.
      const int w = broad ? 10 : 8, d = broad ? 6 : 5;
      Painter p = bodyPainter("body_pelvis" + sfx, w, 2, d, {w * 0.5f, 0.0f, d * 0.5f});
      p.box(0, 0, 0, w, 2, d, kPants);
      p.box(0, 1, 0, w, 2, d, kBelt);
      p.box(w / 2 - 1, 1, 0, w / 2 + 1, 2, 1, kBuckle);
      add(Bone::Pelvis, p);
    }
    const int aw = broad ? 4 : 3;
    for (int side = 0; side < 2; ++side) {
      const char *lr = side == 0 ? "_L" : "_R";
      { // Upper arm: short sleeve over skin.
        Painter p = bodyPainter("body_arm_upper" + sfx + lr, aw, 4, aw, {aw * 0.5f, 4.0f, aw * 0.5f});
        p.box(0, 0, 0, aw, 4, aw, kSkin);
        p.box(0, 2, 0, aw, 4, aw, kShirt);
        p.box(0, 2, 0, aw, 3, aw, kShirtDark);
        add(side == 0 ? Bone::ArmUpperL : Bone::ArmUpperR, p);
      }
      {
        Painter p = bodyPainter("body_arm_lower" + sfx + lr, aw, 3, aw, {aw * 0.5f, 3.0f, aw * 0.5f});
        p.box(0, 0, 0, aw, 3, aw, kSkin);
        add(side == 0 ? Bone::ArmLowerL : Bone::ArmLowerR, p);
      }
      { // Hand: a mitten fist with a darker knuckle row.
        Painter p = bodyPainter("body_hand" + sfx + lr, 3, 3, 3, {1.5f, 3.0f, 1.5f});
        p.box(0, 0, 0, 3, 3, 3, kSkin);
        p.box(0, 0, 0, 3, 1, 3, kSkinShade);
        add(side == 0 ? Bone::HandL : Bone::HandR, p);
      }
      {
        Painter p = bodyPainter("body_leg_upper" + sfx + lr, 4, 4, 4, {2.0f, 4.0f, 2.0f});
        p.box(0, 0, 0, 4, 4, 4, kPants);
        p.box(0, 0, 0, 4, 1, 4, kPantsDark);
        add(side == 0 ? Bone::LegUpperL : Bone::LegUpperR, p);
      }
      {
        Painter p = bodyPainter("body_leg_lower" + sfx + lr, 4, 3, 4, {2.0f, 3.0f, 2.0f});
        p.box(0, 0, 0, 4, 3, 4, kPants);
        p.box(0, 2, 0, 4, 3, 4, kPantsDark);
        add(side == 0 ? Bone::LegLowerL : Bone::LegLowerR, p);
      }
      { // Foot: shoe with a sole row and a toe cap; toes toward z = 0 (front).
        Painter p = bodyPainter("body_foot" + sfx + lr, 4, 2, 6, {2.0f, 2.0f, 4.0f});
        p.box(0, 0, 0, 4, 2, 6, kShoe);
        p.box(0, 0, 0, 4, 1, 6, kSole);
        p.box(0, 1, 0, 4, 2, 1, kSole);
        add(side == 0 ? Bone::FootL : Bone::FootR, p);
      }
    }
    // Root has no geometry.
    lib.setBodyPart(Bone::Root, shape, -1);
  }
}

// ---------------------------------------------------------------------------
// Hair (Socket::Head: pivot at the crown centre)
// ---------------------------------------------------------------------------

const uint32_t kHairColors[ModelLibrary::kHairColorCount][2] = {
    {rgba(110, 66, 36), rgba(84, 48, 26)},    // brown
    {rgba(40, 36, 44), rgba(24, 22, 30)},     // black
    {rgba(250, 214, 110), rgba(222, 176, 72)},// blonde
    {rgba(214, 88, 44), rgba(170, 60, 30)},   // ginger
};

void addHairVariants(VoxelPart &p) {
  for (int c = 1; c < ModelLibrary::kHairColorCount; ++c) {
    std::vector<uint32_t> v = p.palette;
    v[1] = kHairColors[c][0];
    v[2] = kHairColors[c][1];
    p.variantPalettes.insert(p.variantPalettes.end(), v.begin(), v.end());
  }
}

// Common cap: 11 wide/deep around the 9^3 head (which spans x,z 1..10 and
// y < 3 in this part's space; the crown is at y = 3).
void hairCap(Painter &p, int backDown) {
  p.box(0, 2, 0, 11, 5, 11, 1);                   // top cap
  p.box(0, 4, 0, 11, 5, 11, 2);                   // darker top row for depth
  p.box(0, 3 - backDown, 7, 11, 3, 11, 1);        // back of the head
  p.box(0, 0, 3, 1, 3, 8, 1);                     // sideburns
  p.box(10, 0, 3, 11, 3, 8, 1);
  p.box(1, 1, 0, 4, 2, 1, 1);                     // fringe tufts
  p.box(7, 1, 0, 9, 2, 1, 1);
  p.clearBox(1, 0, 1, 10, 2, 7);                  // keep the face free
}

void buildHair(ModelLibrary &lib, std::vector<int16_t> &hair) {
  auto finish = [&](Painter &p) {
    addHairVariants(p.part);
    hair.push_back(int16_t(lib.addPart(std::move(p.part))));
  };
  { // 0: short
    Painter p("hair_short", 11, 5, 11, {5.5f, 3.0f, 5.5f});
    p.part.palette = {0, kHairColors[0][0], kHairColors[0][1]};
    hairCap(p, 3);
    finish(p);
  }
  { // 1: long with a ponytail
    Painter p("hair_long", 11, 13, 13, {5.5f, 11.0f, 5.5f});
    p.part.palette = {0, kHairColors[0][0], kHairColors[0][1]};
    // Same cap shifted up by 8 rows.
    p.box(0, 10, 0, 11, 13, 11, 1);
    p.box(0, 12, 0, 11, 13, 11, 2);
    p.box(0, 2, 7, 11, 11, 11, 1);                // long back
    p.box(0, 2, 7, 11, 3, 11, 2);
    p.box(0, 5, 3, 1, 11, 8, 1);
    p.box(10, 5, 3, 11, 11, 8, 1);
    p.box(1, 9, 0, 4, 10, 1, 1);
    p.box(7, 9, 0, 9, 10, 1, 1);
    p.clearBox(1, 0, 1, 10, 10, 7);
    p.box(4, 0, 11, 7, 8, 13, 1);                 // ponytail
    p.box(4, 7, 11, 7, 8, 13, 2);                 // hair tie band
    finish(p);
  }
  { // 2: spiky
    Painter p("hair_spiky", 11, 8, 11, {5.5f, 3.0f, 5.5f});
    p.part.palette = {0, kHairColors[0][0], kHairColors[0][1]};
    hairCap(p, 2);
    for (int x = 1; x < 11; x += 3)
      for (int z = 1; z < 11; z += 3) {
        const int h = 5 + ((x * 7 + z * 3) % 3);
        p.box(x, 5, z, x + 2, h, z + 2, (x + z) % 2 ? 1 : 2);
      }
    finish(p);
  }
}

// ---------------------------------------------------------------------------
// Equipment
// ---------------------------------------------------------------------------

PieceId replacePiece(ModelLibrary &lib, const char *name, EquipSlot slot,
                     std::initializer_list<std::pair<Bone, int>> parts) {
  EquipPiece e;
  e.name = name;
  e.slot = slot;
  for (const auto &bp : parts)
    e.boneParts[size_t(bp.first)] = int16_t(bp.second);
  return lib.addPiece(std::move(e));
}

PieceId socketPiece(ModelLibrary &lib, const char *name, EquipSlot slot, Socket socket,
                    int part, uint8_t weaponType = 0) {
  EquipPiece e;
  e.name = name;
  e.slot = slot;
  e.socket = socket;
  e.socketPart = int16_t(part);
  e.weaponType = weaponType;
  return lib.addPiece(std::move(e));
}

// Material sets for armour tiers.
struct Tier {
  const char *name;
  uint32_t main, dark, trim, accent;
};
const Tier kLeather{"leather", rgba(150, 96, 52), rgba(112, 68, 36), rgba(196, 150, 96), rgba(230, 196, 90)};
const Tier kIron{"iron", rgba(186, 194, 206), rgba(120, 128, 142), rgba(232, 236, 242), rgba(70, 110, 200)};

void buildHelmets(ModelLibrary &lib) {
  { // Leather cap: rounded cap with a front brim and stitched band.
    Painter p("leather_cap", 11, 6, 12, {5.5f, 3.0f, 6.5f});
    const int m = p.c(kLeather.main), d = p.c(kLeather.dark), t = p.c(kLeather.trim);
    p.box(0, 2, 1, 11, 5, 12, m);
    p.box(1, 5, 2, 10, 6, 11, m);
    p.box(0, 2, 1, 11, 3, 12, d);                  // band
    p.box(2, 2, 0, 9, 3, 1, t);                    // brim
    p.set(5, 5, 6, t);                             // button on top
    p.clearBox(1, 0, 2, 10, 2, 11);
    socketPiece(lib, "leather_cap", EquipSlot::Head, Socket::Head, lib.addPart(std::move(p.part)));
  }
  { // Iron helm: full helm with face opening, nose guard and a blue crest.
    Painter p("iron_helm", 11, 12, 11, {5.5f, 9.0f, 5.5f});
    const int m = p.c(kIron.main), d = p.c(kIron.dark), t = p.c(kIron.trim), a = p.c(kIron.accent);
    p.box(0, 1, 0, 11, 11, 11, m);
    p.clearBox(1, 0, 1, 10, 10, 10);               // hollow for the head
    p.clearBox(1, 3, 0, 10, 7, 1);                 // face opening
    p.box(5, 3, 0, 6, 7, 1, t);                    // nose guard
    p.box(0, 7, 0, 11, 8, 11, d);                  // brow band
    p.box(0, 1, 0, 11, 2, 11, d);                  // rim
    p.box(5, 10, 2, 6, 12, 10, a);                 // crest
    p.set(0, 5, 5, t);
    p.set(10, 5, 5, t);                            // rivets
    socketPiece(lib, "iron_helm", EquipSlot::Head, Socket::Head, lib.addPart(std::move(p.part)));
  }
  { // Crystal crown: gold ring with glowing crystal spikes.
    Painter p("crystal_crown", 11, 7, 11, {5.5f, 2.0f, 5.5f});
    const int gold = p.c(rgba(250, 200, 60)), goldD = p.c(rgba(200, 140, 30));
    const int cr = p.g(rgba(120, 240, 255)), cr2 = p.g(rgba(220, 120, 255));
    p.box(0, 1, 0, 11, 3, 11, gold);
    p.clearBox(1, 1, 1, 10, 3, 10);
    p.box(0, 1, 0, 11, 2, 11, goldD);
    p.clearBox(1, 1, 1, 10, 2, 10);
    const int spikes[5][2] = {{5, 0}, {1, 1}, {9, 1}, {0, 6}, {10, 6}};
    for (int i = 0; i < 5; ++i) {
      const int x = spikes[i][0], z = spikes[i][1];
      p.box(x, 3, z, x + 1, i == 0 ? 7 : 5, z + 1, i == 0 ? cr2 : cr);
    }
    p.set(5, 2, 0, cr2);                           // front gem
    socketPiece(lib, "crystal_crown", EquipSlot::Head, Socket::Head, lib.addPart(std::move(p.part)));
  }
}

// Torso armour: torso shell + two upper-arm pads.
void armourTorso(ModelLibrary &lib, const char *name, const Tier &t, bool plate, bool robe) {
  const int base = robe ? 2 : 0; // robe hangs 2 rows below the waist
  Painter body(std::string(name) + "_torso", 12, 6 + base, 8, {6.0f, float(base), 4.0f});
  const int m = body.c(t.main), d = body.c(t.dark), tr = body.c(t.trim), a = body.c(t.accent);
  body.box(0, 0, 0, 12, 6 + base, 8, m);
  body.box(0, 0, 0, 12, 1, 8, d);
  body.box(0, 5 + base, 0, 12, 6 + base, 8, tr);            // shoulders line
  if (plate) {
    body.box(2, 1 + base, 0, 10, 5 + base, 1, tr);          // breastplate
    body.box(5, 1 + base, 0, 7, 5 + base, 1, a);            // emblem stripe
    body.box(0, base, 0, 12, base + 1, 8, d);
  } else if (robe) {
    body.box(5, 0, 0, 7, 6 + base, 1, tr);                  // front trim
    body.box(0, base + 1, 0, 12, base + 2, 8, a);           // sash
    body.set(6, base + 1, 0, tr);
  } else {
    body.box(5, 2, 0, 7, 6, 1, d);                          // lacing
    for (int y = 2; y < 6; y += 2)
      body.set(6, y, 0, tr);
    body.box(0, 1, 0, 12, 2, 8, a);                         // belt
  }
  body.clearBox(4, 5 + base, 2, 8, 6 + base, 6);            // neck hole
  const int torsoPart = lib.addPart(std::move(body.part));

  int arms[2];
  for (int side = 0; side < 2; ++side) {
    Painter p(std::string(name) + (side ? "_arm_R" : "_arm_L"), 5, 5, 5, {2.5f, 4.0f, 2.5f});
    const int pm = p.c(t.main), pd = p.c(t.dark), ptr = p.c(t.trim);
    p.box(0, 0, 0, 5, 5, 5, pm);
    p.box(0, 0, 0, 5, 1, 5, pd);
    if (plate)
      p.box(0, 3, 0, 5, 5, 5, ptr);                         // pauldron
    else
      p.box(0, 4, 0, 5, 5, 5, ptr);
    arms[side] = lib.addPart(std::move(p.part));
  }
  replacePiece(lib, name, EquipSlot::Torso,
               {{Bone::Torso, torsoPart}, {Bone::ArmUpperL, arms[0]}, {Bone::ArmUpperR, arms[1]}});
}

void armourHands(ModelLibrary &lib, const char *name, const Tier &t, bool plate) {
  int lower[2], hand[2];
  for (int side = 0; side < 2; ++side) {
    const char *lr = side ? "_R" : "_L";
    Painter a(std::string(name) + "_forearm" + lr, 4, 3, 4, {2.0f, 3.0f, 2.0f});
    const int m = a.c(t.main), d = a.c(t.dark), tr = a.c(t.trim);
    a.box(0, 0, 0, 4, 3, 4, m);
    a.box(0, 2, 0, 4, 3, 4, plate ? tr : d);                 // cuff
    lower[side] = lib.addPart(std::move(a.part));
    Painter h(std::string(name) + "_hand" + lr, 4, 3, 4, {2.0f, 3.0f, 2.0f});
    const int hm = h.c(t.main), hd = h.c(t.dark);
    h.box(0, 0, 0, 4, 3, 4, hm);
    h.box(0, 0, 0, 4, 1, 4, hd);                             // knuckles
    hand[side] = lib.addPart(std::move(h.part));
  }
  replacePiece(lib, name, EquipSlot::Hands,
               {{Bone::ArmLowerL, lower[0]}, {Bone::HandL, hand[0]},
                {Bone::ArmLowerR, lower[1]}, {Bone::HandR, hand[1]}});
}

void armourLegs(ModelLibrary &lib, const char *name, const Tier &t, bool plate) {
  Painter pel(std::string(name) + "_pelvis", 11, 3, 7, {5.5f, 1.0f, 3.5f});
  {
    const int m = pel.c(t.main), d = pel.c(t.dark), a = pel.c(t.accent);
    pel.box(0, 0, 0, 11, 3, 7, m);
    pel.box(0, 1, 0, 11, 2, 7, d);                           // belt
    pel.box(5, 1, 0, 6, 2, 1, a);                            // buckle
  }
  const int pelvisPart = lib.addPart(std::move(pel.part));
  int upper[2], lower[2];
  for (int side = 0; side < 2; ++side) {
    const char *lr = side ? "_R" : "_L";
    Painter u(std::string(name) + "_thigh" + lr, 5, 4, 5, {2.5f, 4.0f, 2.5f});
    const int um = u.c(t.main), ud = u.c(t.dark), ut = u.c(t.trim);
    u.box(0, 0, 0, 5, 4, 5, um);
    u.box(0, 0, 0, 5, 1, 5, ud);
    if (plate)
      u.box(1, 1, 0, 4, 3, 1, ut);                           // thigh plate
    upper[side] = lib.addPart(std::move(u.part));
    Painter l(std::string(name) + "_shin" + lr, 5, 3, 5, {2.5f, 3.0f, 2.5f});
    const int lm = l.c(t.main), ld = l.c(t.dark), lt = l.c(t.trim);
    l.box(0, 0, 0, 5, 3, 5, lm);
    l.box(0, 2, 0, 5, 3, 5, ld);
    if (plate)
      l.box(1, 0, 0, 4, 3, 1, lt);                           // knee guard
    lower[side] = lib.addPart(std::move(l.part));
  }
  replacePiece(lib, name, EquipSlot::Legs,
               {{Bone::Pelvis, pelvisPart}, {Bone::LegUpperL, upper[0]}, {Bone::LegLowerL, lower[0]},
                {Bone::LegUpperR, upper[1]}, {Bone::LegLowerR, lower[1]}});
}

void buildBoots(ModelLibrary &lib) {
  int feet[2];
  for (int side = 0; side < 2; ++side) {
    Painter p(side ? "boots_R" : "boots_L", 5, 4, 7, {2.5f, 2.0f, 4.5f});
    const int m = p.c(rgba(132, 84, 46)), d = p.c(rgba(92, 56, 30)), s = p.c(rgba(50, 36, 26)),
              b = p.c(rgba(230, 196, 90));
    p.box(0, 0, 0, 5, 3, 7, m);
    p.box(0, 3, 3, 5, 4, 7, d);                              // cuff
    p.box(0, 0, 0, 5, 1, 7, s);                              // sole
    p.box(0, 1, 0, 5, 2, 1, d);                              // toe cap
    p.set(2, 2, 3, b);                                       // buckle
    feet[side] = lib.addPart(std::move(p.part));
  }
  replacePiece(lib, "boots", EquipSlot::Feet, {{Bone::FootL, feet[0]}, {Bone::FootR, feet[1]}});
}

void buildBack(ModelLibrary &lib) {
  { // Red cape hanging from the shoulders to the knees.
    Painter p("red_cape", 11, 14, 2, {5.5f, 14.0f, 0.0f});
    const int r = p.c(rgba(210, 40, 50)), rd = p.c(rgba(150, 24, 36)), gold = p.c(rgba(250, 200, 60));
    p.box(0, 0, 1, 11, 14, 2, r);
    p.box(0, 0, 1, 11, 2, 2, rd);                            // darker hem
    for (int x = 1; x < 11; x += 3)
      p.box(x, 0, 1, x + 1, 13, 2, rd);                      // folds
    p.box(0, 13, 0, 11, 14, 2, gold);                        // collar
    p.set(2, 13, 0, gold);
    socketPiece(lib, "red_cape", EquipSlot::Back, Socket::Back, lib.addPart(std::move(p.part)));
  }
  { // Glider wings: two membranes on bone ribs, spread wide, glowing tips.
    Painter p("glider_wings", 31, 11, 3, {15.5f, 7.0f, 0.0f});
    const int bone = p.c(rgba(240, 236, 220)), mem = p.c(rgba(120, 200, 255)), memD = p.c(rgba(80, 150, 220));
    const int tip = p.g(rgba(255, 250, 170));
    p.box(13, 4, 0, 18, 9, 2, bone);                         // harness
    for (int side = 0; side < 2; ++side) {
      for (int i = 0; i < 13; ++i) {
        const int x = side ? 18 + i : 12 - i;
        const int top = 10 - i / 4;                          // leading edge slopes down
        const int bottom = 2 + i / 3;
        p.box(x, bottom, 1, x + 1, top, 2, (i % 4 == 3) ? memD : mem);
        p.set(x, top, 1, bone);
      }
      const int tx = side ? 30 : 0;
      p.box(tx, 6, 1, tx + 1, 8, 2, tip);
    }
    socketPiece(lib, "glider_wings", EquipSlot::Back, Socket::Back, lib.addPart(std::move(p.part)));
  }
}

// Weapons: held at the fist centre (the part pivot), pointing forward (-Z,
// i.e. toward voxel z = 0) when the arm hangs down.
int makeSword(ModelLibrary &lib, const char *name, uint32_t blade, uint32_t edge, uint32_t guard,
              uint32_t grip, bool glow) {
  Painter p(name, 3, 5, 20, {1.5f, 2.5f, 17.5f});
  const int g = p.c(guard), gr = p.c(grip);
  const int b = glow ? p.g(blade) : p.c(blade);
  const int e = glow ? p.g(edge) : p.c(edge);
  p.box(1, 2, 19, 2, 3, 20, g);                              // pommel
  p.box(1, 2, 16, 2, 3, 19, gr);                             // grip
  p.box(0, 0, 15, 3, 5, 16, g);                              // crossguard
  p.box(1, 1, 2, 2, 4, 15, b);                               // blade
  p.box(1, 1, 2, 2, 2, 15, e);                               // edges
  p.box(1, 3, 2, 2, 4, 15, e);
  p.box(1, 2, 0, 2, 3, 2, e);                                // tip
  return lib.addPart(std::move(p.part));
}

void buildWeapons(ModelLibrary &lib) {
  constexpr uint8_t kSword = 1, kBow = 2, kStaff = 3, kPick = 4; // = ao::WeaponType
  socketPiece(lib, "wooden_sword", EquipSlot::MainHand, Socket::MainHand,
              makeSword(lib, "wooden_sword", rgba(176, 126, 72), rgba(206, 160, 100),
                        rgba(120, 80, 44), rgba(90, 58, 32), false), kSword);
  socketPiece(lib, "iron_sword", EquipSlot::MainHand, Socket::MainHand,
              makeSword(lib, "iron_sword", rgba(196, 204, 216), rgba(245, 248, 255),
                        rgba(80, 110, 200), rgba(70, 44, 28), false), kSword);
  socketPiece(lib, "crystal_sword", EquipSlot::MainHand, Socket::MainHand,
              makeSword(lib, "crystal_sword", rgba(110, 230, 255), rgba(230, 250, 255),
                        rgba(250, 200, 60), rgba(90, 40, 120), true), kSword);
  { // Bow: vertical arc, string toward the player (+Z).
    Painter p("bow", 3, 21, 6, {1.5f, 10.5f, 1.5f});
    const int wood = p.c(rgba(150, 96, 50)), woodD = p.c(rgba(108, 66, 34)),
              wrap = p.c(rgba(210, 60, 50)), str = p.c(rgba(240, 240, 230));
    for (int y = 0; y < 21; ++y) {
      const int dy = y < 10 ? 10 - y : y - 10;
      const int z = dy < 4 ? 1 : (dy < 7 ? 2 : (dy < 9 ? 3 : 4)); // limbs curve back
      p.set(1, y, z, (dy % 3 == 0) ? woodD : wood);
    }
    p.box(1, 9, 0, 2, 12, 2, wrap);                          // grip wrap
    p.box(1, 1, 5, 2, 20, 6, str);                           // string
    socketPiece(lib, "bow", EquipSlot::MainHand, Socket::MainHand, lib.addPart(std::move(p.part)), kBow);
  }
  { // Staff: long shaft, gold cage, glowing orb.
    Painter p("staff", 5, 28, 5, {2.5f, 9.5f, 2.5f});
    const int wood = p.c(rgba(120, 80, 50)), woodD = p.c(rgba(90, 56, 34)), gold = p.c(rgba(250, 200, 60));
    const int orb = p.g(rgba(170, 110, 255)), orbL = p.g(rgba(230, 200, 255));
    p.box(2, 0, 2, 3, 22, 3, wood);
    for (int y = 1; y < 22; y += 4)
      p.set(2, y, 2, woodD);
    p.box(1, 21, 1, 4, 22, 4, gold);
    p.box(1, 22, 1, 4, 25, 4, orb);
    p.set(2, 25, 2, orbL);
    p.set(2, 23, 1, orbL);
    p.box(0, 22, 2, 1, 26, 3, gold);                         // cage prongs
    p.box(4, 22, 2, 5, 26, 3, gold);
    p.box(2, 22, 0, 3, 26, 1, gold);
    p.box(2, 22, 4, 3, 26, 5, gold);
    p.box(2, 26, 2, 3, 28, 3, gold);
    socketPiece(lib, "staff", EquipSlot::MainHand, Socket::MainHand, lib.addPart(std::move(p.part)), kStaff);
  }
  { // Pickaxe: handle forward, iron head at the far end.
    Painter p("pickaxe", 3, 11, 16, {1.5f, 5.5f, 12.5f});
    const int wood = p.c(rgba(150, 100, 56)), woodD = p.c(rgba(110, 70, 40)),
              iron = p.c(rgba(170, 176, 190)), ironL = p.c(rgba(230, 234, 240));
    p.box(1, 5, 2, 2, 6, 16, wood);
    p.box(1, 5, 14, 2, 6, 16, woodD);
    p.box(0, 4, 0, 3, 7, 3, iron);                           // head socket
    p.box(1, 7, 1, 2, 10, 2, iron);                          // upper pick
    p.box(1, 1, 1, 2, 4, 2, iron);                           // lower pick
    p.set(1, 10, 1, ironL);
    p.set(1, 0, 1, ironL);
    socketPiece(lib, "pickaxe", EquipSlot::MainHand, Socket::MainHand, lib.addPart(std::move(p.part)), kPick);
  }
  { // Wooden shield: planks, iron rim and boss, held in front of the left fist.
    Painter p("wooden_shield", 10, 11, 2, {5.0f, 5.5f, 3.0f});
    const int w1 = p.c(rgba(170, 118, 64)), w2 = p.c(rgba(146, 98, 52)),
              rim = p.c(rgba(150, 156, 170)), boss = p.c(rgba(210, 214, 222));
    for (int x = 0; x < 10; ++x)
      p.box(x, 0, 1, x + 1, 11, 2, (x / 2) % 2 ? w2 : w1);
    p.box(0, 0, 0, 10, 1, 2, rim);
    p.box(0, 10, 0, 10, 11, 2, rim);
    p.box(0, 0, 0, 1, 11, 2, rim);
    p.box(9, 0, 0, 10, 11, 2, rim);
    p.box(4, 4, 0, 6, 7, 1, boss);
    socketPiece(lib, "wooden_shield", EquipSlot::OffHand, Socket::OffHand, lib.addPart(std::move(p.part)));
  }
}

} // namespace

void buildMonsterModels(ModelLibrary &lib); // ModelMonsters.cpp

void ModelLibrary::buildDefaults() {
  parts_.clear();
  pieces_.assign(1, EquipPiece{});
  for (auto &shape : body_)
    shape.fill(-1);
  hair_.clear();
  creatures_.clear();
  materialBase_.clear();
  materialCount_ = 0;

  buildBody(*this);
  buildHair(*this, hair_);
  buildHelmets(*this);
  armourTorso(*this, "leather_tunic", kLeather, false, false);
  armourTorso(*this, "iron_chestplate", kIron, true, false);
  armourTorso(*this, "mage_robe",
              Tier{"robe", rgba(90, 60, 170), rgba(60, 38, 120), rgba(250, 200, 60), rgba(200, 60, 120)},
              false, true);
  armourHands(*this, "leather_gloves", kLeather, false);
  armourHands(*this, "iron_gauntlets", kIron, true);
  armourLegs(*this, "leather_pants", kLeather, false);
  armourLegs(*this, "iron_greaves", kIron, true);
  buildBoots(*this);
  buildBack(*this);
  buildWeapons(*this);
  buildMonsterModels(*this);
}

} // namespace atm::model
