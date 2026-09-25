// Procedural monster models: slime (slime rig), wolf (quadruped rig) and a
// stone golem (humanoid rig, drawn 1.5x). Names match MonsterDef::model.

#include "Character.h"
#include "ModelContentDetail.h"

#include <string>
#include <utility>

namespace atm::model {

using namespace content_detail;

namespace {

void buildSlime(ModelLibrary &lib) {
  // Squashy rounded cube: 12 x 10 x 12, pivot bottom centre, face on z = 0.
  Painter p("slime_body", 12, 10, 12, {6.0f, 0.0f, 6.0f});
  const int body = p.c(rgba(110, 220, 90)), dark = p.c(rgba(70, 170, 60)),
            light = p.c(rgba(200, 255, 170)), eyeW = p.c(rgba(250, 255, 250)),
            eyeD = p.c(rgba(24, 40, 24)), mouth = p.c(rgba(40, 90, 40));
  p.box(0, 0, 0, 12, 9, 12, body);
  p.box(1, 9, 1, 11, 10, 11, body);                     // domed top
  p.roundEdges();
  p.box(0, 0, 0, 12, 1, 12, dark);                      // darker base
  p.clearBox(0, 0, 0, 1, 1, 12);
  p.clearBox(11, 0, 0, 12, 1, 12);
  p.speckle(0, 1, 0, 12, 9, 12, dark, 0.08f, 11u);
  p.box(3, 9, 3, 5, 10, 5, light);                      // shine
  p.set(8, 8, 0, light);
  p.box(2, 4, 0, 5, 7, 1, eyeW);                        // eyes
  p.box(7, 4, 0, 10, 7, 1, eyeW);
  p.box(3, 4, 0, 5, 6, 1, eyeD);
  p.box(7, 4, 0, 9, 6, 1, eyeD);
  p.box(5, 2, 0, 7, 3, 1, mouth);
  const int part = lib.addPart(std::move(p.part));

  CreatureModel m;
  m.name = "slime";
  m.rig = &Rig::slime();
  m.boneParts[size_t(Bone::Pelvis)] = int16_t(part);
  m.idleClip = "slime_idle";
  m.moveClip = "slime_hop";
  m.attackClip = "slime_attack";
  m.hitClip = "slime_hit";
  m.deathClip = "slime_death";
  lib.addCreature(std::move(m));
}

void buildWolf(ModelLibrary &lib) {
  const uint32_t fur = rgba(130, 134, 150), furD = rgba(92, 94, 110), furL = rgba(214, 216, 226);
  CreatureModel m;
  m.name = "wolf";
  m.rig = &Rig::quadruped();

  { // Hind body with a bushy tail sticking out the back (+Z).
    Painter p("wolf_hind", 7, 8, 10, {3.5f, 3.0f, 3.5f});
    const int f = p.c(fur), d = p.c(furD), l = p.c(furL);
    p.box(0, 0, 0, 7, 6, 7, f);
    p.box(0, 5, 0, 7, 6, 7, d);                         // darker back
    p.box(2, 3, 7, 5, 7, 9, f);                         // tail
    p.box(2, 6, 7, 5, 7, 10, d);
    p.box(2, 3, 9, 5, 5, 10, l);                        // tail tip
    p.speckle(0, 0, 0, 7, 5, 7, d, 0.1f, 21u);
    m.boneParts[size_t(Bone::Pelvis)] = int16_t(lib.addPart(std::move(p.part)));
  }
  { // Chest with a pale neck ruff.
    Painter p("wolf_chest", 8, 8, 8, {4.0f, 3.5f, 4.0f});
    const int f = p.c(fur), d = p.c(furD), l = p.c(furL);
    p.box(0, 0, 0, 8, 7, 8, f);
    p.box(0, 6, 1, 8, 8, 7, d);                         // hackles
    p.box(1, 0, 0, 7, 5, 2, l);                         // ruff
    p.speckle(0, 0, 2, 8, 6, 8, d, 0.1f, 22u);
    m.boneParts[size_t(Bone::Torso)] = int16_t(lib.addPart(std::move(p.part)));
  }
  { // Head: skull + snout toward z = 0, ears, eyes, black nose.
    Painter p("wolf_head", 6, 8, 10, {3.0f, 2.0f, 7.0f});
    const int f = p.c(fur), d = p.c(furD), l = p.c(furL), nose = p.c(rgba(24, 24, 30)),
              eye = p.c(rgba(250, 200, 60));
    p.box(0, 0, 4, 6, 5, 10, f);                        // skull
    p.box(1, 0, 0, 5, 3, 4, l);                         // snout
    p.box(1, 2, 0, 5, 3, 4, f);
    p.box(2, 1, 0, 4, 3, 1, nose);
    p.box(0, 5, 7, 2, 8, 9, d);                         // ears
    p.box(4, 5, 7, 6, 8, 9, d);
    p.set(1, 3, 4, eye);
    p.set(4, 3, 4, eye);
    p.box(0, 4, 4, 6, 5, 5, d);                         // brow ridge
    m.boneParts[size_t(Bone::Head)] = int16_t(lib.addPart(std::move(p.part)));
  }
  struct LegSpec { Bone upper, lower, paw; const char *name; float pawPivotY; };
  const LegSpec legs[4] = {
      {Bone::ArmUpperL, Bone::ArmLowerL, Bone::HandL, "front_L", 1.0f},
      {Bone::ArmUpperR, Bone::ArmLowerR, Bone::HandR, "front_R", 1.0f},
      {Bone::LegUpperL, Bone::LegLowerL, Bone::FootL, "hind_L", 1.5f},
      {Bone::LegUpperR, Bone::LegLowerR, Bone::FootR, "hind_R", 1.5f},
  };
  for (const LegSpec &s : legs) {
    Painter u(std::string("wolf_leg_upper_") + s.name, 3, 4, 3, {1.5f, 4.0f, 1.5f});
    u.box(0, 0, 0, 3, 4, 3, u.c(fur));
    m.boneParts[size_t(s.upper)] = int16_t(lib.addPart(std::move(u.part)));
    Painter l(std::string("wolf_leg_lower_") + s.name, 3, 3, 3, {1.5f, 3.0f, 1.5f});
    l.box(0, 0, 0, 3, 3, 3, l.c(furD));
    m.boneParts[size_t(s.lower)] = int16_t(lib.addPart(std::move(l.part)));
    Painter pw(std::string("wolf_paw_") + s.name, 3, 2, 4, {1.5f, s.pawPivotY, 2.5f});
    const int pl = pw.c(furL), claw = pw.c(rgba(40, 40, 44));
    pw.box(0, 0, 0, 3, 2, 4, pl);
    pw.box(0, 0, 0, 3, 1, 1, claw);
    m.boneParts[size_t(s.paw)] = int16_t(lib.addPart(std::move(pw.part)));
  }
  m.idleClip = "quad_idle";
  m.moveClip = "quad_walk";
  m.attackClip = "quad_attack";
  m.hitClip = "quad_hit";
  m.deathClip = "quad_death";
  lib.addCreature(std::move(m));
}

void buildGolem(ModelLibrary &lib) {
  const uint32_t stone = rgba(128, 124, 120), stoneD = rgba(92, 88, 86), moss = rgba(96, 150, 70);
  const uint32_t core = rgba(90, 230, 255);
  CreatureModel m;
  m.name = "golem";
  m.rig = &Rig::humanoid();
  m.scale = 1.5f;

  auto rock = [&](const char *name, int sx, int sy, int sz, glm::vec3 pivot, uint32_t seed,
                  bool mossTop) {
    Painter p(name, sx, sy, sz, pivot);
    const int s = p.c(stone), d = p.c(stoneD), mo = p.c(moss);
    p.box(0, 0, 0, sx, sy, sz, s);
    p.speckle(0, 0, 0, sx, sy, sz, d, 0.25f, seed);
    if (mossTop)
      p.speckle(0, sy - 1, 0, sx, sy, sz, mo, 0.6f, seed + 7u);
    return p;
  };

  {
    Painter p = rock("golem_torso", 12, 7, 8, {6.0f, 0.0f, 4.0f}, 31u, true);
    const int c = p.g(core);
    p.box(5, 2, 0, 7, 5, 1, c);                          // glowing core
    p.set(4, 3, 0, c);
    p.set(7, 3, 0, c);
    m.boneParts[size_t(Bone::Torso)] = int16_t(lib.addPart(std::move(p.part)));
  }
  {
    Painter p = rock("golem_head", 7, 6, 7, {3.5f, 1.0f, 3.5f}, 32u, true);
    const int c = p.g(core);
    p.box(1, 2, 0, 3, 3, 1, c);                          // eyes
    p.box(4, 2, 0, 6, 3, 1, c);
    m.boneParts[size_t(Bone::Head)] = int16_t(lib.addPart(std::move(p.part)));
  }
  {
    Painter p = rock("golem_pelvis", 10, 3, 7, {5.0f, 1.0f, 3.5f}, 33u, false);
    m.boneParts[size_t(Bone::Pelvis)] = int16_t(lib.addPart(std::move(p.part)));
  }
  for (int side = 0; side < 2; ++side) {
    const bool r = side == 1;
    const std::string lr = r ? "_R" : "_L";
    const uint32_t seed = 40u + uint32_t(side) * 10u;
    Painter au = rock(("golem_arm_upper" + lr).c_str(), 5, 5, 5, {2.5f, 4.5f, 2.5f}, seed, true);
    m.boneParts[size_t(r ? Bone::ArmUpperR : Bone::ArmUpperL)] = int16_t(lib.addPart(std::move(au.part)));
    Painter al = rock(("golem_arm_lower" + lr).c_str(), 5, 4, 5, {2.5f, 3.5f, 2.5f}, seed + 1u, false);
    m.boneParts[size_t(r ? Bone::ArmLowerR : Bone::ArmLowerL)] = int16_t(lib.addPart(std::move(al.part)));
    Painter h = rock(("golem_fist" + lr).c_str(), 6, 4, 6, {3.0f, 3.0f, 3.0f}, seed + 2u, false);
    const int c = h.g(core);
    h.set(r ? 0 : 5, 2, 2, c);                           // crystal knuckle
    m.boneParts[size_t(r ? Bone::HandR : Bone::HandL)] = int16_t(lib.addPart(std::move(h.part)));
    Painter lu = rock(("golem_leg_upper" + lr).c_str(), 5, 4, 5, {2.5f, 4.0f, 2.5f}, seed + 3u, false);
    m.boneParts[size_t(r ? Bone::LegUpperR : Bone::LegUpperL)] = int16_t(lib.addPart(std::move(lu.part)));
    Painter ll = rock(("golem_leg_lower" + lr).c_str(), 5, 3, 5, {2.5f, 3.0f, 2.5f}, seed + 4u, false);
    m.boneParts[size_t(r ? Bone::LegLowerR : Bone::LegLowerL)] = int16_t(lib.addPart(std::move(ll.part)));
    Painter f = rock(("golem_foot" + lr).c_str(), 5, 2, 6, {2.5f, 2.0f, 4.0f}, seed + 5u, false);
    m.boneParts[size_t(r ? Bone::FootR : Bone::FootL)] = int16_t(lib.addPart(std::move(f.part)));
  }
  m.idleClip = "idle";
  m.moveClip = "walk";
  m.attackClip = "slam";
  m.hitClip = "hit_react";
  m.deathClip = "death";
  lib.addCreature(std::move(m));
}

} // namespace

void buildMonsterModels(ModelLibrary &lib) {
  buildSlime(lib);
  buildWolf(lib);
  buildGolem(lib);
}

} // namespace atm::model
