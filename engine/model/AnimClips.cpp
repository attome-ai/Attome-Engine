// Procedural keyframed clips (demo content). Authored against the rigs in
// Rig.cpp. Conventions (model space, character faces -Z, right = +X):
//   limb hanging down, rotate +X  -> swings forward (toward -Z)
//   torso/head, rotate -X         -> leans/nods forward
//   left arm rotate -Z / right arm +Z -> raises the arm sideways
//   rotate +Y                     -> turns left (counter-clockwise from above)
// Key times below are fractions of the clip length. Looping clips repeat the
// first key at 1.0 so they wrap seamlessly.

#include "Character.h"

#include <initializer_list>

namespace atm::model {

namespace {

glm::quat euler(const glm::vec3 &deg) { return glm::quat(glm::radians(deg)); }

struct K {
  float t;            // 0..1 of the clip length
  glm::vec3 e;        // euler degrees (x, y, z)
  glm::vec3 o{0.0f};  // offset, voxels
};

constexpr uint32_t bit(Bone b) { return 1u << unsigned(b); }

constexpr uint32_t kUpperBody = bit(Bone::Torso) | bit(Bone::Head) | bit(Bone::ArmUpperL) |
                                bit(Bone::ArmLowerL) | bit(Bone::HandL) | bit(Bone::ArmUpperR) |
                                bit(Bone::ArmLowerR) | bit(Bone::HandR);

struct ClipBuilder {
  AnimClip c;
  ClipBuilder(const char *name, float length, bool loop) {
    c.name = name;
    c.length = length;
    c.loop = loop;
  }
  ClipBuilder &track(Bone b, std::initializer_list<K> keys) {
    auto &tr = c.tracks[size_t(b)];
    tr.clear();
    for (const K &k : keys)
      tr.push_back(BoneKey{k.t * c.length, euler(k.e), k.o});
    return *this;
  }
  ClipBuilder &mask(uint32_t m) {
    c.boneMask = m;
    return *this;
  }
  ClipBuilder &event(float t, uint8_t id) {
    c.events.emplace_back(t * c.length, id);
    return *this;
  }
};

// Biped gait: L leg forward at t=0, R leg forward at t=0.5; opposite arms.
AnimClip gait(const char *name, float length, float legSwing, float kneeBend, float armSwing,
              float elbow, float lean, float bob) {
  ClipBuilder b(name, length, true);
  const float s = legSwing, k = -kneeBend, a = armSwing;
  b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, -bob, 0}},
                         {0.25f, {0, 0, 0}, {0, bob * 0.6f, 0}},
                         {0.5f, {0, 0, 0}, {0, -bob, 0}},
                         {0.75f, {0, 0, 0}, {0, bob * 0.6f, 0}},
                         {1.0f, {0, 0, 0}, {0, -bob, 0}}});
  b.track(Bone::Torso, {{0.0f, {lean, 6, 0}}, {0.5f, {lean, -6, 0}}, {1.0f, {lean, 6, 0}}});
  b.track(Bone::Head, {{0.0f, {-lean * 0.6f, -4, 0}}, {0.5f, {-lean * 0.6f, 4, 0}}, {1.0f, {-lean * 0.6f, -4, 0}}});
  b.track(Bone::LegUpperL, {{0.0f, {s, 0, 0}}, {0.5f, {-s, 0, 0}}, {1.0f, {s, 0, 0}}});
  b.track(Bone::LegUpperR, {{0.0f, {-s, 0, 0}}, {0.5f, {s, 0, 0}}, {1.0f, {-s, 0, 0}}});
  b.track(Bone::LegLowerL, {{0.0f, {k * 0.1f, 0, 0}}, {0.25f, {k * 0.3f, 0, 0}}, {0.5f, {k * 0.6f, 0, 0}},
                            {0.75f, {k, 0, 0}}, {1.0f, {k * 0.1f, 0, 0}}});
  b.track(Bone::LegLowerR, {{0.0f, {k * 0.6f, 0, 0}}, {0.25f, {k, 0, 0}}, {0.5f, {k * 0.1f, 0, 0}},
                            {0.75f, {k * 0.3f, 0, 0}}, {1.0f, {k * 0.6f, 0, 0}}});
  b.track(Bone::FootL, {{0.0f, {-s * 0.4f, 0, 0}}, {0.5f, {s * 0.3f, 0, 0}}, {1.0f, {-s * 0.4f, 0, 0}}});
  b.track(Bone::FootR, {{0.0f, {s * 0.3f, 0, 0}}, {0.5f, {-s * 0.4f, 0, 0}}, {1.0f, {s * 0.3f, 0, 0}}});
  b.track(Bone::ArmUpperL, {{0.0f, {-a, 0, -5}}, {0.5f, {a, 0, -5}}, {1.0f, {-a, 0, -5}}});
  b.track(Bone::ArmUpperR, {{0.0f, {a, 0, 5}}, {0.5f, {-a, 0, 5}}, {1.0f, {a, 0, 5}}});
  b.track(Bone::ArmLowerL, {{0.0f, {elbow * 0.5f, 0, 0}}, {0.5f, {elbow, 0, 0}}, {1.0f, {elbow * 0.5f, 0, 0}}});
  b.track(Bone::ArmLowerR, {{0.0f, {elbow, 0, 0}}, {0.5f, {elbow * 0.5f, 0, 0}}, {1.0f, {elbow, 0, 0}}});
  b.event(0.0f, anim_event::Footstep).event(0.5f, anim_event::Footstep);
  return b.c;
}

// Quadruped gait: diagonal pairs (front L + hind R) move together.
AnimClip quadGait(const char *name, float length, float swing, float bob) {
  ClipBuilder b(name, length, true);
  const float s = swing;
  b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, -bob, 0}}, {0.25f, {0, 0, 0}, {0, bob, 0}},
                         {0.5f, {0, 0, 0}, {0, -bob, 0}}, {0.75f, {0, 0, 0}, {0, bob, 0}},
                         {1.0f, {0, 0, 0}, {0, -bob, 0}}});
  b.track(Bone::Torso, {{0.0f, {2, 3, 0}}, {0.5f, {-2, -3, 0}}, {1.0f, {2, 3, 0}}});
  b.track(Bone::Head, {{0.0f, {-3, 0, 0}}, {0.5f, {3, 0, 0}}, {1.0f, {-3, 0, 0}}});
  b.track(Bone::ArmUpperL, {{0.0f, {s, 0, 0}}, {0.5f, {-s, 0, 0}}, {1.0f, {s, 0, 0}}});
  b.track(Bone::LegUpperR, {{0.0f, {s, 0, 0}}, {0.5f, {-s, 0, 0}}, {1.0f, {s, 0, 0}}});
  b.track(Bone::ArmUpperR, {{0.0f, {-s, 0, 0}}, {0.5f, {s, 0, 0}}, {1.0f, {-s, 0, 0}}});
  b.track(Bone::LegUpperL, {{0.0f, {-s, 0, 0}}, {0.5f, {s, 0, 0}}, {1.0f, {-s, 0, 0}}});
  b.track(Bone::ArmLowerL, {{0.0f, {0, 0, 0}}, {0.75f, {-s, 0, 0}}, {1.0f, {0, 0, 0}}});
  b.track(Bone::ArmLowerR, {{0.0f, {0, 0, 0}}, {0.25f, {-s, 0, 0}}, {0.5f, {0, 0, 0}}, {1.0f, {0, 0, 0}}});
  b.track(Bone::LegLowerL, {{0.0f, {0, 0, 0}}, {0.25f, {s, 0, 0}}, {0.5f, {0, 0, 0}}, {1.0f, {0, 0, 0}}});
  b.track(Bone::LegLowerR, {{0.0f, {0, 0, 0}}, {0.75f, {s, 0, 0}}, {1.0f, {0, 0, 0}}});
  b.event(0.0f, anim_event::Footstep).event(0.5f, anim_event::Footstep);
  return b.c;
}

void humanoidClips(AnimLibrary &lib) {
  // --- locomotion (full body) ----------------------------------------------
  {
    ClipBuilder b("idle", 2.4f, true);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.5f, {0, 0, 0}, {0, -0.3f, 0}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.5f, {2, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.3f, {-2, 3, 0}}, {0.7f, {1, -3, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -4}}, {0.5f, {3, 0, -8}}, {1.0f, {0, 0, -4}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 4}}, {0.5f, {3, 0, 8}}, {1.0f, {0, 0, 4}}});
    b.track(Bone::ArmLowerL, {{0.0f, {8, 0, 0}}, {0.5f, {12, 0, 0}}, {1.0f, {8, 0, 0}}});
    b.track(Bone::ArmLowerR, {{0.0f, {8, 0, 0}}, {0.5f, {12, 0, 0}}, {1.0f, {8, 0, 0}}});
    b.track(Bone::LegLowerL, {{0.0f, {0, 0, 0}}, {0.5f, {-3, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {0, 0, 0}}, {0.5f, {-3, 0, 0}}, {1.0f, {0, 0, 0}}});
    lib.add(b.c);
  }
  lib.add(gait("walk", 0.8f, 26.0f, 40.0f, 22.0f, 12.0f, -3.0f, 0.3f));
  lib.add(gait("run", 0.56f, 42.0f, 75.0f, 48.0f, 65.0f, -10.0f, 0.6f));
  lib.add(gait("sprint", 0.44f, 52.0f, 95.0f, 62.0f, 80.0f, -18.0f, 0.8f));
  {
    // Take-off, then tuck and hold (non-looping: holds the last key).
    ClipBuilder b("jump", 0.45f, false);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, -1.0f, 0}}, {0.3f, {0, 0, 0}, {0, 0.3f, 0}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    b.track(Bone::Torso, {{0.0f, {-12, 0, 0}}, {0.3f, {6, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {6, 0, 0}}, {1.0f, {4, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {-30, 0, -10}}, {0.3f, {40, 0, -35}}, {1.0f, {20, 0, -40}}});
    b.track(Bone::ArmUpperR, {{0.0f, {-30, 0, 10}}, {0.3f, {40, 0, 35}}, {1.0f, {20, 0, 40}}});
    b.track(Bone::ArmLowerL, {{0.0f, {10, 0, 0}}, {1.0f, {35, 0, 0}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {1.0f, {35, 0, 0}}});
    b.track(Bone::LegUpperL, {{0.0f, {20, 0, 0}}, {0.3f, {55, 0, 0}}, {1.0f, {50, 0, 0}}});
    b.track(Bone::LegLowerL, {{0.0f, {-30, 0, 0}}, {0.3f, {-80, 0, 0}}, {1.0f, {-75, 0, 0}}});
    b.track(Bone::LegUpperR, {{0.0f, {20, 0, 0}}, {0.3f, {-10, 0, 0}}, {1.0f, {5, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-30, 0, 0}}, {0.3f, {-20, 0, 0}}, {1.0f, {-35, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Double jump: a quick forward flip-tuck of the whole body.
    ClipBuilder b("double_jump", 0.4f, false);
    // Keys 90 degrees apart so each slerp takes the forward (short) way round.
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.25f, {-90, 0, 0}, {0, 3, 0}},
                           {0.5f, {-180, 0, 0}, {0, 4, 0}}, {0.75f, {-270, 0, 0}, {0, 3, 0}},
                           {1.0f, {-360, 0, 0}, {0, 0, 0}}});
    b.track(Bone::LegUpperL, {{0.0f, {40, 0, 0}}, {0.5f, {90, 0, 0}}, {1.0f, {30, 0, 0}}});
    b.track(Bone::LegUpperR, {{0.0f, {40, 0, 0}}, {0.5f, {90, 0, 0}}, {1.0f, {10, 0, 0}}});
    b.track(Bone::LegLowerL, {{0.0f, {-60, 0, 0}}, {0.5f, {-110, 0, 0}}, {1.0f, {-50, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-60, 0, 0}}, {0.5f, {-110, 0, 0}}, {1.0f, {-30, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {30, 0, -30}}, {0.5f, {60, 0, -20}}, {1.0f, {20, 0, -40}}});
    b.track(Bone::ArmUpperR, {{0.0f, {30, 0, 30}}, {0.5f, {60, 0, 20}}, {1.0f, {20, 0, 40}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("fall", 0.6f, true);
    b.track(Bone::Torso, {{0.0f, {4, 0, 0}}, {0.5f, {6, 0, 0}}, {1.0f, {4, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {-8, 0, 0}}, {1.0f, {-8, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {10, 0, -75}}, {0.5f, {-5, 0, -95}}, {1.0f, {10, 0, -75}}});
    b.track(Bone::ArmUpperR, {{0.0f, {-5, 0, 95}}, {0.5f, {10, 0, 75}}, {1.0f, {-5, 0, 95}}});
    b.track(Bone::ArmLowerL, {{0.0f, {0, 0, -15}}, {1.0f, {0, 0, -15}}});
    b.track(Bone::ArmLowerR, {{0.0f, {0, 0, 15}}, {1.0f, {0, 0, 15}}});
    b.track(Bone::LegUpperL, {{0.0f, {25, 0, -4}}, {0.5f, {10, 0, -4}}, {1.0f, {25, 0, -4}}});
    b.track(Bone::LegUpperR, {{0.0f, {5, 0, 4}}, {0.5f, {22, 0, 4}}, {1.0f, {5, 0, 4}}});
    b.track(Bone::LegLowerL, {{0.0f, {-35, 0, 0}}, {0.5f, {-20, 0, 0}}, {1.0f, {-35, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-15, 0, 0}}, {0.5f, {-40, 0, 0}}, {1.0f, {-15, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Arms spread wide, body leaning into the glide, legs trailing.
    ClipBuilder b("glide", 1.6f, true);
    b.track(Bone::Pelvis, {{0.0f, {-30, 0, 0}, {0, 0, 0}}, {0.5f, {-30, 0, 3}, {0, 0.4f, 0}}, {1.0f, {-30, 0, 0}, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {28, 0, 0}}, {1.0f, {28, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -88}}, {0.5f, {6, 0, -82}}, {1.0f, {0, 0, -88}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 88}}, {0.5f, {6, 0, 82}}, {1.0f, {0, 0, 88}}});
    b.track(Bone::ArmLowerL, {{0.0f, {0, 0, 5}}, {0.5f, {0, 0, -5}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {0, 0, -5}}, {0.5f, {0, 0, 5}}, {1.0f, {0, 0, -5}}});
    b.track(Bone::LegUpperL, {{0.0f, {-8, 0, -6}}, {0.5f, {-2, 0, -6}}, {1.0f, {-8, 0, -6}}});
    b.track(Bone::LegUpperR, {{0.0f, {-2, 0, 6}}, {0.5f, {-8, 0, 6}}, {1.0f, {-2, 0, 6}}});
    b.track(Bone::LegLowerL, {{0.0f, {-20, 0, 0}}, {1.0f, {-20, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-12, 0, 0}}, {1.0f, {-12, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Breaststroke with a flutter kick; body tipped forward.
    ClipBuilder b("swim", 1.2f, true);
    b.track(Bone::Pelvis, {{0.0f, {-55, 0, 0}, {0, 2, 0}}, {0.5f, {-60, 0, 0}, {0, 2.4f, 0}}, {1.0f, {-55, 0, 0}, {0, 2, 0}}});
    b.track(Bone::Head, {{0.0f, {40, 0, 0}}, {1.0f, {40, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {165, 0, -10}}, {0.4f, {80, 0, -80}}, {0.7f, {40, 0, -30}}, {1.0f, {165, 0, -10}}});
    b.track(Bone::ArmUpperR, {{0.0f, {165, 0, 10}}, {0.4f, {80, 0, 80}}, {0.7f, {40, 0, 30}}, {1.0f, {165, 0, 10}}});
    b.track(Bone::ArmLowerL, {{0.0f, {0, 0, 0}}, {0.7f, {70, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmLowerR, {{0.0f, {0, 0, 0}}, {0.7f, {70, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::LegUpperL, {{0.0f, {15, 0, 0}}, {0.25f, {-15, 0, 0}}, {0.5f, {15, 0, 0}}, {0.75f, {-15, 0, 0}}, {1.0f, {15, 0, 0}}});
    b.track(Bone::LegUpperR, {{0.0f, {-15, 0, 0}}, {0.25f, {15, 0, 0}}, {0.5f, {-15, 0, 0}}, {0.75f, {15, 0, 0}}, {1.0f, {-15, 0, 0}}});
    b.track(Bone::LegLowerL, {{0.0f, {-10, 0, 0}}, {1.0f, {-10, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-10, 0, 0}}, {1.0f, {-10, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Low forward dash pose (played as locomotion while the dash lasts).
    ClipBuilder b("dash", 0.25f, false);
    b.track(Bone::Pelvis, {{0.0f, {-10, 0, 0}, {0, -0.5f, 0}}, {0.3f, {-28, 0, 0}, {0, -1.0f, 0}}, {1.0f, {-25, 0, 0}, {0, -0.8f, 0}}});
    b.track(Bone::Head, {{0.0f, {10, 0, 0}}, {1.0f, {22, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {-20, 0, -10}}, {0.3f, {-65, 0, -20}}, {1.0f, {-60, 0, -20}}});
    b.track(Bone::ArmUpperR, {{0.0f, {-20, 0, 10}}, {0.3f, {-65, 0, 20}}, {1.0f, {-60, 0, 20}}});
    b.track(Bone::LegUpperL, {{0.0f, {20, 0, 0}}, {0.3f, {55, 0, 0}}, {1.0f, {50, 0, 0}}});
    b.track(Bone::LegLowerL, {{0.0f, {-20, 0, 0}}, {0.3f, {-60, 0, 0}}, {1.0f, {-55, 0, 0}}});
    b.track(Bone::LegUpperR, {{0.0f, {-10, 0, 0}}, {0.3f, {-40, 0, 0}}, {1.0f, {-35, 0, 0}}});
    b.track(Bone::LegLowerR, {{0.0f, {-10, 0, 0}}, {0.3f, {-30, 0, 0}}, {1.0f, {-25, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Fall backward and lie flat (holds the last key).
    ClipBuilder b("death", 1.1f, false);
    b.track(Bone::Root, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.25f, {12, 0, 0}, {0, 0, 0}},
                         {0.7f, {90, 0, 0}, {0, 3.0f, 0}}, {0.8f, {84, 0, 0}, {0, 3.4f, 0}},
                         {1.0f, {90, 0, 0}, {0, 3.0f, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.25f, {20, 0, 0}}, {1.0f, {-10, 25, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, 0}}, {0.25f, {60, 0, -30}}, {1.0f, {20, 0, -70}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 0}}, {0.25f, {60, 0, 30}}, {1.0f, {20, 0, 75}}});
    b.track(Bone::LegUpperL, {{0.0f, {0, 0, 0}}, {0.25f, {20, 0, 0}}, {1.0f, {15, 0, -10}}});
    b.track(Bone::LegUpperR, {{0.0f, {0, 0, 0}}, {1.0f, {5, 0, 12}}});
    b.track(Bone::LegLowerL, {{0.0f, {0, 0, 0}}, {1.0f, {-25, 0, 0}}});
    lib.add(b.c);
  }

  // --- actions (upper body) ------------------------------------------------
  {
    ClipBuilder b("mine", 0.65f, false);
    b.mask(kUpperBody).event(0.5f, anim_event::Impact);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.35f, {10, -12, 0}}, {0.5f, {-18, 5, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.35f, {170, 0, 10}}, {0.5f, {45, 0, 5}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {0.35f, {35, 0, 0}}, {0.5f, {10, 0, 0}}, {1.0f, {10, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.35f, {150, 0, 15}}, {0.5f, {40, 0, 15}}, {1.0f, {0, 0, -5}}});
    b.track(Bone::ArmLowerL, {{0.0f, {10, 0, 0}}, {0.35f, {40, 0, 0}}, {0.5f, {15, 0, 0}}, {1.0f, {10, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("place", 0.35f, false);
    b.mask(kUpperBody).event(0.4f, anim_event::Impact);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.4f, {-8, 6, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.4f, {75, 0, 0}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {0.4f, {25, 0, 0}}, {1.0f, {10, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Wind up over the right shoulder, snap across, recover.
    ClipBuilder b("sword_swing", 0.5f, false);
    b.mask(kUpperBody).event(0.42f, anim_event::Hit);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.25f, {4, -30, 0}}, {0.45f, {-10, 35, 0}}, {0.7f, {-6, 25, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.25f, {0, 20, 0}}, {0.45f, {0, -25, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.25f, {125, 0, 50}}, {0.45f, {70, 0, -40}}, {0.7f, {55, 0, -35}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {0.25f, {50, 0, 0}}, {0.45f, {5, 0, 0}}, {1.0f, {10, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.25f, {-20, 0, -30}}, {0.45f, {30, 0, -20}}, {1.0f, {0, 0, -5}}});
    lib.add(b.c);
  }
  {
    // Bow held in the right hand, left hand pulls the string back.
    ClipBuilder b("bow_draw", 0.55f, false);
    b.mask(kUpperBody);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.6f, {0, 25, 0}}, {1.0f, {0, 28, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.6f, {0, -22, 0}}, {1.0f, {0, -25, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.5f, {88, 0, 0}}, {1.0f, {90, 0, 0}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.5f, {85, -10, 30}}, {1.0f, {88, -15, 45}}});
    b.track(Bone::ArmLowerL, {{0.0f, {10, 0, 0}}, {0.5f, {20, 0, 60}}, {1.0f, {25, 0, 95}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("bow_shoot", 0.4f, false);
    b.mask(kUpperBody).event(0.05f, anim_event::Release);
    b.track(Bone::Torso, {{0.0f, {0, 28, 0}}, {0.2f, {4, 28, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, -25, 0}}, {0.5f, {0, -20, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {90, 0, 0}}, {0.15f, {98, 0, 0}}, {0.6f, {80, 0, 0}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmUpperL, {{0.0f, {88, -15, 45}}, {0.12f, {80, -25, 70}}, {0.6f, {60, 0, 20}}, {1.0f, {0, 0, -5}}});
    b.track(Bone::ArmLowerL, {{0.0f, {25, 0, 95}}, {0.12f, {10, 0, 20}}, {1.0f, {10, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Staff raised, then thrust forward to release the spell.
    ClipBuilder b("staff_cast", 0.6f, false);
    b.mask(kUpperBody).event(0.55f, anim_event::Release);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.35f, {10, -10, 0}}, {0.55f, {-14, 8, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.35f, {160, 0, 15}}, {0.55f, {85, 0, 0}}, {0.8f, {80, 0, 0}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {0.35f, {20, 0, 0}}, {0.55f, {0, 0, 0}}, {1.0f, {10, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.35f, {40, 0, -45}}, {0.55f, {70, 0, -10}}, {1.0f, {0, 0, -5}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("punch", 0.4f, false);
    b.mask(kUpperBody).event(0.3f, anim_event::Hit);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.15f, {0, -20, 0}}, {0.3f, {-8, 25, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.15f, {40, 0, 20}}, {0.3f, {90, 0, 0}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {10, 0, 0}}, {0.15f, {100, 0, 0}}, {0.3f, {0, 0, 0}}, {1.0f, {10, 0, 0}}});
    lib.add(b.c);
  }
  {
    // Two-handed overhead slam (golem).
    ClipBuilder b("slam", 0.9f, false);
    b.mask(kUpperBody).event(0.6f, anim_event::Hit);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.45f, {15, 0, 0}}, {0.6f, {-25, 0, 0}}, {0.8f, {-20, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.45f, {170, 0, 10}}, {0.6f, {70, 0, 5}}, {0.8f, {60, 0, 5}}, {1.0f, {0, 0, -5}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.45f, {170, 0, -10}}, {0.6f, {70, 0, -5}}, {0.8f, {60, 0, -5}}, {1.0f, {0, 0, 5}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("hit_react", 0.35f, false);
    b.mask(kUpperBody);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.25f, {18, 6, 4}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.25f, {16, -8, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, -5}}, {0.25f, {-15, 0, -35}}, {1.0f, {0, 0, -5}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.25f, {-15, 0, 35}}, {1.0f, {0, 0, 5}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("wave", 1.3f, false);
    b.mask(kUpperBody);
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.2f, {-4, 0, 8}}, {0.8f, {-4, 0, 8}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 5}}, {0.15f, {0, 0, 150}}, {0.85f, {0, 0, 150}}, {1.0f, {0, 0, 5}}});
    b.track(Bone::ArmLowerR, {{0.0f, {0, 0, 0}}, {0.15f, {0, 0, 25}}, {0.3f, {0, 0, -25}}, {0.45f, {0, 0, 25}},
                              {0.6f, {0, 0, -25}}, {0.75f, {0, 0, 25}}, {0.9f, {0, 0, 0}}, {1.0f, {0, 0, 0}}});
    lib.add(b.c);
  }
}

void quadrupedClips(AnimLibrary &lib) {
  {
    ClipBuilder b("quad_idle", 2.0f, true);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.5f, {-2, 0, 0}, {0, 0.25f, 0}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.3f, {-5, 12, 0}}, {0.6f, {3, -10, 0}}, {1.0f, {0, 0, 0}}});
    lib.add(b.c);
  }
  lib.add(quadGait("quad_walk", 0.55f, 32.0f, 0.4f));
  {
    // Crouch, lunge and bite.
    ClipBuilder b("quad_attack", 0.6f, false);
    b.event(0.45f, anim_event::Hit);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.3f, {6, 0, 0}, {0, -1.2f, 1.0f}}, {0.45f, {-8, 0, 0}, {0, 0.6f, -3.0f}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.3f, {12, 0, 0}}, {0.45f, {-18, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperL, {{0.0f, {0, 0, 0}}, {0.3f, {-20, 0, 0}}, {0.45f, {45, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::ArmUpperR, {{0.0f, {0, 0, 0}}, {0.3f, {-20, 0, 0}}, {0.45f, {45, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::LegUpperL, {{0.0f, {0, 0, 0}}, {0.3f, {25, 0, 0}}, {0.45f, {-35, 0, 0}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::LegUpperR, {{0.0f, {0, 0, 0}}, {0.3f, {25, 0, 0}}, {0.45f, {-35, 0, 0}}, {1.0f, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("quad_hit", 0.3f, false);
    b.track(Bone::Torso, {{0.0f, {0, 0, 0}}, {0.3f, {10, 0, 8}}, {1.0f, {0, 0, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {0.3f, {15, 10, 0}}, {1.0f, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("quad_death", 0.9f, false);
    b.track(Bone::Root, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.6f, {0, 0, 90}, {0, 2.5f, 0}}, {0.75f, {0, 0, 84}, {0, 2.8f, 0}}, {1.0f, {0, 0, 90}, {0, 2.5f, 0}}});
    b.track(Bone::Head, {{0.0f, {0, 0, 0}}, {1.0f, {20, 0, 0}}});
    lib.add(b.c);
  }
}

void slimeClips(AnimLibrary &lib) {
  {
    ClipBuilder b("slime_idle", 1.4f, true);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.25f, {0, 0, 3}, {0, 0.4f, 0}},
                           {0.5f, {0, 0, 0}, {0, 0, 0}}, {0.75f, {0, 0, -3}, {0, 0.4f, 0}},
                           {1.0f, {0, 0, 0}, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("slime_hop", 0.7f, true);
    b.event(0.0f, anim_event::Footstep);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.15f, {6, 0, 0}, {0, -0.5f, 0}},
                           {0.5f, {-10, 0, 0}, {0, 5.0f, 0}}, {0.85f, {4, 0, 0}, {0, 0.5f, 0}},
                           {1.0f, {0, 0, 0}, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("slime_attack", 0.55f, false);
    b.event(0.55f, anim_event::Hit);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.25f, {10, 0, 0}, {0, -0.8f, 1.0f}},
                           {0.55f, {-20, 0, 0}, {0, 3.5f, -5.0f}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("slime_hit", 0.3f, false);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.3f, {12, 0, 6}, {0, 0, 1.0f}}, {1.0f, {0, 0, 0}, {0, 0, 0}}});
    lib.add(b.c);
  }
  {
    ClipBuilder b("slime_death", 0.8f, false);
    b.track(Bone::Pelvis, {{0.0f, {0, 0, 0}, {0, 0, 0}}, {0.3f, {0, 20, 0}, {0, 1.0f, 0}}, {1.0f, {0, 45, 0}, {0, -9.0f, 0}}});
    lib.add(b.c);
  }
}

} // namespace

void AnimLibrary::buildDefaults() {
  clips_.clear();
  humanoidClips(*this);
  quadrupedClips(*this);
  slimeClips(*this);
}

} // namespace atm::model
