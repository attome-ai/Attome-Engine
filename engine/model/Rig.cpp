// Rigs: bind-pose bone offsets in model voxels (1 voxel = kModelVoxelScale
// blocks). Model space: +Y up, the character faces -Z, its right is +X,
// origin at the feet (ground contact) in the middle of the body.
//
// Humanoid (Trove proportions, 26 voxels tall):
//   feet 0..2, lower legs 2..5, upper legs 5..9, pelvis 9..11,
//   torso 11..17, head 17..26 (9^3 head), arms hang from shoulders at y 17
//   (upper 4, lower 3, hand 3 -> fingertips at y 7).
//
// Each bone's pivot is its joint; its voxel part is placed so the part's
// `pivot` sits on the joint. Bone order in the Bone enum is parent-first,
// which Animator::evaluate relies on (checked by the unit tests).

#include "Character.h"

namespace atm::model {

namespace {

std::array<RigBone, kBoneCount> humanoidBones() {
  return {{
    {"root", Bone::Root, {0.0f, 0.0f, 0.0f}},
    {"pelvis", Bone::Root, {0.0f, 9.0f, 0.0f}},
    {"torso", Bone::Pelvis, {0.0f, 2.0f, 0.0f}},
    {"head", Bone::Torso, {0.0f, 6.0f, 0.0f}},
    {"arm_upper_L", Bone::Torso, {-6.5f, 6.0f, 0.0f}},
    {"arm_lower_L", Bone::ArmUpperL, {0.0f, -4.0f, 0.0f}},
    {"hand_L", Bone::ArmLowerL, {0.0f, -3.0f, 0.0f}},
    {"arm_upper_R", Bone::Torso, {6.5f, 6.0f, 0.0f}},
    {"arm_lower_R", Bone::ArmUpperR, {0.0f, -4.0f, 0.0f}},
    {"hand_R", Bone::ArmLowerR, {0.0f, -3.0f, 0.0f}},
    {"leg_upper_L", Bone::Pelvis, {-2.5f, 0.0f, 0.0f}},
    {"leg_lower_L", Bone::LegUpperL, {0.0f, -4.0f, 0.0f}},
    {"foot_L", Bone::LegLowerL, {0.0f, -3.0f, 0.0f}},
    {"leg_upper_R", Bone::Pelvis, {2.5f, 0.0f, 0.0f}},
    {"leg_lower_R", Bone::LegUpperR, {0.0f, -4.0f, 0.0f}},
    {"foot_R", Bone::LegLowerR, {0.0f, -3.0f, 0.0f}},
}};
}

// Quadruped (wolf): body along Z, head toward -Z. Legs 8 voxels long.
std::array<RigBone, kBoneCount> quadBones() {
  return {{
    {"root", Bone::Root, {0.0f, 0.0f, 0.0f}},
    {"hind_body", Bone::Root, {0.0f, 8.0f, 3.5f}},
    {"chest", Bone::Pelvis, {0.0f, 0.5f, -6.0f}},
    {"head", Bone::Torso, {0.0f, 2.5f, -3.5f}},
    {"front_leg_upper_L", Bone::Torso, {-2.0f, -1.0f, -0.5f}},
    {"front_leg_lower_L", Bone::ArmUpperL, {0.0f, -3.5f, 0.0f}},
    {"front_paw_L", Bone::ArmLowerL, {0.0f, -3.0f, 0.0f}},
    {"front_leg_upper_R", Bone::Torso, {2.0f, -1.0f, -0.5f}},
    {"front_leg_lower_R", Bone::ArmUpperR, {0.0f, -3.5f, 0.0f}},
    {"front_paw_R", Bone::ArmLowerR, {0.0f, -3.0f, 0.0f}},
    {"hind_leg_upper_L", Bone::Pelvis, {-2.0f, -0.5f, 1.0f}},
    {"hind_leg_lower_L", Bone::LegUpperL, {0.0f, -3.5f, 0.0f}},
    {"hind_paw_L", Bone::LegLowerL, {0.0f, -2.5f, 0.0f}},
    {"hind_leg_upper_R", Bone::Pelvis, {2.0f, -0.5f, 1.0f}},
    {"hind_leg_lower_R", Bone::LegUpperR, {0.0f, -3.5f, 0.0f}},
    {"hind_paw_R", Bone::LegLowerR, {0.0f, -2.5f, 0.0f}},
}};
}

// Slime: the body sits on Pelvis at the ground; other bones unused.
std::array<RigBone, kBoneCount> slimeBones() {
  return {{
    {"root", Bone::Root, {0.0f, 0.0f, 0.0f}},
    {"body", Bone::Root, {0.0f, 0.0f, 0.0f}},
    {"top", Bone::Pelvis, {0.0f, 10.0f, 0.0f}},
    {"eyes", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused4", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused5", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused6", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused7", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused8", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused9", Bone::Torso, {0.0f, 0.0f, 0.0f}},
    {"unused10", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
    {"unused11", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
    {"unused12", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
    {"unused13", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
    {"unused14", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
    {"unused15", Bone::Pelvis, {0.0f, 0.0f, 0.0f}},
}};
}

} // namespace

const Rig &Rig::humanoid() {
  static const Rig rig = [] {
    Rig r{};
    r.bones = humanoidBones();
    r.socketBone = {Bone::HandR, Bone::HandL, Bone::Torso, Bone::Head};
    // MainHand/OffHand: centre of the fist (hand part is 3 voxels, pivot on
    // top). Back: centre of the upper back surface. Head: top of the head,
    // hats/helmets/hair are authored with their pivot at the crown centre.
    r.socketOffset = {glm::vec3(0.0f, -1.5f, 0.0f), glm::vec3(0.0f, -1.5f, 0.0f),
                      glm::vec3(0.0f, 5.0f, 3.5f), glm::vec3(0.0f, 9.0f, 0.0f)};
    return r;
  }();
  return rig;
}

const Rig &Rig::quadruped() {
  static const Rig rig = [] {
    Rig r{};
    r.bones = quadBones();
    r.socketBone = {Bone::Head, Bone::Head, Bone::Pelvis, Bone::Head};
    r.socketOffset = {glm::vec3(0.0f, -1.0f, -5.0f), glm::vec3(0.0f, -1.0f, -5.0f),
                      glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 4.0f, 0.0f)};
    return r;
  }();
  return rig;
}

const Rig &Rig::slime() {
  static const Rig rig = [] {
    Rig r{};
    r.bones = slimeBones();
    r.socketBone = {Bone::Pelvis, Bone::Pelvis, Bone::Pelvis, Bone::Torso};
    r.socketOffset = {glm::vec3(0.0f, 5.0f, -6.0f), glm::vec3(0.0f, 5.0f, -6.0f),
                      glm::vec3(0.0f, 5.0f, 6.0f), glm::vec3(0.0f)};
    return r;
  }();
  return rig;
}

} // namespace atm::model
