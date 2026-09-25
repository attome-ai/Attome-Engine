// Drawing modular voxel characters, monsters, dropped items and projectiles
// (GAME_DESIGN §10.1): the shared rig is evaluated once per character, and
// every body/equipment part is one instanced renderer draw placed at its
// bone (or socket) transform.

#include "App.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cmath>

namespace ao::client {

namespace {

using atm::model::Bone;
using atm::model::kBoneCount;
using atm::model::Socket;

struct ClipIds {
  int idle = -1, walk = -1, run = -1, sprint = -1, jump = -1, fall = -1, glide = -1,
      swim = -1, dash = -1, death = -1;
  int mine = -1, place = -1, swordSwing = -1, bowShoot = -1, staffCast = -1, punch = -1,
      hitReact = -1, wave = -1;
  bool ready = false;
};

ClipIds &clips(const atm::model::AnimLibrary &lib) {
  static ClipIds ids;
  if (!ids.ready) {
    ids.idle = lib.indexOf("idle");
    ids.walk = lib.indexOf("walk");
    ids.run = lib.indexOf("run");
    ids.sprint = lib.indexOf("sprint");
    ids.jump = lib.indexOf("jump");
    ids.fall = lib.indexOf("fall");
    ids.glide = lib.indexOf("glide");
    ids.swim = lib.indexOf("swim");
    ids.dash = lib.indexOf("dash");
    ids.death = lib.indexOf("death");
    ids.mine = lib.indexOf("mine");
    ids.place = lib.indexOf("place");
    ids.swordSwing = lib.indexOf("sword_swing");
    ids.bowShoot = lib.indexOf("bow_shoot");
    ids.staffCast = lib.indexOf("staff_cast");
    ids.punch = lib.indexOf("punch");
    ids.hitReact = lib.indexOf("hit_react");
    ids.wave = lib.indexOf("wave");
    ids.ready = true;
  }
  return ids;
}

constexpr uint8_t kFlagGliding = 1, kFlagWater = 2, kFlagDead = 4, kFlagGround = 8;

// Places one part: world = feet + yawRot * scale * (boneTransform * (v - pivot)).
atm::render::ModelInstance partInstance(atm::render::ModelMeshId mesh, const glm::dvec3 &feet,
                                        const glm::quat &yawRot, const glm::mat4 &bone,
                                        const glm::vec3 &pivot, float scale, uint32_t tint,
                                        uint16_t paletteOffset) {
  atm::render::ModelInstance inst;
  inst.mesh = mesh;
  const glm::quat boneRot = glm::quat_cast(glm::mat3(bone));
  const glm::vec3 boneT = glm::vec3(bone[3]);
  inst.rotation = yawRot * boneRot;
  inst.origin = feet + glm::dvec3(yawRot * (boneT * scale));
  inst.pivot = pivot;
  inst.voxelScale = scale;
  inst.tint = tint;
  inst.paletteOffset = paletteOffset;
  return inst;
}

} // namespace

// ---------------------------------------------------------------------------
// Animation selection
// ---------------------------------------------------------------------------

void App::selectLocomotion(atm::model::Animator &anim, const glm::vec3 &vel, uint8_t flags,
                           bool isLocal, int creature) {
  const float speed = std::sqrt(vel.x * vel.x + vel.z * vel.z);

  if (creature >= 0) {
    const auto &c = models_.creature(creature);
    const bool moving = speed > 0.4f;
    const int want = anims_.indexOf((flags & kFlagDead) ? c.deathClip
                                    : moving           ? c.moveClip
                                                       : c.idleClip);
    if (want >= 0 && anim.locomotionClip() != want)
      anim.setLocomotion(want);
    return;
  }

  const ClipIds &ids = clips(anims_);
  int want = ids.idle;
  if (flags & kFlagDead)
    want = ids.death;
  else if (flags & kFlagWater)
    want = ids.swim;
  else if (flags & kFlagGliding)
    want = ids.glide;
  else if (!(flags & kFlagGround))
    want = vel.y > 0.5f ? ids.jump : ids.fall;
  else if (speed > 8.0f)
    want = ids.sprint;
  else if (speed > 3.5f)
    want = ids.run;
  else if (speed > 0.3f)
    want = ids.walk;
  (void)isLocal;
  if (want >= 0 && anim.locomotionClip() != want)
    anim.setLocomotion(want);
}

void App::playActionAnim(atm::model::Animator &anim, uint8_t action, WeaponType weapon,
                         int creature) {
  if (creature >= 0) {
    const auto &c = models_.creature(creature);
    const std::string *name = nullptr;
    switch (action) {
    case ao::action::Swing:
    case ao::action::BowShoot:
    case ao::action::Cast: name = &c.attackClip; break;
    case ao::action::Hit: name = &c.hitClip; break;
    case ao::action::Death: name = &c.deathClip; break;
    default: break;
    }
    if (name) {
      const int clip = anims_.indexOf(*name);
      if (clip >= 0) {
        if (action == ao::action::Death)
          anim.setLocomotion(clip, 0.05f);
        else
          anim.playAction(clip);
      }
    }
    return;
  }

  const ClipIds &ids = clips(anims_);
  int clip = -1;
  switch (action) {
  case ao::action::Swing:
    clip = weapon == WeaponType::None ? ids.punch : ids.swordSwing;
    break;
  case ao::action::BowShoot: clip = ids.bowShoot; break;
  case ao::action::Cast: clip = ids.staffCast; break;
  case ao::action::Mine: clip = ids.mine; break;
  case ao::action::Place: clip = ids.place; break;
  case ao::action::Hit: clip = ids.hitReact; break;
  case ao::action::Wave: clip = ids.wave; break;
  case ao::action::Death:
    if (ids.death >= 0)
      anim.setLocomotion(ids.death, 0.05f);
    return;
  default: break;
  }
  if (clip >= 0)
    anim.playAction(clip);
}

int App::creatureForMonster(uint8_t type) {
  if (size_t(type) >= monsterCreature_.size()) {
    const size_t old = monsterCreature_.size();
    monsterCreature_.resize(size_t(type) + 1, -2);
    for (size_t i = old; i < monsterCreature_.size(); ++i)
      monsterCreature_[i] = -2;
  }
  int &c = monsterCreature_[type];
  if (c == -2) {
    const MonsterDef &def = monsterDef(type);
    c = def.model ? models_.findCreature(def.model) : -1;
  }
  return c;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

void App::drawCharacter(const atm::model::Appearance &appearance,
                        const atm::model::Animator &animator, const glm::dvec3 &feet, float yaw,
                        uint32_t tint) {
  const atm::model::Rig &rig = atm::model::Rig::humanoid();
  std::array<glm::mat4, kBoneCount> bones;
  animator.evaluate(anims_, rig, bones);

  const atm::model::ResolvedModel resolved = atm::model::resolveAppearance(models_, appearance);
  const glm::quat yawRot = glm::angleAxis(yaw, glm::vec3(0, 1, 0));
  const float scale = atm::model::kModelVoxelScale;
  const auto &parts = models_.parts();

  for (int b = 0; b < kBoneCount; ++b) {
    const int part = resolved.boneParts[size_t(b)];
    if (part < 0 || size_t(part) >= partMeshes_.size() ||
        partMeshes_[size_t(part)] == atm::render::kInvalidModelMesh)
      continue;
    renderer_.drawModel(partInstance(partMeshes_[size_t(part)], feet, yawRot, bones[size_t(b)],
                                     parts[size_t(part)].pivot, scale, tint,
                                     resolved.bonePaletteOffset[size_t(b)]));
  }

  for (int s = 0; s < int(Socket::Count); ++s) {
    const int part = resolved.socketParts[size_t(s)];
    if (part < 0 || size_t(part) >= partMeshes_.size() ||
        partMeshes_[size_t(part)] == atm::render::kInvalidModelMesh)
      continue;
    const Bone bone = rig.socketBone[size_t(s)];
    const glm::mat4 socket =
        glm::translate(bones[size_t(bone)], rig.socketOffset[size_t(s)]);
    renderer_.drawModel(partInstance(partMeshes_[size_t(part)], feet, yawRot, socket,
                                     parts[size_t(part)].pivot, scale, tint,
                                     resolved.socketPaletteOffset[size_t(s)]));
  }
}

void App::drawMonster(uint8_t type, const atm::model::Animator &animator,
                      const glm::dvec3 &feet, float yaw, uint32_t tint) {
  const int ci = creatureForMonster(type);
  if (ci < 0)
    return;
  const atm::model::CreatureModel &c = models_.creature(ci);
  if (!c.rig)
    return;
  std::array<glm::mat4, kBoneCount> bones;
  animator.evaluate(anims_, *c.rig, bones);
  const glm::quat yawRot = glm::angleAxis(yaw, glm::vec3(0, 1, 0));
  const float scale = atm::model::kModelVoxelScale * c.scale;
  const auto &parts = models_.parts();
  for (int b = 0; b < kBoneCount; ++b) {
    const int part = c.boneParts[size_t(b)];
    if (part < 0 || size_t(part) >= partMeshes_.size() ||
        partMeshes_[size_t(part)] == atm::render::kInvalidModelMesh)
      continue;
    renderer_.drawModel(partInstance(partMeshes_[size_t(part)], feet, yawRot, bones[size_t(b)],
                                     parts[size_t(part)].pivot, scale, tint, 0));
  }
}

void App::drawItemEntity(uint16_t item, const glm::dvec3 &pos, float spin) {
  const ItemDef &d = itemDef(item);
  atm::render::ModelInstance inst;
  inst.origin = pos + glm::dvec3(0.0, 0.35 + 0.08 * std::sin(double(spin) * 2.0), 0.0);
  inst.rotation = glm::angleAxis(spin, glm::vec3(0, 1, 0));
  inst.voxelScale = 1.0f / 12.0f;
  if (d.kind == ItemKind::Block && d.placesBlock < blockItemMeshes_.size() &&
      blockItemMeshes_[d.placesBlock] != atm::render::kInvalidModelMesh) {
    inst.mesh = blockItemMeshes_[d.placesBlock];
    inst.pivot = {2.0f, 2.0f, 2.0f};
    renderer_.drawModel(inst);
    return;
  }
  // Equipment and resources: draw the item's model piece if it has one.
  if (d.piece) {
    const atm::model::PieceId pid = models_.findPiece(d.piece);
    const atm::model::EquipPiece &piece = models_.piece(pid);
    int part = piece.socketPart;
    if (part < 0) {
      for (int16_t p : piece.boneParts)
        if (p >= 0) { part = p; break; }
    }
    if (part >= 0 && size_t(part) < partMeshes_.size() &&
        partMeshes_[size_t(part)] != atm::render::kInvalidModelMesh) {
      inst.mesh = partMeshes_[size_t(part)];
      inst.pivot = models_.parts()[size_t(part)].pivot;
      renderer_.drawModel(inst);
      return;
    }
  }
  // Fallback: a small stone-coloured cube.
  if (atm::voxel::blocks::Stone < blockItemMeshes_.size() &&
      blockItemMeshes_[atm::voxel::blocks::Stone] != atm::render::kInvalidModelMesh) {
    inst.mesh = blockItemMeshes_[atm::voxel::blocks::Stone];
    inst.pivot = {2.0f, 2.0f, 2.0f};
    inst.voxelScale = 1.0f / 20.0f;
    renderer_.drawModel(inst);
  }
}

void App::drawProjectile(const glm::dvec3 &pos, const glm::vec3 &vel) {
  if (arrowMesh_ == atm::render::kInvalidModelMesh)
    return;
  atm::render::ModelInstance inst;
  inst.mesh = arrowMesh_;
  inst.origin = pos;
  const float len = glm::length(vel);
  // The arrow part points along -Z (tip at z = 0).
  if (len > 1e-3f) {
    const glm::vec3 dir = vel / len;
    const float yaw = std::atan2(-dir.x, -dir.z);
    const float pitch = std::asin(glm::clamp(dir.y, -1.0f, 1.0f));
    inst.rotation = glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * glm::angleAxis(pitch, glm::vec3(1, 0, 0));
  }
  inst.pivot = {0.5f, 0.5f, 6.0f};
  inst.voxelScale = 1.0f / 16.0f;
  renderer_.drawModel(inst);
}

} // namespace ao::client
