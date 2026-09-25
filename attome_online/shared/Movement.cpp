// Trove-style movement (GAME_DESIGN §7), shared by client prediction and the
// server. Deterministic: no randomness, no clocks; only the inputs, the world
// and the tunables decide the result. (Floating-point results can still
// differ across compilers/CPUs; the server stays authoritative.)
//
// Conventions (lead contract): +Y up, yaw 0 faces -Z, positive yaw turns
// counter-clockwise seen from above.
//   forward(yaw) = (-sin yaw, 0, -cos yaw), right(yaw) = (cos yaw, 0, -sin yaw)
//   look(yaw, pitch) = (-sin yaw cos pitch, sin pitch, -cos yaw cos pitch)

#include "Movement.h"

#include "../../engine/ATMConfig.h"
#include "../../engine/voxel/BlockRegistry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ao {

// Editable in the tunables JSON under "movement" (see ATMConfig.h).
namespace movement_tunables {
ATM_TUNABLE_SECTION("movement");
ATM_TUNABLE(float, walkSpeed, 6.0f);
ATM_TUNABLE(float, sprintSpeed, 9.0f);
ATM_TUNABLE(float, accelGround, 60.0f);
ATM_TUNABLE(float, accelAir, 18.0f);
ATM_TUNABLE(float, friction, 12.0f);
ATM_TUNABLE(float, gravity, 30.0f);
ATM_TUNABLE(float, jumpSpeed, 10.5f);
ATM_TUNABLE(float, maxFall, 50.0f);
ATM_TUNABLE(int, extraJumps, 1);
ATM_TUNABLE(float, glideFallSpeed, 2.5f);
ATM_TUNABLE(float, glideSpeed, 11.0f);
ATM_TUNABLE(float, glideStaminaMax, 3.0f);
ATM_TUNABLE(float, dashSpeed, 22.0f);
ATM_TUNABLE(float, dashTime, 0.18f);
ATM_TUNABLE(float, dashCooldown, 0.9f);
ATM_TUNABLE(float, swimSpeed, 4.0f);
ATM_TUNABLE(float, halfWidth, 0.3f);
ATM_TUNABLE(float, height, 1.8f);
ATM_TUNABLE(float, eyeHeight, 1.6f);
ATM_TUNABLE(float, stepHeight, 1.05f);
} // namespace movement_tunables

const MoveTuning &moveTuning() {
  // Refreshed on every call so JSON reloads apply; callers take a copy once
  // per step (stepMovement) so a step always sees consistent values.
  // thread_local: the client's main thread and the in-process (--local)
  // server thread both call this; a shared static would be a data race.
  thread_local MoveTuning t;
  namespace mt = movement_tunables;
  t.walkSpeed = mt::walkSpeed;
  t.sprintSpeed = mt::sprintSpeed;
  t.accelGround = mt::accelGround;
  t.accelAir = mt::accelAir;
  t.friction = mt::friction;
  t.gravity = mt::gravity;
  t.jumpSpeed = mt::jumpSpeed;
  t.maxFall = mt::maxFall;
  t.extraJumps = mt::extraJumps;
  t.glideFallSpeed = mt::glideFallSpeed;
  t.glideSpeed = mt::glideSpeed;
  t.glideStaminaMax = mt::glideStaminaMax;
  t.dashSpeed = mt::dashSpeed;
  t.dashTime = mt::dashTime;
  t.dashCooldown = mt::dashCooldown;
  t.swimSpeed = mt::swimSpeed;
  t.halfWidth = mt::halfWidth;
  t.height = mt::height;
  t.eyeHeight = mt::eyeHeight;
  t.stepHeight = mt::stepHeight;
  return t;
}

namespace {

using atm::voxel::BlockRegistry;
using atm::voxel::IBlockAccess;
using atm::voxel::kWorldHeight;

constexpr double kEps = 1e-4;

int32_t floorToInt(double v) { return int32_t(std::floor(v)); }

struct Collider {
  const IBlockAccess &world;
  const BlockRegistry &blocks;
  double hw, h;

  // Unloaded chunks and everything below the world are solid; above the
  // world is open sky.
  bool solidAt(int32_t x, int32_t y, int32_t z) const {
    if (y < 0)
      return true;
    if (y >= kWorldHeight)
      return false;
    const BlockPos p{x, y, z};
    if (!world.isLoaded(atm::voxel::chunkOf(p)))
      return true;
    return blocks.solid(world.blockAt(p));
  }

  bool boxFree(const glm::dvec3 &pos) const {
    const int32_t x0 = floorToInt(pos.x - hw + kEps), x1 = floorToInt(pos.x + hw - kEps);
    const int32_t y0 = floorToInt(pos.y + kEps), y1 = floorToInt(pos.y + h - kEps);
    const int32_t z0 = floorToInt(pos.z - hw + kEps), z1 = floorToInt(pos.z + hw - kEps);
    for (int32_t y = y0; y <= y1; ++y)
      for (int32_t z = z0; z <= z1; ++z)
        for (int32_t x = x0; x <= x1; ++x)
          if (solidAt(x, y, z))
            return false;
    return true;
  }

  // Moves `pos` along one axis by up to `d`, stopping flush against the first
  // solid block layer. Blocks the box already overlaps are ignored (so a
  // player stuck inside a block can walk out). Returns the distance moved.
  double sweep(glm::dvec3 &pos, int axis, double d) const {
    if (d == 0.0)
      return 0.0;
    double mn[3] = {pos.x - hw, pos.y, pos.z - hw};
    double mx[3] = {pos.x + hw, pos.y + h, pos.z + hw};
    const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
    const int32_t lo1 = floorToInt(mn[a1] + kEps), hi1 = floorToInt(mx[a1] - kEps);
    const int32_t lo2 = floorToInt(mn[a2] + kEps), hi2 = floorToInt(mx[a2] - kEps);

    auto layerSolid = [&](int32_t i) {
      int32_t c[3];
      c[axis] = i;
      for (int32_t u = lo1; u <= hi1; ++u)
        for (int32_t v = lo2; v <= hi2; ++v) {
          c[a1] = u;
          c[a2] = v;
          if (solidAt(c[0], c[1], c[2]))
            return true;
        }
      return false;
    };

    double moved = d;
    if (d > 0.0) {
      const double edge = mx[axis];
      const int32_t first = floorToInt(edge - kEps) + 1;
      const int32_t last = floorToInt(edge + d - kEps);
      for (int32_t i = first; i <= last; ++i)
        if (layerSolid(i)) {
          moved = std::max(0.0, double(i) - edge);
          break;
        }
    } else {
      const double edge = mn[axis];
      const int32_t first = floorToInt(edge + kEps) - 1;
      const int32_t last = floorToInt(edge + d + kEps);
      for (int32_t i = first; i >= last; --i)
        if (layerSolid(i)) {
          moved = std::min(0.0, double(i + 1) - edge);
          break;
        }
    }
    (axis == 0 ? pos.x : axis == 1 ? pos.y : pos.z) += moved;
    return moved;
  }
};

bool isLiquidAt(const IBlockAccess &world, const BlockRegistry &blocks, const glm::dvec3 &p) {
  const BlockPos b{floorToInt(p.x), floorToInt(p.y), floorToInt(p.z)};
  if (b.y < 0 || b.y >= kWorldHeight || !world.isLoaded(atm::voxel::chunkOf(b)))
    return false;
  return blocks.get(world.blockAt(b)).liquid;
}

float approach(float v, float target, float maxDelta) {
  if (v < target)
    return std::min(v + maxDelta, target);
  return std::max(v - maxDelta, target);
}

} // namespace

void stepMovement(MoveState &s, const MoveInput &in, const IBlockAccess &world,
                  const BlockRegistry &blocks, float dt) {
  const MoveTuning T = moveTuning(); // hoisted copy: consistent within the step
  if (!(dt > 0.0f))
    return;

  const uint16_t buttons = in.buttons;
  const uint16_t pressed = uint16_t(buttons & ~s.prevButtons);
  const uint16_t released = uint16_t(~buttons & s.prevButtons);
  const bool jumpHeld = (buttons & (button::Jump | button::Glide)) != 0;
  const bool jumpPressed = (pressed & button::Jump) != 0;

  // Camera-relative wish direction.
  s.yaw = in.yaw;
  const float sy = std::sin(in.yaw), cy = std::cos(in.yaw);
  const glm::vec3 forward(-sy, 0.0f, -cy), right(cy, 0.0f, -sy);
  glm::vec3 wish = right * std::clamp(in.moveX, -1.0f, 1.0f) + forward * std::clamp(in.moveZ, -1.0f, 1.0f);
  const float wishLen = glm::length(wish);
  if (wishLen > 1.0f)
    wish /= wishLen;
  const bool moving = wishLen > 0.05f;

  // Water: sample at mid-body.
  s.inWater = isLiquidAt(world, blocks, s.pos + glm::dvec3(0.0, T.height * 0.45, 0.0));
  const bool headInWater = isLiquidAt(world, blocks, s.pos + glm::dvec3(0.0, T.eyeHeight, 0.0));

  s.dashCooldown = std::max(0.0f, s.dashCooldown - dt);
  s.dashTimer = std::max(0.0f, s.dashTimer - dt);

  // Dash: burst in the move direction (or facing), brief, on cooldown.
  if ((pressed & button::Dash) && s.dashCooldown <= 0.0f && !s.inWater) {
    s.dashDir = moving ? glm::normalize(wish) : forward;
    s.dashTimer = T.dashTime;
    s.dashCooldown = T.dashCooldown;
    s.gliding = false;
  }
  const bool dashing = s.dashTimer > 0.0f;

  glm::vec2 vh(s.vel.x, s.vel.z);
  if (dashing) {
    vh = glm::vec2(s.dashDir.x, s.dashDir.z) * T.dashSpeed;
    s.vel.y = std::max(s.vel.y, 0.0f); // dashes are flat: no falling mid-dash
  } else if (s.inWater) {
    // Swimming: slower, free 3D with pitch, gentle buoyancy.
    s.gliding = false;
    s.jumpsLeft = uint8_t(std::max(T.extraJumps, 0));
    const glm::vec2 target = glm::vec2(wish.x, wish.z) * T.swimSpeed * std::cos(in.pitch);
    const glm::vec2 diff = target - vh;
    const float maxDelta = T.accelAir * dt;
    const float dl = glm::length(diff);
    vh += dl > maxDelta ? diff * (maxDelta / dl) : diff;
    float targetY = -1.0f; // slow sink
    if (in.moveZ != 0.0f)
      targetY = std::clamp(in.moveZ, -1.0f, 1.0f) * std::sin(in.pitch) * T.swimSpeed;
    if (jumpHeld)
      targetY = T.swimSpeed;
    s.vel.y = approach(s.vel.y, targetY, 20.0f * dt);
    // Hop out at the surface.
    if (jumpPressed && !headInWater)
      s.vel.y = T.jumpSpeed * 0.8f;
  } else {
    const float speed = (buttons & button::Sprint) ? T.sprintSpeed : T.walkSpeed;

    // Jumps.
    if (s.onGround)
      s.jumpsLeft = uint8_t(std::max(T.extraJumps, 0));
    if (jumpPressed) {
      if (s.onGround) {
        s.vel.y = T.jumpSpeed;
        s.onGround = false;
      } else if (s.jumpsLeft > 0) {
        s.vel.y = T.jumpSpeed * 0.95f;
        --s.jumpsLeft;
        s.gliding = false;
      }
    }
    // Variable jump height: releasing jump while rising cuts the rise.
    if ((released & button::Jump) && s.vel.y > 0.0f && !s.onGround)
      s.vel.y *= 0.5f;

    // Glide: hold jump while falling once the extra jumps are used.
    s.gliding = !s.onGround && jumpHeld && s.vel.y <= 0.0f && s.jumpsLeft == 0 &&
                s.glideStamina > 0.0f && !jumpPressed;

    if (s.gliding) {
      const glm::vec3 dir = moving ? wish : forward;
      const glm::vec2 target = glm::vec2(dir.x, dir.z) * T.glideSpeed;
      const glm::vec2 diff = target - vh;
      const float maxDelta = T.accelAir * dt;
      const float dl = glm::length(diff);
      vh += dl > maxDelta ? diff * (maxDelta / dl) : diff;
      s.vel.y = approach(s.vel.y, -T.glideFallSpeed, T.gravity * 2.0f * dt);
      s.glideStamina = std::max(0.0f, s.glideStamina - dt);
    } else {
      // Run: high acceleration toward the target, snappy stop on the ground.
      const glm::vec2 target = glm::vec2(wish.x, wish.z) * speed;
      float rate;
      if (s.onGround)
        rate = moving ? T.accelGround : T.friction * T.walkSpeed;
      else
        rate = moving ? T.accelAir : T.accelAir * 0.25f;
      const glm::vec2 diff = target - vh;
      const float maxDelta = rate * dt;
      const float dl = glm::length(diff);
      vh += dl > maxDelta ? diff * (maxDelta / dl) : diff;
      s.vel.y = std::max(s.vel.y - T.gravity * dt, -T.maxFall);
    }
  }
  if (s.onGround && !s.gliding)
    s.glideStamina = std::min(T.glideStaminaMax, s.glideStamina + dt * 2.0f);

  s.vel.x = vh.x;
  s.vel.z = vh.y;

  // --- collision: swept AABB per axis (Y, then X, then Z) -------------------
  const Collider col{world, blocks, double(T.halfWidth), double(T.height)};
  const bool wasOnGround = s.onGround;
  const double dy = double(s.vel.y) * dt;
  const double dx = double(s.vel.x) * dt;
  const double dz = double(s.vel.z) * dt;

  const double movedY = col.sweep(s.pos, 1, dy);
  if (dy < 0.0 && movedY > dy + 1e-9) {
    s.onGround = true;
    s.vel.y = 0.0f;
  } else if (dy > 0.0 && movedY < dy - 1e-9) {
    s.vel.y = 0.0f; // bumped the ceiling
    s.onGround = false;
  } else {
    s.onGround = false;
  }
  // Still standing on something when not moving vertically.
  if (!s.onGround && dy == 0.0) {
    glm::dvec3 probe = s.pos;
    s.onGround = col.sweep(probe, 1, -0.01) > -0.01 + 1e-9;
  }

  const glm::dvec3 before = s.pos;
  const double mx = col.sweep(s.pos, 0, dx);
  const double mz = col.sweep(s.pos, 2, dz);
  const bool blocked = std::abs(mx - dx) > 1e-9 || std::abs(mz - dz) > 1e-9;

  // Auto step-up of one block when blocked on the ground (not while swimming).
  if (blocked && (wasOnGround || s.onGround) && !s.inWater && s.vel.y <= 0.0f) {
    glm::dvec3 alt = before;
    const double up = col.sweep(alt, 1, double(T.stepHeight));
    if (up > 0.0) {
      col.sweep(alt, 0, dx);
      col.sweep(alt, 2, dz);
      col.sweep(alt, 1, -up);
      const double gainPlain = (s.pos.x - before.x) * (s.pos.x - before.x) +
                               (s.pos.z - before.z) * (s.pos.z - before.z);
      const double gainStep = (alt.x - before.x) * (alt.x - before.x) +
                              (alt.z - before.z) * (alt.z - before.z);
      if (gainStep > gainPlain + 1e-9) {
        s.pos = alt;
        s.onGround = true;
        s.vel.y = 0.0f;
      }
    }
  }
  if (std::abs(s.pos.x - before.x) < std::abs(dx) - 1e-9 && !s.gliding && !dashing)
    s.vel.x = float((s.pos.x - before.x) / dt);
  if (std::abs(s.pos.z - before.z) < std::abs(dz) - 1e-9 && !s.gliding && !dashing)
    s.vel.z = float((s.pos.z - before.z) / dt);

  if (s.onGround) {
    s.gliding = false;
    s.jumpsLeft = uint8_t(std::max(T.extraJumps, 0));
  }
  s.prevButtons = buttons;
}

// Amanatides & Woo voxel traversal.
bool raycastBlock(const IBlockAccess &world, const BlockRegistry &blocks, glm::dvec3 origin,
                  glm::vec3 dir, float maxDistance, BlockPos &hit, atm::voxel::FaceDir &face) {
  const double len = glm::length(glm::dvec3(dir));
  if (!(len > 1e-9) || !(maxDistance > 0.0f))
    return false;
  const glm::dvec3 d = glm::dvec3(dir) / len;

  int32_t x = floorToInt(origin.x), y = floorToInt(origin.y), z = floorToInt(origin.z);
  const int sx = d.x > 0 ? 1 : (d.x < 0 ? -1 : 0);
  const int sy = d.y > 0 ? 1 : (d.y < 0 ? -1 : 0);
  const int sz = d.z > 0 ? 1 : (d.z < 0 ? -1 : 0);
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const double tdx = sx ? std::abs(1.0 / d.x) : kInf;
  const double tdy = sy ? std::abs(1.0 / d.y) : kInf;
  const double tdz = sz ? std::abs(1.0 / d.z) : kInf;
  double tmx = sx > 0 ? (std::floor(origin.x) + 1.0 - origin.x) * tdx
             : sx < 0 ? (origin.x - std::floor(origin.x)) * tdx : kInf;
  double tmy = sy > 0 ? (std::floor(origin.y) + 1.0 - origin.y) * tdy
             : sy < 0 ? (origin.y - std::floor(origin.y)) * tdy : kInf;
  double tmz = sz > 0 ? (std::floor(origin.z) + 1.0 - origin.z) * tdz
             : sz < 0 ? (origin.z - std::floor(origin.z)) * tdz : kInf;

  using atm::voxel::FaceDir;
  const int maxSteps = int(maxDistance * 3.0f) + 4;
  for (int i = 0; i < maxSteps; ++i) {
    double t;
    FaceDir f;
    if (tmx <= tmy && tmx <= tmz) {
      t = tmx;
      x += sx;
      tmx += tdx;
      f = sx > 0 ? FaceDir::NegX : FaceDir::PosX;
    } else if (tmy <= tmz) {
      t = tmy;
      y += sy;
      tmy += tdy;
      f = sy > 0 ? FaceDir::NegY : FaceDir::PosY;
    } else {
      t = tmz;
      z += sz;
      tmz += tdz;
      f = sz > 0 ? FaceDir::NegZ : FaceDir::PosZ;
    }
    if (t > double(maxDistance))
      return false;
    if (y < 0 || y >= kWorldHeight)
      continue; // outside the world vertically: keep going (may come back)
    const BlockPos p{x, y, z};
    if (!world.isLoaded(atm::voxel::chunkOf(p)))
      return false;
    const BlockId id = world.blockAt(p);
    if (id == atm::voxel::kAir)
      continue;
    const auto &def = blocks.get(id);
    if (def.liquid || def.render == atm::voxel::BlockRender::None)
      continue;
    hit = p;
    face = f;
    return true;
  }
  return false;
}

} // namespace ao
