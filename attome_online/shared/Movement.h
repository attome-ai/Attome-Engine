#pragma once

// Trove-style character movement (GAME_DESIGN §7). Deterministic, fixed-step,
// and shared: the client runs it to predict its own player, the server runs
// the same code to verify. Keep it free of randomness and wall-clock time.

#include "GameTypes.h"

#include "../../engine/voxel/Chunk.h"

namespace ao {

struct MoveInput {
  Tick tick = 0;
  uint32_t seq = 0;          // increments per input; used for reconciliation
  float moveX = 0, moveZ = 0; // -1..1 relative to yaw (strafe, forward)
  float yaw = 0, pitch = 0;   // radians
  uint16_t buttons = 0;       // ao::button bits
};

struct MoveTuning {           // backed by ATM_TUNABLEs (see Movement.cpp)
  float walkSpeed = 6.0f, sprintSpeed = 9.0f;
  float accelGround = 60.0f, accelAir = 18.0f, friction = 12.0f;
  float gravity = 30.0f, jumpSpeed = 10.5f, maxFall = 50.0f;
  int extraJumps = 1;                         // double jump
  float glideFallSpeed = 2.5f, glideSpeed = 11.0f, glideStaminaMax = 3.0f;
  float dashSpeed = 22.0f, dashTime = 0.18f, dashCooldown = 0.9f;
  float swimSpeed = 4.0f;
  float halfWidth = 0.3f, height = 1.8f, eyeHeight = 1.6f;
  float stepHeight = 1.05f;                   // auto step up one block
};
const MoveTuning &moveTuning();

struct MoveState {
  glm::dvec3 pos{0.0};       // feet position, blocks
  glm::vec3 vel{0.0f};
  float yaw = 0.0f;
  bool onGround = false;
  bool inWater = false;
  bool gliding = false;
  uint8_t jumpsLeft = 0;
  float glideStamina = 0.0f;
  float dashTimer = 0.0f, dashCooldown = 0.0f;
  glm::vec3 dashDir{0.0f};
  uint16_t prevButtons = 0;
};

// Advances one fixed step. `world` may report unloaded chunks: movement then
// treats them as solid so players can't fall through unloaded ground.
void stepMovement(MoveState &state, const MoveInput &input,
                  const atm::voxel::IBlockAccess &world,
                  const atm::voxel::BlockRegistry &blocks, float dt);

// Voxel ray cast (DDA) for block targeting; returns false when nothing within
// maxDistance. `face` is the FaceDir of the hit face (for placing blocks).
bool raycastBlock(const atm::voxel::IBlockAccess &world,
                  const atm::voxel::BlockRegistry &blocks, glm::dvec3 origin,
                  glm::vec3 dir, float maxDistance, BlockPos &hit,
                  atm::voxel::FaceDir &face);

} // namespace ao
