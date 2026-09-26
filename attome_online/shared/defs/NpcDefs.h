#pragma once

// NPC / monster definitions, loaded from data/npcs.json. The NPC id is the
// snapshot entity `type` of monsters (part of the protocol).

#include "../CoreTypes.h"

#include <string_view>
#include <vector>

namespace ao {

// Hit volume, fitted to the model: a vertical capsule (axis from `bottom` to
// `top` above the feet, radius `radius`). Used by the server for projectile /
// melee hits and by the client for aim assist and the F8 hitbox view.
struct MonsterHitShape {
  float radius = 0.8f, bottom = 0.8f, top = 0.8f;
};

struct MonsterDef {
  std::string_view name;    // internal id name ("golem")
  std::string_view display; // shown to players ("Stone Golem")
  uint16_t level = 1;
  uint16_t maxHp = 1;
  uint16_t damage = 0;
  float speed = 3.0f;          // blocks/s
  float aggroRange = 10.0f;    // blocks
  float attackRange = 1.5f;
  float attackCooldown = 1.0f; // s
  uint32_t xp = 0;             // combat XP per kill (shared by damage)
  const char *model = "";      // ModelLibrary creature name
  MonsterHitShape hit;
  // Loot (GAME_DESIGN §11): common rolls per eligible player + one rare roll.
  struct Drop { ItemId item = 0; uint16_t min = 1, max = 1; float chance = 0.0f; };
  std::vector<Drop> common;
  Drop rare;
};

const MonsterDef &monsterDef(uint8_t type);
uint8_t monsterTypeCount();
int findNpc(std::string_view name); // -1 if unknown

// Well-known NPC ids (checked against data/npcs.json at load).
namespace monsters {
inline constexpr uint8_t Slime = 0, Wolf = 1, Golem = 2;
} // namespace monsters

inline MonsterHitShape monsterHitShape(uint8_t type) { return monsterDef(type).hit; }

// Squared distance from point (px,py,pz) to the capsule of a monster whose
// feet are at (fx,fy,fz); also returns the capsule's closest axis point y.
inline double monsterHitDistance2(uint8_t type, double fx, double fy, double fz, double px, double py,
                                  double pz, double *axisY = nullptr) {
  const MonsterHitShape s = monsterHitShape(type);
  double y = py;
  if (y < fy + s.bottom) y = fy + s.bottom;
  if (y > fy + s.top) y = fy + s.top;
  if (axisY) *axisY = y;
  const double dx = px - fx, dy = py - y, dz = pz - fz;
  return dx * dx + dy * dy + dz * dz;
}

} // namespace ao
