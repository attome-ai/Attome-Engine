#pragma once

// World snapshots (server -> client, unreliable, 20 Hz) and the custom codecs
// used by Protocol.h.
//
// Delta encoding (NETWORK_PLAN §6.3): each EntityState carries a field mask.
// The server leaves a field out when the client has *acknowledged* a snapshot
// that already contained the same value (the client reports the newest
// snapshot tick it received in InputBatch::lastSnapshotTick; the server keeps
// per-client acked values in its relevancy set). The client fills missing
// fields from its last known state of that entity.
//
// Interest (NETWORK_PLAN §6.2): the server sends only entities in the
// client's relevancy set (spatial grid, near/mid/far rings) and at most
// kMaxSnapshotEntities per snapshot, highest priority first.

#include "GameTypes.h"
#include "Movement.h"

#include "../../engine/net2/BitStream.h"

#include <vector>

namespace ao {

inline constexpr int kMaxSnapshotEntities = 250; // hotspot cap (Decided)

namespace field {
inline constexpr uint16_t Pos = 1u << 0;
inline constexpr uint16_t Vel = 1u << 1;
inline constexpr uint16_t Yaw = 1u << 2;
inline constexpr uint16_t Anim = 1u << 3;   // locomotion + action animation ids
inline constexpr uint16_t Health = 1u << 4;
inline constexpr uint16_t Flags = 1u << 5;
inline constexpr uint16_t Type = 1u << 6;   // kind + type (first time seen)
inline constexpr uint16_t All = 0x7F;
} // namespace field

// Action animation codes (EntityState::actionAnim). The client derives
// locomotion itself from vel + flags; the server sets actionAnim and bumps
// actionSeq whenever a new action starts.
namespace action {
inline constexpr uint8_t None = 0, Swing = 1, BowShoot = 2, Cast = 3, Mine = 4,
                         Place = 5, Hit = 6, Death = 7, Wave = 8;
} // namespace action

// EntityState::flags bits.
namespace eflag {
inline constexpr uint8_t Gliding = 1u << 0, InWater = 1u << 1, Dead = 1u << 2,
                         OnGround = 1u << 3;
} // namespace eflag

struct EntityState {
  EntityId id = kNoEntity;
  uint16_t mask = field::All;       // which fields below are present
  EntityKind kind = EntityKind::Player;
  uint8_t type = 0;                 // monster type / projectile type (weapon) / item id low byte
  uint16_t item = 0;                // DroppedItem only: full item id (sent with field::Type)
  glm::dvec3 pos{0.0};
  glm::vec3 vel{0.0f};
  float yaw = 0.0f;
  uint8_t locoAnim = 0, actionAnim = 0; // locoAnim unused (client derives it); actionAnim = ao::action code
  uint8_t actionSeq = 0;            // increments when a new action starts (replays one-shots)
  uint16_t hp = 0, maxHp = 0;
  uint8_t flags = 0;                // bit0 gliding, bit1 in water, bit2 dead, bit3 onGround
};

// The receiving player's own authoritative state, always sent in full, used
// for prediction reconciliation (rewind to this state, replay inputs after
// ackInputSeq).
struct SelfState {
  MoveState move;
  uint16_t hp = 0, maxHp = 0;
  uint16_t stamina = 0;
};

struct Snapshot {
  Tick tick = 0;
  uint32_t ackInputSeq = 0;         // newest input the server has applied
  SelfState self;
  std::vector<EntityState> entities;
  std::vector<EntityId> removed;    // left the relevancy set or despawned
};

// --- codecs used by Protocol.h (implemented in Snapshot.cpp) ----------------

struct SnapshotCodec {
  static void encode(atm::net2::BitWriter &w, const Snapshot &s);
  static bool decode(atm::net2::BitReader &r, Snapshot &s);
  static constexpr const char *kName = "Snapshot.v1";
};
struct MoveInputCodec {
  static void encode(atm::net2::BitWriter &w, const MoveInput &in);
  static bool decode(atm::net2::BitReader &r, MoveInput &in);
  static constexpr const char *kName = "MoveInput.v1";
};
struct AppearanceCodec {
  static void encode(atm::net2::BitWriter &w, const atm::model::Appearance &a);
  static bool decode(atm::net2::BitReader &r, atm::model::Appearance &a);
  static constexpr const char *kName = "Appearance.v1";
};
// One EntityState as it appears inside a snapshot (exposed so the server can
// measure entities against the per-snapshot byte budget).
void encodeEntityState(atm::net2::BitWriter &w, const EntityState &e);

struct ItemStackCodec {
  static void encode(atm::net2::BitWriter &w, const ItemStack &s);
  static bool decode(atm::net2::BitReader &r, ItemStack &s);
  static constexpr const char *kName = "ItemStack.v1";
};

} // namespace ao
