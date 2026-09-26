#include "Snapshot.h"

#include "../../engine/net2/Schema.h"

#include <cmath>

// Codecs for Protocol.h Custom<...> fields. Every decode is bounds-checked
// (BitReader never reads past the end) and range-checked (counts, enums,
// non-finite floats), so garbage input can never crash or corrupt the game.

namespace ao {

using atm::net2::BitReader;
using atm::net2::BitWriter;
namespace schema = atm::net2::schema;

namespace {

inline float finiteOr0(float v) { return std::isfinite(v) ? v : 0.0f; }
inline double finiteOr0(double v) { return std::isfinite(v) ? v : 0.0; }
inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

constexpr uint64_t kMaxSnapshotList = 1024; // > kMaxSnapshotEntities, bounds removed lists too
constexpr int kYawBits = 11;

} // namespace

// ---- MoveInput ---------------------------------------------------------------
// Sent exactly (f32): the server replays the same inputs the client predicted
// with, so any quantisation here would cause constant reconciliation drift.
void MoveInputCodec::encode(BitWriter &w, const MoveInput &in) {
  w.u32(in.tick);
  w.varu(in.seq);
  w.f32(in.moveX);
  w.f32(in.moveZ);
  w.f32(in.yaw);
  w.f32(in.pitch);
  w.u16(in.buttons);
}

bool MoveInputCodec::decode(BitReader &r, MoveInput &in) {
  in.tick = r.u32();
  const uint64_t seq = r.varu();
  in.seq = uint32_t(seq);
  in.moveX = clampf(finiteOr0(r.f32()), -1.0f, 1.0f);
  in.moveZ = clampf(finiteOr0(r.f32()), -1.0f, 1.0f);
  // Keep the client's exact yaw (wrapping would change the bits and make the
  // server's replay differ from the prediction); only absurd values wrap.
  const float yaw = finiteOr0(r.f32());
  in.yaw = std::fabs(yaw) <= 1.0e4f ? yaw : schema::wrapAngle(yaw);
  in.pitch = clampf(finiteOr0(r.f32()), -1.5707964f, 1.5707964f);
  in.buttons = r.u16();
  return r.ok() && seq <= 0xFFFFFFFFull;
}

// ---- Appearance --------------------------------------------------------------
void AppearanceCodec::encode(BitWriter &w, const atm::model::Appearance &a) {
  for (auto piece : a.pieces) w.varu(piece);
  w.u8(a.skinTone);
  w.u8(a.hairStyle);
  w.u8(a.hairColor);
  w.u8(a.bodyShape);
  for (auto d : a.dyes) w.u8(d);
}

bool AppearanceCodec::decode(BitReader &r, atm::model::Appearance &a) {
  for (auto &piece : a.pieces) {
    const uint64_t v = r.varu();
    if (v > 0xFFFFu) return false;
    piece = atm::model::PieceId(v);
  }
  a.skinTone = r.u8();
  a.hairStyle = r.u8();
  a.hairColor = r.u8();
  a.bodyShape = r.u8();
  for (auto &d : a.dyes) d = r.u8();
  return r.ok();
}

// ---- ItemStack ---------------------------------------------------------------
void ItemStackCodec::encode(BitWriter &w, const ItemStack &s) {
  w.varu(s.item);
  w.varu(s.item ? s.count : 0u);
}

bool ItemStackCodec::decode(BitReader &r, ItemStack &s) {
  const uint64_t item = r.varu();
  const uint64_t count = r.varu();
  if (!r.ok() || item > 0xFFFFu || count > 0xFFFFu) return false;
  s.item = ItemId(item);
  s.count = uint16_t(count);
  if (s.item == 0) s.count = 0;
  return true;
}

// ---- Snapshot ----------------------------------------------------------------
namespace {

void encodeSelf(BitWriter &w, const SelfState &s) {
  // Full precision: reconciliation replays inputs from exactly this state.
  const MoveState &m = s.move;
  w.f64(m.pos.x);
  w.f64(m.pos.y);
  w.f64(m.pos.z);
  w.f32(m.vel.x);
  w.f32(m.vel.y);
  w.f32(m.vel.z);
  w.f32(m.yaw);
  w.boolean(m.onGround);
  w.boolean(m.inWater);
  w.boolean(m.gliding);
  w.u8(m.jumpsLeft);
  w.f32(m.glideStamina);
  w.f32(m.dashTimer);
  w.f32(m.dashCooldown);
  w.f32(m.dashDir.x);
  w.f32(m.dashDir.y);
  w.f32(m.dashDir.z);
  w.u16(m.prevButtons);
  w.u16(s.hp);
  w.u16(s.maxHp);
  w.u16(s.stamina);
}

bool decodeSelf(BitReader &r, SelfState &s) {
  MoveState &m = s.move;
  m.pos.x = finiteOr0(r.f64());
  m.pos.y = finiteOr0(r.f64());
  m.pos.z = finiteOr0(r.f64());
  m.vel.x = finiteOr0(r.f32());
  m.vel.y = finiteOr0(r.f32());
  m.vel.z = finiteOr0(r.f32());
  m.yaw = finiteOr0(r.f32());
  m.onGround = r.boolean();
  m.inWater = r.boolean();
  m.gliding = r.boolean();
  m.jumpsLeft = r.u8();
  m.glideStamina = finiteOr0(r.f32());
  m.dashTimer = finiteOr0(r.f32());
  m.dashCooldown = finiteOr0(r.f32());
  m.dashDir.x = finiteOr0(r.f32());
  m.dashDir.y = finiteOr0(r.f32());
  m.dashDir.z = finiteOr0(r.f32());
  m.prevButtons = r.u16();
  s.hp = r.u16();
  s.maxHp = r.u16();
  s.stamina = r.u16();
  return r.ok();
}

} // namespace

void encodeEntityState(BitWriter &w, const EntityState &e) {
  const uint16_t mask = uint16_t(e.mask & field::All);
  w.varu(e.id);
  w.bits(mask, 7);
  if (mask & field::Type) {
    w.bits(uint32_t(e.kind), 2);
    w.u8(e.type);
    if (e.kind == EntityKind::DroppedItem) w.u16(e.item);
  }
  if (mask & field::Pos) schema::Pos::encode(w, e.pos);
  if (mask & field::Vel) schema::Vel::encode(w, e.vel);
  if (mask & field::Yaw) schema::Angle<kYawBits>::encode(w, e.yaw);
  if (mask & field::Anim) {
    w.u8(e.locoAnim);
    w.u8(e.actionAnim);
    w.u8(e.actionSeq);
  }
  if (mask & field::Health) {
    w.varu(e.hp);
    w.varu(e.maxHp);
  }
  if (mask & field::Flags) w.u8(e.flags);
}

size_t entityStateBits(const EntityState &e) {
  auto varuBits = [](uint64_t v) {
    size_t n = 8;
    for (; v > 0x7F; v >>= 7) n += 8;
    return n;
  };
  const uint16_t mask = uint16_t(e.mask & field::All);
  size_t n = varuBits(e.id) + 7;
  if (mask & field::Type) n += 2 + 8 + (e.kind == EntityKind::DroppedItem ? 16 : 0);
  if (mask & field::Pos) n += 32 + 16 + 32;
  if (mask & field::Vel) n += 3 * 16;
  if (mask & field::Yaw) n += kYawBits;
  if (mask & field::Anim) n += 3 * 8;
  if (mask & field::Health) n += varuBits(e.hp) + varuBits(e.maxHp);
  if (mask & field::Flags) n += 8;
  return n;
}

namespace {

bool decodeEntity(BitReader &r, EntityState &e) {
  const uint64_t id = r.varu();
  if (!r.ok() || id > 0xFFFFFFFFull) return false;
  e.id = EntityId(id);
  e.mask = uint16_t(r.bits(7));
  if (e.mask & field::Type) {
    e.kind = EntityKind(r.bits(2));
    e.type = r.u8();
    if (e.kind == EntityKind::DroppedItem) e.item = r.u16();
  }
  if (e.mask & field::Pos) schema::Pos::decode(r, e.pos);
  if (e.mask & field::Vel) schema::Vel::decode(r, e.vel);
  if (e.mask & field::Yaw) schema::Angle<kYawBits>::decode(r, e.yaw);
  if (e.mask & field::Anim) {
    e.locoAnim = r.u8();
    e.actionAnim = r.u8();
    e.actionSeq = r.u8();
  }
  if (e.mask & field::Health) {
    const uint64_t hp = r.varu(), maxHp = r.varu();
    if (hp > 0xFFFFu || maxHp > 0xFFFFu) return false;
    e.hp = uint16_t(hp);
    e.maxHp = uint16_t(maxHp);
  }
  if (e.mask & field::Flags) e.flags = r.u8();
  return r.ok();
}

} // namespace

void SnapshotCodec::encode(BitWriter &w, const Snapshot &s) {
  w.u32(s.tick);
  w.varu(s.ackInputSeq);
  encodeSelf(w, s.self);
  const size_t n = s.entities.size() < kMaxSnapshotList ? s.entities.size() : size_t(kMaxSnapshotList);
  w.varu(n);
  for (size_t i = 0; i < n; ++i) encodeEntityState(w, s.entities[i]);
  const size_t nr = s.removed.size() < kMaxSnapshotList ? s.removed.size() : size_t(kMaxSnapshotList);
  w.varu(nr);
  for (size_t i = 0; i < nr; ++i) w.varu(s.removed[i]);
}

bool SnapshotCodec::decode(BitReader &r, Snapshot &s) {
  s.tick = r.u32();
  const uint64_t ack = r.varu();
  if (!r.ok() || ack > 0xFFFFFFFFull) return false;
  s.ackInputSeq = uint32_t(ack);
  if (!decodeSelf(r, s.self)) return false;
  const uint64_t n = r.varu();
  if (!r.ok() || n > kMaxSnapshotList || n * 8 > r.remainingBits()) return false;
  s.entities.clear(); // keeps capacity: no allocation in steady state
  s.entities.resize(size_t(n));
  for (auto &e : s.entities)
    if (!decodeEntity(r, e)) return false;
  const uint64_t nr = r.varu();
  if (!r.ok() || nr > kMaxSnapshotList || nr * 8 > r.remainingBits()) return false;
  s.removed.clear();
  s.removed.resize(size_t(nr));
  for (auto &id : s.removed) {
    const uint64_t v = r.varu();
    if (v > 0xFFFFFFFFull) return false;
    id = EntityId(v);
  }
  return r.ok();
}

} // namespace ao
