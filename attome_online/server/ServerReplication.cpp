// Replication (NETWORK_PLAN §6): interest grid, near/mid/far rings, priority
// accumulators, field-mask deltas against acknowledged snapshots, and
// edited-chunk streaming.
//
// Delta rule (matches a client that merges each snapshot into its latest
// known state): a field is left out only if its value is unchanged since the
// snapshot tick it was first sent with (changeTick) AND the client has
// acknowledged a snapshot >= changeTick that contained the field. Until then
// the field is re-sent in every snapshot that includes the entity.

#include "ServerState.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace ao::server {

using namespace ao::proto;
namespace vx = atm::voxel;
namespace schema = atm::net2::schema;

namespace {

constexpr int kAngleBits = 11; // must match Snapshot.cpp
constexpr uint64_t kBulkBacklogLimit = 256u * 1024u;
constexpr int kChunksPerSync = 8;

inline uint64_t cellKey(int32_t cx, int32_t cz) { return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz); }
inline int32_t cellOf(double v) { return int32_t(std::floor(v / kGridCell)); }

uint16_t quantYaw(float yaw) {
  const float t = (schema::wrapAngle(yaw) + schema::kPi) / (2.0f * schema::kPi);
  return uint16_t(uint32_t(std::lround(double(t) * double(1u << kAngleBits))) & ((1u << kAngleBits) - 1u));
}

bool fieldEqual(const Quant &a, const Quant &b, int f) {
  switch (f) {
  case 0: return a.px == b.px && a.py == b.py && a.pz == b.pz;
  case 1: return a.vx == b.vx && a.vy == b.vy && a.vz == b.vz;
  case 2: return a.yaw == b.yaw;
  case 3: return a.loco == b.loco && a.action == b.action && a.actionSeq == b.actionSeq;
  case 4: return a.hp == b.hp && a.maxHp == b.maxHp;
  case 5: return a.flags == b.flags;
  case 6: return a.kind == b.kind && a.type == b.type && a.item == b.item;
  default: return true;
  }
}

} // namespace

Quant ServerState::quantize(const Entity &e) const {
  Quant q;
  q.px = schema::Pos::qxz(e.move.pos.x);
  q.py = schema::Pos::qy(e.move.pos.y);
  q.pz = schema::Pos::qxz(e.move.pos.z);
  q.vx = schema::Vel::q(e.move.vel.x);
  q.vy = schema::Vel::q(e.move.vel.y);
  q.vz = schema::Vel::q(e.move.vel.z);
  q.yaw = quantYaw(e.move.yaw);
  q.loco = 0;
  q.action = e.actionAnim;
  q.actionSeq = e.actionSeq;
  q.hp = e.hp;
  q.maxHp = e.maxHp;
  uint8_t flags = 0;
  if (e.move.gliding) flags |= eflag::Gliding;
  if (e.move.inWater) flags |= eflag::InWater;
  if (e.dead) flags |= eflag::Dead;
  if (e.move.onGround) flags |= eflag::OnGround;
  q.flags = flags;
  q.kind = uint8_t(e.kind);
  q.type = e.type;
  q.item = e.kind == EntityKind::DroppedItem ? e.item : uint16_t(0);
  return q;
}

void ServerState::buildGrid() {
  if (grid.size() > 4096 && grid.size() > entities.size() * 4) grid.clear(); // forget stale cells
  for (auto &kv : grid) kv.second.clear();
  for (const auto &[id, e] : entities) {
    if (e.remove) continue;
    grid[cellKey(cellOf(e.move.pos.x), cellOf(e.move.pos.z))].push_back(id);
  }
}

void ServerState::sendSnapshots() {
  for (auto &[key, pl] : players) {
    if (!pl.welcomed) continue;
    Entity *self = findEntity(pl.entity);
    if (!self) continue;
    buildSnapshot(pl, *self);
  }
}

void ServerState::onSnapshotAck(Player &pl, Tick ack) {
  pl.lastSnapshotAck = ack;
  pl.hasSnapshotAck = true;
  const SnapRecord &rec = pl.records[ack % kSnapRecords];
  if (!rec.valid || rec.tick != ack) return; // too old: fields stay unconfirmed (sent again)
  for (const auto &[id, mask] : rec.included) {
    auto it = pl.rel.find(id);
    if (it == pl.rel.end()) continue;
    RelTrack &rt = it->second;
    for (int f = 0; f < kFieldCount; ++f) {
      const uint8_t bit = uint8_t(1u << f);
      if ((mask & bit) && (rt.sentMask & bit) && rt.changeTick[size_t(f)] <= ack) rt.confirmed |= bit;
    }
  }
}

void ServerState::buildSnapshot(Player &pl, Entity &self) {
  const Tick now = tick;
  const uint32_t pass = ++pl.snapshotsSent;
  const glm::dvec3 origin = self.move.pos;

  // ---- 1. Relevancy: grid cells within the far ring ----
  candidates.clear();
  const int32_t cx0 = cellOf(origin.x), cz0 = cellOf(origin.z);
  const int32_t r = int32_t(std::ceil(kFarRange / kGridCell));
  for (int32_t dz = -r; dz <= r; ++dz) {
    for (int32_t dx = -r; dx <= r; ++dx) {
      auto cit = grid.find(cellKey(cx0 + dx, cz0 + dz));
      if (cit == grid.end()) continue;
      for (EntityId id : cit->second) {
        if (id == self.id) continue;
        Entity *e = findEntity(id);
        if (!e || e->remove) continue;
        if (e->kind == EntityKind::DroppedItem && e->owner != kNoEntity && e->owner != self.id &&
            now < e->publicTick)
          continue; // personal loot: only its owner sees it (GAME_DESIGN §11)
        const double ddx = e->move.pos.x - origin.x, ddz = e->move.pos.z - origin.z;
        const double d2 = ddx * ddx + ddz * ddz;
        if (d2 > kFarRange * kFarRange) continue;

        auto [it, inserted] = pl.rel.try_emplace(id);
        RelTrack &rt = it->second;
        if (inserted) rt.priority = 1000.0f; // newcomers first
        rt.seenTick = now;

        // Appearance for players entering the set / changing gear (reliable).
        if (e->kind == EntityKind::Player) {
          if (Player *other = playerOf(*e); other && rt.appearanceVersion != other->appearanceVersion) {
            AppearanceMsg am;
            am.entity = e->id;
            am.name = other->name;
            am.appearance = other->appearance;
            if (sendTo(pl, am, Channel::ReliableOrdered)) rt.appearanceVersion = other->appearanceVersion;
          }
        }

        // Ring schedule: near every snapshot, mid every 2nd, far every 4th.
        float weight = 4.0f;
        if (d2 > kMidRange * kMidRange) {
          if (pass % 4 != 0) continue;
          weight = 1.0f;
        } else if (d2 > kNearRange * kNearRange) {
          if (pass % 2 != 0) continue;
          weight = 2.0f;
        }
        if (e->kind == EntityKind::Monster && e->target == self.id) weight *= 4.0f; // attacking me
        if (e->kind == EntityKind::Projectile) weight *= 2.0f;
        rt.priority += weight;
        candidates.emplace_back(rt.priority, id);
      }
    }
  }

  // ---- 2. Left the set (or despawned): removal, repeated for ~1 s ----
  for (auto it = pl.rel.begin(); it != pl.rel.end();) {
    if (it->second.seenTick != now) {
      pl.pendingRemoved.emplace_back(it->first, now);
      it = pl.rel.erase(it);
    } else {
      ++it;
    }
  }
  pl.pendingRemoved.erase(
      std::remove_if(pl.pendingRemoved.begin(), pl.pendingRemoved.end(),
                     [&](const std::pair<EntityId, Tick> &p) {
                       return now - p.second > Tick(kSimHz) || pl.rel.count(p.first) != 0;
                     }),
      pl.pendingRemoved.end());

  // ---- 3. Snapshot header + self ----
  Snapshot &snap = snapMsg.snapshot;
  snap.tick = now;
  snap.ackInputSeq = pl.lastAppliedSeq;
  snap.self.move = self.move;
  snap.self.hp = self.hp;
  snap.self.maxHp = self.maxHp;
  snap.self.stamina = uint16_t(std::clamp(self.move.glideStamina * 1000.0f, 0.0f, 65535.0f));
  snap.entities.clear();
  snap.removed.clear();
  for (const auto &p : pl.pendingRemoved) {
    if (snap.removed.size() >= 256) break;
    snap.removed.push_back(p.first);
  }

  // Base size (header, self, removed list, counts).
  measureBuf.clear();
  {
    atm::net2::BitWriter w(measureBuf);
    w.varu(SnapshotMsg::kId);
    SnapshotCodec::encode(w, snap);
  }
  const size_t budgetBits = size_t(cfg.snapshotBudgetBytes) * 8;
  size_t usedBits = measureBuf.size() * 8 + 16; // + room for a 3-byte entity count

  // ---- 4. Highest priority first, until the packet budget is full ----
  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<float, EntityId> &a, const std::pair<float, EntityId> &b) {
              return a.first > b.first;
            });
  SnapRecord &rec = pl.records[now % kSnapRecords];
  rec.tick = now;
  rec.valid = true;
  rec.included.clear();
  for (const auto &[prio, id] : candidates) {
    if (snap.entities.size() >= size_t(kMaxSnapshotEntities)) break;
    Entity *e = findEntity(id);
    auto rit = pl.rel.find(id);
    if (!e || rit == pl.rel.end()) continue;
    RelTrack &rt = rit->second;
    const Quant q = quantize(*e);
    uint16_t mask = 0;
    for (int f = 0; f < kFieldCount; ++f) {
      const uint8_t bit = uint8_t(1u << f);
      const bool changed = !(rt.sentMask & bit) || !fieldEqual(q, rt.last, f);
      if (changed || !(rt.confirmed & bit)) mask |= bit;
    }
    if (mask == 0) { // client is up to date: nothing to send
      rt.priority = 0.0f;
      continue;
    }
    EntityState es;
    es.id = id;
    es.mask = mask;
    es.kind = e->kind;
    es.type = e->type;
    es.item = e->item;
    es.pos = e->move.pos;
    es.vel = e->move.vel;
    es.yaw = e->move.yaw;
    es.locoAnim = 0;
    es.actionAnim = e->actionAnim;
    es.actionSeq = e->actionSeq;
    es.hp = e->hp;
    es.maxHp = e->maxHp;
    es.flags = q.flags;

    measureBuf.clear();
    {
      atm::net2::BitWriter w(measureBuf);
      encodeEntityState(w, es);
    }
    const size_t bits = measureBuf.size() * 8; // byte-rounded: conservative
    if (usedBits + bits > budgetBits) continue; // try smaller ones; this one keeps its priority
    usedBits += bits;
    snap.entities.push_back(es);
    rec.included.emplace_back(id, mask);

    for (int f = 0; f < kFieldCount; ++f) {
      const uint8_t bit = uint8_t(1u << f);
      if (!(mask & bit)) continue;
      const bool changed = !(rt.sentMask & bit) || !fieldEqual(q, rt.last, f);
      if (changed) {
        rt.changeTick[size_t(f)] = now;
        rt.confirmed = uint8_t(rt.confirmed & ~bit);
      }
      rt.sentMask |= bit;
    }
    rt.last = q;
    rt.priority = 0.0f;
  }

  atm::net2::encodeMessage(msgBuf, snapMsg);
  if (!msgBuf.empty()) host->send(pl.peer, Channel::Unreliable, msgBuf);
}

// ---------------------------------------------------------------------------
// Edited chunks
// ---------------------------------------------------------------------------

void ServerState::onBlockEdited(vx::BlockPos p, vx::BlockId id) {
  const vx::ChunkCoord c = vx::chunkOf(p);
  const uint64_t key = c.key();
  if (auto data = world->chunk(c)) {
    if (editedStore.find(key) == editedStore.end()) editedOrder.push_back(key);
    editedStore[key] = std::move(data);
  }
  BlockChanged bc;
  bc.x = p.x;
  bc.y = p.y;
  bc.z = p.z;
  bc.block = id;
  atm::net2::encodeMessage(msgBuf, bc);
  if (msgBuf.empty()) return;
  const int R = cfg.viewRadiusChunks + 1;
  for (auto &[k, pl] : players) {
    if (!pl.welcomed) continue;
    // Any ChunkData already queued for this chunk is now stale: send it again.
    pl.chunkSentMs.erase(key);
    const Entity *e = findEntity(pl.entity);
    if (!e) continue;
    const vx::ChunkCoord pc = vx::chunkOf(vx::BlockPos{int32_t(std::floor(e->move.pos.x)), 0,
                                                       int32_t(std::floor(e->move.pos.z))});
    if (std::abs(pc.x - c.x) <= R && std::abs(pc.z - c.z) <= R)
      host->send(pl.peer, Channel::ReliableOrdered, msgBuf);
  }
}

void ServerState::syncChunks(Player &pl, bool force) {
  pl.nextChunkSyncTick = tick + Tick(kSimHz / 2);
  const Entity *e = findEntity(pl.entity);
  if (!e || editedOrder.empty()) return;
  const vx::ChunkCoord pc = vx::chunkOf(vx::BlockPos{int32_t(std::floor(e->move.pos.x)), 0,
                                                     int32_t(std::floor(e->move.pos.z))});
  const int R = cfg.viewRadiusChunks;
  auto inRange = [&](uint64_t key) {
    const vx::ChunkCoord c = vx::ChunkCoord::fromKey(key);
    return std::abs(c.x - pc.x) <= R && std::abs(c.z - pc.z) <= R;
  };

  // 1. Announce edited chunks in range (reliable, before their data).
  keyScratch.clear();
  for (uint64_t key : editedOrder)
    if (inRange(key) && pl.announced.insert(key).second) keyScratch.push_back(key);
  for (size_t i = 0; i < keyScratch.size(); i += 4096) {
    EditedChunks ec;
    const size_t n = std::min<size_t>(4096, keyScratch.size() - i);
    ec.keys.assign(keyScratch.begin() + std::ptrdiff_t(i), keyScratch.begin() + std::ptrdiff_t(i + n));
    sendTo(pl, ec, Channel::ReliableOrdered);
  }

  // 2. Stream their contents on the bulk channel with leftover bandwidth,
  //    nearest first, while the peer's bulk backlog is small.
  keyScratch.clear();
  for (uint64_t key : editedOrder)
    if (inRange(key) && pl.chunkSentMs.find(key) == pl.chunkSentMs.end()) keyScratch.push_back(key);
  if (keyScratch.empty()) return;
  std::sort(keyScratch.begin(), keyScratch.end(), [&](uint64_t a, uint64_t b) {
    const vx::ChunkCoord ca = vx::ChunkCoord::fromKey(a), cb = vx::ChunkCoord::fromKey(b);
    const int da = std::abs(ca.x - pc.x) + std::abs(ca.z - pc.z);
    const int db = std::abs(cb.x - pc.x) + std::abs(cb.z - pc.z);
    return da < db;
  });
  int sent = 0;
  const uint64_t msNow = atm::net2::nowMs();
  for (uint64_t key : keyScratch) {
    if (!force && sent >= kChunksPerSync) break;
    if (host->stats(pl.peer).bulkQueuedBytes > kBulkBacklogLimit) break; // don't let chunks crowd out play
    auto it = editedStore.find(key);
    if (it == editedStore.end() || !it->second) continue;
    const vx::ChunkCoord c = vx::ChunkCoord::fromKey(key);
    ChunkData cd;
    cd.cx = c.x;
    cd.cy = c.y;
    cd.cz = c.z;
    it->second->serialize(cd.data);
    if (cd.data.size() > 70000) continue; // cannot happen for palette+RLE chunks; never send a truncated blob
    if (!sendTo(pl, cd, Channel::Bulk)) break;
    pl.chunkSentMs[key] = msNow;
    ++sent;
  }
}

} // namespace ao::server
