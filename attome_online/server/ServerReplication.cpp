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
#include <chrono>
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
constexpr int kAppearancePerSnapshot = 24; // AppearanceMsgs per client per snapshot pass


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
  gridCount = 0;
  gridRefs.clear();
  for (auto &[id, e] : entities) {
    if (e.remove) {
      e.gridIndex = 0xFFFFFFFFu;
      continue;
    }
    e.gridIndex = gridCount++;
    GridRef g;
    g.x = float(e.move.pos.x);
    g.z = float(e.move.pos.z);
    g.index = e.gridIndex;
    g.id = e.id;
    g.appearanceVersion = e.appearanceVersion;
    g.kind = e.kind;
    g.e = &e;
    grid[gridCellKey(gridCellOf(e.move.pos.x), gridCellOf(e.move.pos.z))].push_back(g);
    gridRefs.push_back(g);
  }
  size_t cap = 16;
  while (cap < size_t(gridCount) * 2) cap <<= 1;
  gridIds.assign(cap, {kNoEntity, 0u});
  for (const GridRef &g : gridRefs) {
    for (size_t i = size_t(uint64_t(g.id) * 0x9E3779B97F4A7C15ull >> 40) & (cap - 1);; i = (i + 1) & (cap - 1)) {
      if (gridIds[i].first == kNoEntity) {
        gridIds[i] = {g.id, g.index};
        break;
      }
    }
  }
}

void ServerState::sendSnapshots() {
  using clk = std::chrono::steady_clock;
  for (auto &v : shardPlayers) v.clear();
  // Due when this player's own kSnapshotHz clock ticks over; the phase
  // (peer index) staggers players so each tick carries an even share.
  auto due = [&](const Player &pl) {
    const uint64_t t = uint64_t(tick) + pl.peer.index() % uint32_t(kSimHz);
    return (t * kSnapshotHz) / kSimHz != ((t - 1) * kSnapshotHz) / kSimHz;
  };
  for (auto &[key, pl] : players) {
    if (!pl.welcomed || !due(pl) || !findEntity(pl.entity)) continue;
    shardPlayers[std::min(shardOf(pl.peer), shardPlayers.size() - 1)].push_back(&pl);
  }
  snapshotOrder.clear();
  for (auto &v : shardPlayers) snapshotOrder.insert(snapshotOrder.end(), v.begin(), v.end());

  // 1. Build every client's snapshot in parallel. Each call reads the shared
  //    world (entities, grid, players) and writes only its own Player and
  //    worker scratch; sends go to the player's outbox.
  for (SnapWorker &w : snapWorkers) {
    if (w.stamp.size() < gridCount) {
      w.stamp.assign(gridCount, 0u);
      w.stampGen = 0;
    }
  }
  jobs.run(
      snapshotOrder.size(),
      [&](size_t i, size_t worker) {
        Player &pl = *snapshotOrder[i];
        buildSnapshot(pl, *findEntity(pl.entity), snapWorkers[worker]);
      },
      16);

  // 2. Hand the outboxes to the network, one thread per shard.
  const auto t0 = clk::now();
  jobs.run(shardPlayers.size(), [&](size_t k, size_t) {
    for (Player *pl : shardPlayers[k]) flushOutbox(*pl);
  });
  phaseMs[14] += std::chrono::duration<double, std::milli>(clk::now() - t0).count();

  // Profile: sub-phase times are CPU time summed over workers; report them
  // as wall time (divided by the worker count).
  for (SnapWorker &w : snapWorkers) {
    for (size_t i = 0; i < 4; ++i) phaseMs[10 + i] += w.phaseMs[i] / double(jobs.threads());
    phaseMs[16] += w.phaseMs[4] / double(jobs.threads());
    w.phaseMs = {};
    snapScanned += w.scanned;
    snapAppearance += w.appearance;
    snapEntities += w.entities;
    w.scanned = w.appearance = w.entities = 0;
  }
}

void ServerState::flushOutbox(Player &pl) {
  const size_t k = shardOf(pl.peer);
  if (k >= hosts.size()) return;
  atm::net2::Host &h = *hosts[k];
  const PeerId local = toLocal(pl.peer);
  for (size_t i = 0; i < pl.outAppearanceFor.size(); ++i) {
    if (!h.send(local, Channel::ReliableOrdered, pl.outAppearance[i])) break; // backlog: retry next pass
    if (RelTrack *rt = pl.rel.find(pl.outAppearanceFor[i].first)) rt->appearanceVersion = pl.outAppearanceFor[i].second;
  }
  pl.outAppearanceFor.clear();
  if (!pl.outSnapshot.empty()) h.send(local, Channel::Unreliable, pl.outSnapshot);
  pl.outSnapshot.clear();
}

void ServerState::onSnapshotAck(Player &pl, Tick ack) {
  pl.lastSnapshotAck = ack;
  pl.hasSnapshotAck = true;
  const SnapRecord &rec = pl.records[ack % kSnapRecords];
  if (!rec.valid || rec.tick != ack) return; // too old: fields stay unconfirmed (sent again)
  for (const auto &[id, mask] : rec.included) {
    RelTrack *rtp = pl.rel.find(id);
    if (!rtp) continue;
    RelTrack &rt = *rtp;
    for (int f = 0; f < kFieldCount; ++f) {
      const uint8_t bit = uint8_t(1u << f);
      if ((mask & bit) && (rt.sentMask & bit) && rt.changeTick[size_t(f)] <= ack) rt.confirmed |= bit;
    }
  }
}

// Runs on job threads, several players at once: reads shared state only and
// writes `pl`, `w` and pl's outbox (sent later by flushOutbox()).
void ServerState::buildSnapshot(Player &pl, Entity &self, SnapWorker &w) {
  const Tick now = tick;
  const uint32_t pass = ++pl.snapshotsSent;
  const glm::dvec3 origin = self.move.pos;
  using clk = std::chrono::steady_clock;
  auto mark = clk::now();
  double appearanceMs = 0.0;
  auto phase = [&](int i) { // tick profiler sub-phases (ServerStats::kPhaseNames[10 + i])
    const auto t = clk::now();
    w.phaseMs[size_t(i)] += std::chrono::duration<double, std::milli>(t - mark).count();
    mark = t;
  };

  // ---- 1. Relevancy: the nearest maxRelevantEntities within the far ring ----
  // Bounded per client, whatever the crowd: the tracked set is re-ranked,
  // then new entities are looked for cell ring by ring outward with a fixed
  // scan budget. Within a cell the scan starts at a rotating offset, so a
  // packed cell (a town crowd) is covered over successive snapshots and the
  // set converges on the nearest. Tracked entities rank 20% closer
  // (hysteresis: no flicker at the edge of the set); monsters attacking this
  // player always rank first. Grid records carry positions, and "already
  // tracked" is a stamp per grid index, so the scan does no hash lookups.
  w.candidates.clear();
  w.nearby.clear();
  if (++w.stampGen == 0) { // wrapped: old stamps could collide
    std::fill(w.stamp.begin(), w.stamp.end(), 0u);
    w.stampGen = 1;
  }
  const uint32_t stamp = w.stampGen;
  const size_t cap = size_t(std::max(1, cfg.maxRelevantEntities));
  constexpr double kFar2 = kFarRange * kFarRange;
  auto visible = [&](const Entity &e) {
    return !(e.kind == EntityKind::DroppedItem && e.owner != kNoEntity && e.owner != self.id &&
             now < e.publicTick); // personal loot: only its owner sees it (GAME_DESIGN §11)
  };
  auto rankOf = [&](const GridRef &g, double d2, bool tracked) {
    if (g.kind == EntityKind::Monster && g.e->target == self.id) return -1.0;
    return tracked ? d2 * 0.64 : d2;
  };
  auto dist2 = [&](const GridRef &g) {
    const double ddx = double(g.x) - origin.x, ddz = double(g.z) - origin.z;
    return ddx * ddx + ddz * ddz;
  };
  // Monsters and dropped items need a look at the Entity; players never do.
  auto admissible = [&](const GridRef &g) {
    return (g.kind != EntityKind::DroppedItem && g.kind != EntityKind::Monster) || (!g.e->remove && visible(*g.e));
  };
  if (self.gridIndex < w.stamp.size()) w.stamp[self.gridIndex] = stamp; // never list yourself
  pl.rel.forEach([&](EntityId id, RelTrack &) {
    const uint32_t gi = gridLookup(id);
    if (gi >= gridRefs.size() || gi >= w.stamp.size()) return; // despawned: removed below
    const GridRef &g = gridRefs[gi];
    w.stamp[gi] = stamp;
    const double d2 = dist2(g);
    if (d2 <= kFar2 && admissible(g)) w.nearby.emplace_back(rankOf(g, d2, true), gi);
  });
  phase(4);
  size_t budget = cap * 2; // entities examined looking for newcomers
  const int32_t cx0 = gridCellOf(origin.x), cz0 = gridCellOf(origin.z);
  const int32_t rings = int32_t(std::ceil(kFarRange / kGridCell));
  for (int32_t ring = 0; ring <= rings && budget > 0; ++ring) {
    for (int32_t dz = -ring; dz <= ring && budget > 0; ++dz) {
      const bool edgeRow = dz == -ring || dz == ring;
      for (int32_t dx = -ring; dx <= ring && budget > 0; dx += edgeRow ? 1 : 2 * std::max(ring, 1)) {
        auto cit = grid.find(gridCellKey(cx0 + dx, cz0 + dz));
        if (cit == grid.end() || cit->second.empty()) continue;
        const std::vector<GridRef> &refs = cit->second;
        const size_t n = refs.size();
        const size_t start = n > budget ? size_t(pass * 7919u + uint32_t(self.id) * 131u) % n : 0;
        for (size_t k = 0; k < n && budget > 0; ++k) {
          const GridRef &g = refs[start + k < n ? start + k : start + k - n];
          --budget;
          ++w.scanned;
          if (g.index >= w.stamp.size() || w.stamp[g.index] == stamp) continue; // tracked or self
          const double d2 = dist2(g);
          if (d2 <= kFar2 && admissible(g)) w.nearby.emplace_back(rankOf(g, d2, false), g.index);
        }
      }
    }
    // Every cell of the next ring is at least ring * kGridCell away: stop
    // once the cap is filled with entities at least that close.
    if (w.nearby.size() < cap) continue;
    const double safe = double(ring) * kGridCell * 0.8; // 0.8: tracked entities rank closer
    size_t certain = 0;
    for (const auto &[rank, gi] : w.nearby)
      if (rank <= safe * safe) ++certain;
    if (certain >= cap) break;
  }
  if (w.nearby.size() > cap) {
    std::nth_element(w.nearby.begin(), w.nearby.begin() + std::ptrdiff_t(cap), w.nearby.end(),
                     [](const auto &a, const auto &b) { return a.first < b.first; });
    w.nearby.resize(cap);
  }
  phase(0);

  int appearanceBudget = kAppearancePerSnapshot;
  for (const auto &[rank, gi] : w.nearby) {
    const GridRef &g = gridRefs[gi];
    const double d2 = dist2(g);

    auto [rtp, inserted] = pl.rel.tryEmplace(g.id);
    RelTrack &rt = *rtp;
    if (inserted) rt.priority = 1000.0f; // newcomers first
    rt.seenTick = now;

    // Appearance for players entering the set / changing gear (reliable),
    // a few per snapshot so a crowd doesn't flood the reliable channel.
    if (g.kind == EntityKind::Player && appearanceBudget > 0 && rt.appearanceVersion != g.appearanceVersion) {
      if (const Player *other = playerOf(*g.e)) {
        const auto ta = clk::now();
        ++w.appearance;
        --appearanceBudget;
        AppearanceMsg am;
        am.entity = g.id;
        am.name = other->name;
        am.appearance = other->appearance;
        const size_t slot = pl.outAppearanceFor.size();
        if (pl.outAppearance.size() <= slot) pl.outAppearance.emplace_back();
        atm::net2::encodeMessage(pl.outAppearance[slot], am);
        if (!pl.outAppearance[slot].empty()) pl.outAppearanceFor.emplace_back(g.id, g.appearanceVersion);
        appearanceMs += std::chrono::duration<double, std::milli>(clk::now() - ta).count();
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
    if (rank < 0.0) weight *= 4.0f; // monster attacking me
    if (g.kind == EntityKind::Projectile) weight *= 2.0f;
    rt.priority += weight;
    w.candidates.push_back({rt.priority, g.e});
  }

  phase(4);
  w.phaseMs[4] -= appearanceMs;
  w.phaseMs[1] += appearanceMs;

  // ---- 2. Left the set (or despawned): removal, repeated for ~1 s ----
  pl.rel.eraseIf([&](EntityId id, const RelTrack &rt) {
    if (rt.seenTick == now) return false;
    pl.pendingRemoved.emplace_back(id, now);
    return true;
  });
  pl.pendingRemoved.erase(
      std::remove_if(pl.pendingRemoved.begin(), pl.pendingRemoved.end(),
                     [&](const std::pair<EntityId, Tick> &p) {
                       return now - p.second > Tick(kSimHz) || pl.rel.contains(p.first);
                     }),
      pl.pendingRemoved.end());

  phase(2);

  // ---- 3. Snapshot header + self ----
  Snapshot &snap = w.snapMsg.snapshot;
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
  w.measureBuf.clear();
  {
    atm::net2::BitWriter bw(w.measureBuf);
    bw.varu(SnapshotMsg::kId);
    SnapshotCodec::encode(bw, snap);
  }
  const size_t budgetBits = size_t(cfg.snapshotBudgetBytes) * 8;
  size_t usedBits = w.measureBuf.size() * 8 + 16; // + room for a 3-byte entity count

  // ---- 4. Highest priority first, until the packet budget is full ----
  // Only the head of the list fits in one packet: order the top 64 (the rest,
  // rarely reached, follows unordered).
  auto byPriority = [](const SnapCandidate &a, const SnapCandidate &b) { return a.priority > b.priority; };
  const size_t ordered = std::min<size_t>(w.candidates.size(), 64);
  std::partial_sort(w.candidates.begin(), w.candidates.begin() + std::ptrdiff_t(ordered), w.candidates.end(),
                    byPriority);
  SnapRecord &rec = pl.records[now % kSnapRecords];
  rec.tick = now;
  rec.valid = true;
  rec.included.clear();
  int misfits = 0; // consecutive entities that did not fit: the packet is full
  for (const SnapCandidate &c : w.candidates) {
    if (snap.entities.size() >= size_t(kMaxSnapshotEntities)) break;
    if (misfits >= 8 || usedBits + 48 > budgetBits) break;
    const Entity *e = c.e;
    RelTrack *rtp = pl.rel.find(e->id); // the table may have been rebuilt by eraseIf
    if (!rtp) continue;
    RelTrack &rt = *rtp;
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
    es.id = e->id;
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

    const size_t bits = entityStateBits(es);
    if (usedBits + bits > budgetBits) { // try smaller ones; this one keeps its priority
      ++misfits;
      continue;
    }
    misfits = 0;
    usedBits += bits;
    snap.entities.push_back(es);
    rec.included.emplace_back(e->id, mask);

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

  w.entities += snap.entities.size();
  phase(3);
  atm::net2::encodeMessage(pl.outSnapshot, w.snapMsg);
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
      netSend(pl.peer, Channel::ReliableOrdered, msgBuf);
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
    if (netStats(pl.peer).bulkQueuedBytes > kBulkBacklogLimit) break; // don't let chunks crowd out play
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
