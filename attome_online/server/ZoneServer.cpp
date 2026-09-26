// Zone server lifecycle, network pump and message handlers.

#include "ServerState.h"

#include "../../engine/ATMConfig.h"
#include "../../engine/ATMJson.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

namespace ao::server {

using namespace ao::proto;
namespace net = atm::net2;
namespace vx = atm::voxel;

glm::vec3 aimDirection(float yaw, float pitch) {
  const float cp = std::cos(pitch);
  return glm::vec3(-std::sin(yaw) * cp, std::sin(pitch), -std::cos(yaw) * cp);
}

glm::dvec3 eyePosition(const MoveState &m) {
  return m.pos + glm::dvec3(0.0, double(moveTuning().eyeHeight), 0.0);
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

ServerConfig ServerConfig::load(const std::string &path, std::string *error) {
  ServerConfig c;
  atm::Json root;
  std::string err;
  if (!atm::Json::parseFile(path, root, &err)) {
    if (error) *error = path + ": " + err;
    return c;
  }
  auto num = [&](const char *key, double lo, double hi, double fallback) {
    const atm::Json *v = root.find(key);
    if (!v || !v->isNumber()) return fallback;
    return std::clamp(v->asNumber(), lo, hi);
  };
  c.port = uint16_t(num("port", 0, 65535, c.port));
  c.seed = uint64_t(num("seed", 0, 9007199254740991.0, double(c.seed)));
  c.maxPlayers = uint16_t(num("maxPlayers", 1, 65534, c.maxPlayers));
  c.viewRadiusChunks = int(num("viewRadiusChunks", 2, 32, c.viewRadiusChunks));
  c.simRadiusChunks = int(num("simRadiusChunks", 1, 16, c.simRadiusChunks));
  c.workerThreads = int(num("workerThreads", 0, 64, c.workerThreads));
  c.netShards = int(num("netShards", 1, 64, c.netShards));
  c.jobThreads = int(num("jobThreads", 0, 256, c.jobThreads));
  c.monstersPerPlayer = int(num("monstersPerPlayer", 0, 100, c.monstersPerPlayer));
  c.maxMonsters = int(num("maxMonsters", 0, 100000, c.maxMonsters));
  c.bandwidthBytesPerSec = uint32_t(num("bandwidthKBps", 16, 100000, c.bandwidthBytesPerSec / 1024.0) * 1024.0);
  c.snapshotBudgetBytes = uint32_t(num("snapshotBudgetBytes", 256, 1150, c.snapshotBudgetBytes));
  c.maxRelevantEntities = int(num("maxRelevantEntities", 16, 1000, c.maxRelevantEntities));
  c.timeoutMs = uint32_t(num("timeoutMs", 1000, 600000, c.timeoutMs));
  if (const atm::Json *v = root.find("verbose"); v && v->isBool()) c.verbose = v->asBool();
  return c;
}

// ---------------------------------------------------------------------------
// ZoneServer (thin wrapper)
// ---------------------------------------------------------------------------

ZoneServer::ZoneServer() : s_(std::make_unique<ServerState>()) {}
ZoneServer::~ZoneServer() { stop(); }

bool ZoneServer::start(const ServerConfig &config, std::string *error) {
  if (s_->running) {
    if (error) *error = "server already running";
    return false;
  }
  return s_->start(config, error);
}

void ZoneServer::tick() {
  if (s_ && s_->running) s_->step();
}

void ZoneServer::run(std::atomic<bool> &stopFlag) {
  using clock = std::chrono::steady_clock;
  const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / kSimHz));
  auto next = clock::now();
  while (running() && !stopFlag.load(std::memory_order_relaxed)) {
    s_->step();
    next += dt;
    const auto now = clock::now();
    if (now < next) {
      std::this_thread::sleep_until(next);
    } else if (now - next > dt * 10) {
      next = now; // badly behind (debugger, overload): don't spiral
    }
  }
  stop();
}

void ZoneServer::stop() {
  if (s_ && s_->running) s_->stop();
}

bool ZoneServer::running() const { return s_ && s_->running; }
uint16_t ZoneServer::port() const { return s_ && !s_->hosts.empty() ? s_->hosts[0]->localPort() : 0; }

ServerStats ZoneServer::stats() const {
  ServerStats st;
  if (!s_) return st;
  const ServerState &s = *s_;
  st.tick = s.tick;
  st.entities = uint32_t(s.entities.size());
  st.monsters = s.monsterCount;
  for (const auto &kv : s.players)
    if (kv.second.welcomed) ++st.players;
  if (s.tickMsCount > 0) {
    float buf[300];
    const size_t n = s.tickMsCount;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
      buf[i] = s.tickMs[i];
      sum += buf[i];
    }
    std::sort(buf, buf + n);
    st.tickMsAvg = sum / double(n);
    st.tickMsP99 = buf[std::min(n - 1, size_t(double(n) * 0.99))];
    st.tickMsMax = buf[n - 1];
  }
  for (const auto &h : s.hosts) {
    const net::HostStats hs = h->hostStats();
    st.bytesSent += hs.bytesSent;
    st.bytesReceived += hs.bytesReceived;
    st.malformedPackets += hs.malformedPackets;
  }
  if (s.world) st.loadedChunks = s.world->stats().loadedChunks;
  st.editedChunks = s.editedStore.size();
  st.phaseMsTotal = s.phaseMs;
  st.profiledTicks = s.profiledTicks;
  st.snapScanned = s.snapScanned;
  st.snapAppearance = s.snapAppearance;
  st.snapEntities = s.snapEntities;
  for (const auto &kv : s.players) st.relTracks += kv.second.rel.size();
  return st;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

bool ServerState::start(const ServerConfig &config, std::string *error) {
  cfg = config;
  rng.seed(uint64_t(std::random_device{}()) ^ (cfg.seed * 0x9E3779B97F4A7C15ull) ^
           uint64_t(net::nowMs()));

  // Content definitions (items, NPCs, map regions) from data/*.json.
  if (!loadGameData(atm::resolve_path("data"), error))
    return false;

  blocks = vx::BlockRegistry{};
  blocks.registerDefaults();

  vx::VoxelWorldConfig wc;
  wc.seed = cfg.seed;
  wc.meshing = false;
  wc.workerThreads = cfg.workerThreads;
  wc.viewRadiusChunks = cfg.simRadiusChunks;
  wc.unloadMarginChunks = 1;
  wc.verticalChunksBelow = 2;
  wc.verticalChunksAbove = 2;
  ao::world::installHomeTown(wc); // deterministic home town, same as the clients
  world = std::make_unique<vx::VoxelWorld>(wc, blocks);

  models = atm::model::ModelLibrary{};
  models.buildDefaults();
  itemPiece.assign(itemCount(), atm::model::kNoPiece);
  for (ItemId i = 1; i < itemCount(); ++i) {
    const ItemDef &d = itemDef(i);
    if (d.piece) itemPiece[i] = models.findPiece(d.piece);
  }

  ao::world::townSpawnPoint(ao::world::homeTown(world->generator()), spawnPoint.x, spawnPoint.y, spawnPoint.z);

  net::HostConfig hc;
  hc.maxPeers = cfg.maxPlayers;
  hc.protocolId = kTransportProtocolId;
  hc.timeoutMs = cfg.timeoutMs;
  hc.bandwidthBytesPerSec = cfg.bandwidthBytesPerSec;
  hc.maxBulkBytes = 4u << 20;
  // Network shards: consecutive ports, the player cap split between them.
  const int shards = std::clamp(cfg.netShards, 1, 64);
  shardPeers = uint16_t(std::min<int>((cfg.maxPlayers + shards - 1) / shards, 65534 / shards));
  hc.maxPeers = shardPeers;
  hosts.clear();
  for (int k = 0; k < shards; ++k) {
    auto h = std::make_unique<net::Host>(hc);
    const uint16_t port = cfg.port == 0 ? uint16_t(0) : uint16_t(cfg.port + k);
    if (!h->listen(port, error)) {
      hosts.clear();
      world.reset();
      return false;
    }
    hosts.push_back(std::move(h));
  }
  const int threads = cfg.jobThreads > 0 ? cfg.jobThreads
                                         : int(std::clamp(std::thread::hardware_concurrency() / 2, 1u, 16u));
  jobs.resize(threads);
  snapWorkers.assign(size_t(jobs.threads()), SnapWorker{});
  shardPlayers.assign(hosts.size(), {});
  netScratch.assign(hosts.size(), NetShardScratch{});
  // New connections arrive on the first port and are redirected to the
  // least-loaded shard (the callback runs on shard 0's pump thread; the load
  // figures are refreshed on this thread between pumps).
  shardLoad.assign(hosts.size(), 0u);
  shardPending.assign(hosts.size(), 0u);
  if (hosts.size() > 1) {
    hosts[0]->setRedirect([this](uint64_t) -> uint16_t {
      size_t best = 0;
      for (size_t k = 1; k < hosts.size(); ++k)
        if (shardLoad[k] + shardPending[k] < shardLoad[best] + shardPending[best]) best = k;
      if (shardLoad[best] + shardPending[best] >= shardPeers) return 0; // all full: shard 0 answers
      ++shardPending[best];
      return best == 0 ? uint16_t(0) : hosts[best]->localPort();
    });
  }

  entities.clear();
  players.clear();
  spawnQueue.clear();
  editedStore.clear();
  editedOrder.clear();
  regrowths.clear();
  spawnerSlots.clear();
  tick = 0;
  nextEntityId = 1;
  nextMonsterSpawnTick = kSimHz * 2;
  msgBuf.reserve(1 << 16);
  running = true;

  world->clearFoci();
  world->addFocus(spawnPoint.x, spawnPoint.y, spawnPoint.z);
  world->update();

  std::printf("[server] listening on UDP %u (%zu port shards), %d job threads, seed %llu, max players %u, "
              "schema %016llx\n",
              unsigned(hosts[0]->localPort()), hosts.size(), jobs.threads(), (unsigned long long)cfg.seed,
              unsigned(cfg.maxPlayers),
              (unsigned long long)kSchemaHash);
  std::fflush(stdout);
  return true;
}

void ServerState::stop() {
  if (!running) return;
  running = false;
  for (auto &kv : players) netDisconnect(kv.second.peer);
  for (auto &h : hosts) h->update(net::nowMs()); // disconnect packets go out now
  players.clear();
  entities.clear();
  spawnQueue.clear();
  hosts.clear();
  jobs.resize(1);
  world.reset();
  std::printf("[server] stopped\n");
  std::fflush(stdout);
}

void ServerState::step() {
  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  auto mark = t0;
  auto phase = [&](int i) { // tick profiler: time since the previous mark goes to phase i
    const auto t = clk::now();
    phaseMs[size_t(i)] += std::chrono::duration<double, std::milli>(t - mark).count();
    mark = t;
  };
  ++tick;

  pumpNetwork();
  phase(0);

  simulatePlayers();
  phase(1);
  simulateMonsters();
  phase(2);
  simulateProjectiles();
  simulateItems();
  phase(3);
  if (tick >= nextMonsterSpawnTick) {
    spawnMonsters();
    nextMonsterSpawnTick = tick + kSimHz;
  }
  updateSpawners();
  regrowNodes();
  flushSpawns();
  removeDead();
  phase(4);
  buildGrid(); // interest grid for proximity queries (next tick) and snapshots
  phase(7);

  // One streaming focus per occupied chunk, in a stable order: VoxelWorld's
  // cost is O(chunks x foci) and it rescans whenever the list changes, so
  // 10k players must not mean 10k foci reshuffled every tick.
  keyScratch.clear();
  auto chunkKey = [](const glm::dvec3 &p) {
    const auto c = [](double v) { return uint64_t(uint32_t(int32_t(std::floor(v / vx::kChunkSize))) & 0x1FFFFFu); };
    return (c(p.x) << 42) | (c(p.y) << 21) | c(p.z);
  };
  keyScratch.push_back(chunkKey(spawnPoint)); // first joiners never wait
  for (auto &kv : players) {
    if (const Entity *e = findEntity(kv.second.entity)) keyScratch.push_back(chunkKey(e->move.pos));
  }
  std::sort(keyScratch.begin(), keyScratch.end());
  keyScratch.erase(std::unique(keyScratch.begin(), keyScratch.end()), keyScratch.end());
  world->clearFoci();
  for (uint64_t k : keyScratch) {
    const auto un = [](uint64_t v) { return double(int32_t(uint32_t(v) << 11) >> 11) * vx::kChunkSize + 0.5 * vx::kChunkSize; };
    world->addFocus(un((k >> 42) & 0x1FFFFFu), un((k >> 21) & 0x1FFFFFu), un(k & 0x1FFFFFu));
  }
  world->update();
  phase(5);

  for (auto &kv : players) {
    Player &pl = kv.second;
    if (!pl.welcomed) {
      if (tick - pl.joinTick > Tick(kSimHz * 10)) netDisconnect(pl.peer); // no Hello
      continue;
    }
    flushXp(pl);
    if (pl.inventoryDirty) sendInventory(pl);
    if (tick >= pl.nextChunkSyncTick) syncChunks(pl, false);
  }
  phase(6);

  // Every tick, for the players whose snapshot is due (staggered: each
  // player still gets kSnapshotHz, but the load is spread evenly over ticks).
  sendSnapshots();
  phase(8);

  // Flush everything queued this tick, one thread per network shard.
  const uint64_t flushMs = net::nowMs();
  jobs.run(hosts.size(), [&](size_t k, size_t) { hosts[k]->update(flushMs); });
  phase(9);
  ++profiledTicks;

  const float ms = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
  tickMs[tickMsHead] = ms;
  tickMsHead = (tickMsHead + 1) % tickMs.size();
  tickMsCount = std::min(tickMsCount + 1, tickMs.size());
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

void ServerState::pumpNetwork() {
  // Per shard, in parallel: socket reads, acks, reassembly, and the input
  // messages (the bulk of the traffic: they only touch their own Player).
  // Everything else is kept in order and handled below on this thread.
  // Event payloads stay valid until the shard's next update().
  const uint64_t ms = net::nowMs();
  jobs.run(hosts.size(), [&](size_t k, size_t) {
    NetShardScratch &sc = netScratch[k];
    sc.deferred.clear();
    hosts[k]->update(ms);
    net::Event ev;
    while (hosts[k]->poll(ev)) {
      if (ev.type == net::EventType::Message) {
        net::BitReader r(ev.data);
        uint16_t id = 0;
        if (net::peekMessageId(ev.data, id, r) && id == InputBatch::kId) {
          auto it = players.find(toGlobal(k, ev.peer).value);
          if (it != players.end() && it->second.welcomed) {
            if (net::decodeMessage(r, sc.batch)) handleInput(it->second, sc.batch);
            continue;
          }
        }
      }
      sc.deferred.push_back(ev);
    }
  });
  for (size_t k = 0; k < hosts.size(); ++k) { // redirect balancing (see start())
    shardLoad[k] = uint32_t(hosts[k]->peerCount());
    shardPending[k] = 0;
  }
  const auto tEvents = std::chrono::steady_clock::now();
  for (size_t k = 0; k < hosts.size(); ++k) {
    for (const net::Event &ev : netScratch[k].deferred) {
      const PeerId peer = toGlobal(k, ev.peer);
      switch (ev.type) {
      case net::EventType::Connected: onConnected(peer); break;
      case net::EventType::Disconnected: onDisconnected(peer); break;
      case net::EventType::Message: {
        auto it = players.find(peer.value);
        if (it != players.end()) onMessage(it->second, ev.data);
        break;
      }
      }
    }
  }
  phaseMs[15] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tEvents).count();
}

void ServerState::onConnected(PeerId peer) {
  Player &pl = players[peer.value];
  pl = Player{};
  pl.peer = peer;
  pl.joinTick = tick;
}

void ServerState::onDisconnected(PeerId peer) {
  auto it = players.find(peer.value);
  if (it == players.end()) return;
  Player &pl = it->second;
  if (Entity *e = findEntity(pl.entity)) e->remove = true;
  if (cfg.verbose && pl.welcomed) {
    std::printf("[server] %s left (%zu players)\n", pl.name.c_str(), players.size() - 1);
    std::fflush(stdout);
  }
  players.erase(it);
}

void ServerState::onMessage(Player &pl, std::span<const uint8_t> data) {
  net::BitReader r(data);
  uint16_t id = 0;
  if (!net::peekMessageId(data, id, r)) return;
  if (!pl.welcomed && id != Hello::kId) return;
  bool ok = true;
  switch (id) {
  case Hello::kId: {
    Hello m;
    ok = net::decodeMessage(r, m);
    if (ok) handleHello(pl, m);
    break;
  }
  case InputBatch::kId:
    ok = net::decodeMessage(r, inBatch);
    if (ok) handleInput(pl, inBatch);
    break;
  case BlockAction::kId: {
    BlockAction m;
    ok = net::decodeMessage(r, m);
    if (ok) handleBlockAction(pl, m);
    break;
  }
  case Attack::kId: {
    Attack m;
    ok = net::decodeMessage(r, m);
    if (ok) handleAttack(pl, m);
    break;
  }
  case Equip::kId: {
    Equip m;
    ok = net::decodeMessage(r, m);
    if (ok) handleEquip(pl, m);
    break;
  }
  case ChatSend::kId: {
    ChatSend m;
    ok = net::decodeMessage(r, m);
    if (ok) handleChat(pl, m);
    break;
  }
  case Pickup::kId: {
    Pickup m;
    ok = net::decodeMessage(r, m);
    if (ok) handlePickup(pl, m);
    break;
  }
  default:
    ok = false; // server->client ids or unknown: ignore
    break;
  }
  (void)ok; // malformed client messages are dropped (TODO(demo): rate-limit + flag)
}

namespace {

std::string sanitizeName(const std::string &in, EntityId id) {
  std::string out;
  for (char c : in) {
    if (out.size() >= 24) break;
    const unsigned char u = static_cast<unsigned char>(c);
    if (u >= 32 && u < 127) out.push_back(c);
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  if (out.empty()) out = "Player" + std::to_string(id);
  return out;
}

std::string sanitizeText(const std::string &in, size_t maxLen) {
  std::string out;
  for (char c : in) {
    if (out.size() >= maxLen) break;
    const unsigned char u = static_cast<unsigned char>(c);
    if (u >= 32 && u != 127) out.push_back(c); // keeps UTF-8 bytes, drops controls
  }
  return out;
}

} // namespace

void ServerState::handleHello(Player &pl, const Hello &m) {
  if (pl.welcomed) return;
  if (m.schemaHash != kSchemaHash) {
    std::printf("[server] rejecting client with schema %016llx\n", (unsigned long long)m.schemaHash);
    netDisconnect(pl.peer);
    return;
  }
  const EntityId id = nextEntityId++;
  pl.name = sanitizeName(m.name, id);
  pl.appearance = m.appearance; // cosmetic fields; equipment pieces come from the server
  giveStartingKit(pl);

  Entity e;
  e.id = id;
  e.kind = EntityKind::Player;
  e.playerKey = pl.peer.value;
  const int32_t sx = int32_t(std::floor(spawnPoint.x)) + randi(-3, 3);
  const int32_t sz = int32_t(std::floor(spawnPoint.z)) + randi(-3, 3);
  const int h = ao::world::groundHeight(world->generator(), sx, sz);
  e.move.pos = glm::dvec3(double(sx) + 0.5, double(h) + 1.05, double(sz) + 0.5);
  e.maxHp = maxHpFor(pl);
  e.hp = e.maxHp;
  entities.emplace(id, std::move(e));
  pl.entity = id;
  pl.welcomed = true;
  pl.regenTick = tick;
  refreshAppearance(pl);

  const Entity &pe = entities.at(id);
  Welcome w;
  w.schemaHash = kSchemaHash;
  w.playerEntity = id;
  w.serverTick = tick;
  w.worldSeed = cfg.seed;
  w.spawn = pe.move.pos;
  sendTo(pl, w, Channel::ReliableOrdered);

  sendInventory(pl);
  for (int s = 0; s < kSkillCount; ++s) {
    XpGain x;
    x.skill = uint8_t(s);
    x.amount = 0;
    x.totalXp = pl.xp[size_t(s)];
    sendTo(pl, x, Channel::ReliableOrdered);
  }
  AppearanceMsg am;
  am.entity = id;
  am.name = pl.name;
  am.appearance = pl.appearance;
  sendTo(pl, am, Channel::ReliableOrdered);
  syncChunks(pl, true);

  if (cfg.verbose) {
    std::printf("[server] %s joined as entity %u (%zu players)\n", pl.name.c_str(), unsigned(id),
                players.size());
    std::fflush(stdout);
  }
}

void ServerState::handleInput(Player &pl, const InputBatch &m) {
  // Snapshot acknowledgement (drives delta baselines).
  if (m.lastSnapshotTick <= tick && (!pl.hasSnapshotAck || m.lastSnapshotTick > pl.lastSnapshotAck))
    onSnapshotAck(pl, m.lastSnapshotTick);

  // Queue new inputs in seq order; stale/duplicate seqs are ignored.
  for (size_t guard = 0; guard < m.inputs.size(); ++guard) {
    const MoveInput *best = nullptr;
    for (const MoveInput &in : m.inputs) {
      const bool newer = !pl.anyInput || in.seq > pl.lastQueuedSeq;
      if (newer && (!best || in.seq < best->seq)) best = &in;
    }
    if (!best) break;
    MoveInput in = *best;
    in.buttons &= uint16_t(0x3F); // known button bits only
    if (pl.inputCount == kInputQueue) { // full: drop the oldest
      pl.inputHead = (pl.inputHead + 1) % kInputQueue;
      --pl.inputCount;
    }
    pl.inputs[size_t((pl.inputHead + pl.inputCount) % kInputQueue)] = in;
    ++pl.inputCount;
    pl.lastQueuedSeq = in.seq;
    pl.anyInput = true;
  }
}

void ServerState::handleBlockAction(Player &pl, const BlockAction &m) {
  Entity *pe = findEntity(pl.entity);
  if (!pe) return;
  const vx::BlockPos target{m.x, m.y, m.z};
  auto revert = [&](vx::BlockPos p) { // undo the client's optimistic edit
    if (p.y < 0 || p.y >= vx::kWorldHeight) return;
    BlockChanged bc;
    bc.x = p.x;
    bc.y = p.y;
    bc.z = p.z;
    bc.block = world->blockAt(p);
    sendTo(pl, bc, Channel::ReliableOrdered);
  };
  const bool isPlace = m.action == 1;
  vx::BlockPos dest = target;
  if (isPlace) {
    if (m.face >= vx::kFaceDirCount) return;
    dest.x += vx::kFaceNormal[m.face][0];
    dest.y += vx::kFaceNormal[m.face][1];
    dest.z += vx::kFaceNormal[m.face][2];
  } else if (m.action != 0) {
    return;
  }
  if (pe->dead || tick < pl.blockReadyTick || dest.y < 0 || dest.y >= vx::kWorldHeight ||
      target.y < 0 || target.y >= vx::kWorldHeight || !world->isLoaded(vx::chunkOf(dest)) ||
      !world->isLoaded(vx::chunkOf(target))) {
    revert(dest);
    return;
  }
  const glm::dvec3 eye = eyePosition(pe->move);
  const glm::dvec3 centre(dest.x + 0.5, dest.y + 0.5, dest.z + 0.5);
  if (glm::length(centre - eye) > double(kReach)) {
    revert(dest);
    return;
  }

  if (!isPlace && !editableWorld) { // overworld: harvest resource nodes only
    if (!harvestNode(pl, *pe, target, world->blockAt(target))) revert(target);
    return;
  }
  if (isPlace && !editableWorld) {
    revert(dest);
    return;
  }

  if (!isPlace) {
    const vx::BlockId old = world->blockAt(target);
    const vx::BlockDef &def = blocks.get(old);
    if (old == vx::kAir || old == vx::blocks::Bedrock || def.hardness < 0.0f || def.liquid ||
        skillLevel(pl, Skill::Mining) < def.miningLevel) {
      revert(target);
      return;
    }
    if (!world->setBlock(target, vx::kAir)) {
      revert(target);
      return;
    }
    pl.blockReadyTick = tick + 3; // ~0.1 s between edits (TODO(demo): hardness-based break time)
    startAction(*pe, action::Mine);
    onBlockEdited(target, vx::kAir);
    if (const ItemId drop = blockDropItem(old))
      giveOrDrop(pl, drop, 1, glm::dvec3(target.x + 0.5, target.y + 0.5, target.z + 0.5), kNoEntity, false);
    addXp(pl, Skill::Mining, double(def.miningXp));
    return;
  }

  // Place
  const int slot = m.hotbarSlot;
  if (slot >= kHotbarSlots || slot >= kInventorySlots) {
    revert(dest);
    return;
  }
  const ItemStack st = pl.inv[size_t(slot)];
  const ItemDef &idef = itemDef(st.item);
  const vx::BlockId aimed = world->blockAt(target);
  const vx::BlockId existing = world->blockAt(dest);
  if (st.item == 0 || st.count == 0 || idef.kind != ItemKind::Block || idef.placesBlock == vx::kAir ||
      aimed == vx::kAir || blocks.solid(existing) || existing == vx::blocks::Bedrock) {
    revert(dest);
    return;
  }
  // Must not intersect any player or monster.
  const float hw = moveTuning().halfWidth, ht = moveTuning().height;
  for (const auto &[eid, e] : entities) {
    if (e.dead || (e.kind != EntityKind::Player && e.kind != EntityKind::Monster)) continue;
    const glm::dvec3 &p = e.move.pos;
    if (p.x + hw > dest.x && p.x - hw < dest.x + 1 && p.z + hw > dest.z && p.z - hw < dest.z + 1 &&
        p.y + ht > dest.y && p.y < dest.y + 1) {
      revert(dest);
      return;
    }
  }
  if (!world->setBlock(dest, idef.placesBlock)) {
    revert(dest);
    return;
  }
  removeFromSlot(pl, slot, 1);
  pl.blockReadyTick = tick + 3;
  startAction(*pe, action::Place);
  onBlockEdited(dest, idef.placesBlock);
  addXp(pl, Skill::Construction, 2.0);
}

void ServerState::handleAttack(Player &pl, const Attack &m) {
  Entity *pe = findEntity(pl.entity);
  // Two ticks of grace absorb network jitter; the next ready tick counts from
  // the scheduled one, so early arrivals cannot raise the attack rate.
  if (!pe || pe->dead || tick + 2 < pl.attackReadyTick) return;
  const ItemId weaponItem = heldWeapon(pl);
  const WeaponType weapon = weaponItem ? itemDef(weaponItem).weapon : WeaponType::None;
  const float pitch = std::clamp(m.pitch, -1.55f, 1.55f);
  const glm::vec3 dir = aimDirection(m.yaw, pitch);

  const Tick cooldownTicks = Tick(std::lround(weaponCooldown(weapon) * kSimHz));
  pl.attackReadyTick = std::max(tick, pl.attackReadyTick) + cooldownTicks;

  if (weapon == WeaponType::Bow || weapon == WeaponType::Staff) {
    rangedAttack(pl, *pe, dir, weapon);
    return;
  }
  Skill style = Skill::Attack;
  if (m.ability == 1) style = Skill::Strength;
  else if (m.ability == 2) style = Skill::Defence;
  meleeAttack(pl, *pe, dir, style, Tick(m.tick));
}

void ServerState::handleEquip(Player &pl, const Equip &m) {
  if (m.equipSlot == kEquipSelectHotbar) { // hold the selected hotbar item
    const uint8_t slot = m.inventorySlot < 9 ? m.inventorySlot : uint8_t(0xFF);
    if (slot != pl.heldSlot) {
      pl.heldSlot = slot;
      refreshAppearance(pl);
    }
    return;
  }
  if (m.equipSlot == kEquipConsume) {
    consumeItem(pl, m.inventorySlot);
    return;
  }
  if (m.equipSlot >= atm::model::kEquipSlotCount) return;
  if (m.unequip) {
    ItemId &slotItem = pl.equipped[m.equipSlot];
    if (slotItem == 0) return;
    if (addItem(pl, slotItem, 1) != 0) { // inventory full
      pl.inventoryDirty = true;
      return;
    }
    slotItem = 0;
  } else {
    if (m.inventorySlot >= kInventorySlots) return;
    ItemStack &st = pl.inv[m.inventorySlot];
    if (st.item == 0 || st.count == 0) return;
    const ItemDef &d = itemDef(st.item);
    if (d.kind != ItemKind::Weapon && d.kind != ItemKind::Tool && d.kind != ItemKind::Armour) return;
    if (skillLevel(pl, d.skill) < d.levelReq) {
      ChatMsg cm;
      cm.from = "Server";
      cm.text = "You need " + std::string(skillName(d.skill)) + " level " + std::to_string(d.levelReq) +
                " to equip that.";
      sendTo(pl, cm, Channel::ReliableOrdered);
      return;
    }
    const size_t slot = size_t(d.slot); // the item decides its slot
    const ItemId previous = pl.equipped[slot];
    pl.equipped[slot] = st.item;
    if (st.count > 1) {
      --st.count;
      if (previous && addItem(pl, previous, 1) != 0) { // no room: undo
        pl.equipped[slot] = previous;
        ++st.count;
        return;
      }
    } else {
      st.item = previous; // swap into the same inventory slot
      st.count = previous ? 1 : 0;
    }
  }
  pl.inventoryDirty = true;
  refreshAppearance(pl);
  if (Entity *pe = findEntity(pl.entity)) {
    pe->maxHp = maxHpFor(pl);
    pe->hp = std::min(pe->hp, pe->maxHp);
  }
}

void ServerState::handleChat(Player &pl, const ChatSend &m) {
  if (tick < pl.chatReadyTick) return;
  pl.chatReadyTick = tick + kSimHz / 2;
  const std::string text = sanitizeText(m.text, 200);
  if (text.empty()) return;
  ChatMsg cm;
  cm.from = pl.name;
  cm.text = text;
  sendAll(cm, Channel::ReliableOrdered);
  if (cfg.verbose) {
    std::printf("[chat] %s: %s\n", pl.name.c_str(), text.c_str());
    std::fflush(stdout);
  }
}

} // namespace ao::server
