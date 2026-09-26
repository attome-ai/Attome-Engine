#pragma once

// Internal state of the zone server (not part of the public API). The
// implementation is split by topic:
//   ZoneServer.cpp          lifecycle, network pump, message handlers
//   ServerGameplay.cpp      players, monsters, combat, loot, inventory, skills
//   ServerReplication.cpp   interest grid, priority, delta snapshots, chunk sync

#include "JobPool.h"
#include "ZoneServer.h"

#include "../shared/GameTypes.h"
#include "../shared/Movement.h"
#include "../shared/Protocol.h"
#include "../shared/Snapshot.h"
#include "../shared/world/Town.h"

#include "../../engine/model/Character.h"
#include "../../engine/net2/Net.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/voxel/VoxelWorld.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ao::server {

using atm::net2::Channel;
using atm::net2::PeerId;

inline constexpr int kFieldCount = 7;          // field::Pos .. field::Type
inline constexpr int kSnapRecords = 32;        // per-client snapshot history (acks)
inline constexpr int kInputQueue = 16;         // queued inputs per player
inline constexpr int kHotbarSlots = 9;         // inventory slots 0..8 are the hotbar
inline constexpr double kGridCell = 16.0;      // interest grid cell (blocks)
inline constexpr double kNearRange = 32.0, kMidRange = 96.0, kFarRange = 160.0;
inline constexpr double kEventRange = 64.0;    // DamageEvent broadcast radius
inline uint64_t gridCellKey(int32_t cx, int32_t cz) { return (uint64_t(uint32_t(cx)) << 32) | uint32_t(cz); }
inline int32_t gridCellOf(double v) { return int32_t(std::floor(v / kGridCell)); }
inline constexpr float kReach = 6.5f;          // block edit reach from the eye

// Quantised replicated state of one entity (what the client ends up with).
struct Quant {
  int32_t px = 0, pz = 0;
  uint16_t py = 0;
  uint16_t vx = 0, vy = 0, vz = 0;
  uint16_t yaw = 0;
  uint8_t loco = 0, action = 0, actionSeq = 0;
  uint16_t hp = 0, maxHp = 0;
  uint8_t flags = 0;
  uint8_t kind = 0, type = 0;
  uint16_t item = 0;
};

// Per (client, entity) replication state.
struct RelTrack {
  Quant last;                                // values last included
  std::array<Tick, kFieldCount> changeTick{}; // snapshot where `last` values were first sent
  uint8_t confirmed = 0;                     // fields the client provably has
  uint8_t sentMask = 0;                      // fields ever sent
  float priority = 0.0f;
  uint32_t appearanceVersion = 0;            // players: appearance sent to this client
  Tick seenTick = 0;                         // last snapshot pass it was relevant
};

// Per-client EntityId -> RelTrack map: a small open-addressing index (8-byte
// slots, backward-shift deletion) over a dense record array with a free
// list. Contiguous and compact, so the snapshot pass stays in cache instead
// of chasing unordered_map nodes. A RelTrack pointer stays valid until the
// next tryEmplace (the record array may grow).
class RelMap {
public:
  size_t size() const { return size_; }
  RelTrack *find(EntityId id) {
    if (index_.empty()) return nullptr;
    for (size_t i = home(id);; i = (i + 1) & mask_) {
      if (index_[i].first == id) return &items_[index_[i].second];
      if (index_[i].first == kNoEntity) return nullptr;
    }
  }
  bool contains(EntityId id) const { return const_cast<RelMap *>(this)->find(id) != nullptr; }
  std::pair<RelTrack *, bool> tryEmplace(EntityId id) {
    if ((size_ + 1) * 2 > index_.size()) grow();
    size_t i = home(id);
    for (;; i = (i + 1) & mask_) {
      if (index_[i].first == id) return {&items_[index_[i].second], false};
      if (index_[i].first == kNoEntity) break;
    }
    uint32_t slot;
    if (!free_.empty()) {
      slot = free_.back();
      free_.pop_back();
      items_[slot] = RelTrack{};
      ids_[slot] = id;
    } else {
      slot = uint32_t(items_.size());
      items_.emplace_back();
      ids_.push_back(id);
    }
    index_[i] = {id, slot};
    ++size_;
    return {&items_[slot], true};
  }
  template <class F> void forEach(F &&fn) {
    for (size_t s = 0; s < items_.size(); ++s)
      if (ids_[s] != kNoEntity) fn(ids_[s], items_[s]);
  }
  // Removes entries where pred(id, rt) is true.
  template <class P> void eraseIf(P &&pred) {
    for (size_t s = 0; s < items_.size(); ++s)
      if (ids_[s] != kNoEntity && pred(ids_[s], items_[s])) erase(ids_[s]);
  }

private:
  size_t home(EntityId id) const { return size_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 32) & mask_; }
  void erase(EntityId id) {
    size_t i = home(id);
    for (;; i = (i + 1) & mask_) {
      if (index_[i].first == kNoEntity) return;
      if (index_[i].first == id) break;
    }
    const uint32_t slot = index_[i].second;
    ids_[slot] = kNoEntity;
    free_.push_back(slot);
    --size_;
    for (size_t j = i;;) { // backward-shift deletion: no tombstones
      j = (j + 1) & mask_;
      if (index_[j].first == kNoEntity) break;
      const size_t h = home(index_[j].first);
      if ((i <= j) ? (h > i && h <= j) : (h > i || h <= j)) continue; // already on its probe path
      index_[i] = index_[j];
      i = j;
    }
    index_[i] = {kNoEntity, 0u};
  }
  void grow() {
    const size_t cap = std::max<size_t>(64, index_.size() * 2);
    index_.assign(cap, {kNoEntity, 0u});
    mask_ = cap - 1;
    for (size_t s = 0; s < items_.size(); ++s) {
      if (ids_[s] == kNoEntity) continue;
      for (size_t i = home(ids_[s]);; i = (i + 1) & mask_) {
        if (index_[i].first == kNoEntity) {
          index_[i] = {ids_[s], uint32_t(s)};
          break;
        }
      }
    }
  }
  std::vector<std::pair<EntityId, uint32_t>> index_;
  std::vector<RelTrack> items_;
  std::vector<EntityId> ids_; // per record; kNoEntity = free
  std::vector<uint32_t> free_;
  size_t mask_ = 0, size_ = 0;
};

struct SnapRecord {
  Tick tick = 0;
  bool valid = false;
  std::vector<std::pair<EntityId, uint16_t>> included; // entity, field mask sent
};

struct Player {
  PeerId peer;
  EntityId entity = kNoEntity;
  bool welcomed = false;
  std::string name;
  atm::model::Appearance appearance;
  uint32_t appearanceVersion = 1;

  std::array<ItemStack, kInventorySlots> inv{};
  std::array<ItemId, atm::model::kEquipSlotCount> equipped{};
  std::array<uint64_t, kSkillCount> xp{};
  std::array<uint64_t, kSkillCount> xpGained{}; // this tick (sent as XpGain)
  std::array<double, kSkillCount> xpFraction{};
  bool inventoryDirty = false;

  // Input (ring buffer, oldest first)
  std::array<MoveInput, kInputQueue> inputs{};
  int inputHead = 0, inputCount = 0;
  bool anyInput = false;
  uint32_t lastQueuedSeq = 0, lastAppliedSeq = 0;
  float inputCredit = 0.0f;
  uint16_t lastButtons = 0;

  // Timers (server ticks)
  Tick attackReadyTick = 0, blockReadyTick = 0, chatReadyTick = 0;
  Tick lastHurtTick = 0, respawnTick = 0, regenTick = 0;
  Tick joinTick = 0;
  Skill lastStyle = Skill::Attack;
  uint8_t heldSlot = 0xFF;     // selected hotbar slot (0..8), 0xFF = none
  Tick eatReadyTick = 0;

  // Replication
  Tick lastSnapshotAck = 0;
  bool hasSnapshotAck = false;
  uint32_t snapshotsSent = 0;
  RelMap rel;
  std::vector<std::pair<EntityId, Tick>> pendingRemoved; // id, first sent
  std::array<SnapRecord, kSnapRecords> records;

  // Edited-chunk sync
  std::unordered_set<uint64_t> announced;                 // told via EditedChunks
  std::unordered_map<uint64_t, uint64_t> chunkSentMs;     // ChunkData queued (ms)
  Tick nextChunkSyncTick = 0;

  // Outbox filled by the parallel snapshot pass, sent per network shard:
  // the snapshot, and AppearanceMsgs (bytes + the entity/version each one
  // carries, recorded in `rel` once the send is accepted).
  std::vector<uint8_t> outSnapshot;
  std::vector<std::vector<uint8_t>> outAppearance;
  std::vector<std::pair<EntityId, uint32_t>> outAppearanceFor;
};

struct Entity {
  EntityId id = kNoEntity;
  EntityKind kind = EntityKind::Player;
  uint8_t type = 0;       // monster type / projectile weapon type / item id low byte
  uint16_t item = 0;      // dropped item id
  MoveState move;
  uint16_t hp = 0, maxHp = 0;
  uint8_t actionAnim = action::None, actionSeq = 0;
  bool dead = false;
  bool remove = false;    // despawn at the end of the tick
  uint32_t playerKey = 0xFFFFFFFFu; // players: PeerId value
  uint32_t gridIndex = 0xFFFFFFFFu; // dense index from the last buildGrid() (0xFFFFFFFF = not in it)
  uint32_t appearanceVersion = 0; // players: mirror of Player::appearanceVersion

  // Monsters
  EntityId target = kNoEntity;
  Tick attackReadyTick = 0, wanderUntil = 0, despawnTick = 0;
  glm::dvec3 wanderTarget{0.0};
  float stuckTime = 0.0f;
  std::vector<std::pair<EntityId, uint32_t>> damageBy; // player entity -> damage dealt
  Tick pendingHitTick = 0;  // telegraphed attack lands at this tick (0 = none)
  Tick staggerUntil = 0;    // hit reaction: no moving / attacking until then
  int16_t spawner = -1;     // index into mapSpawners() (-1 = ambient spawn)
  glm::dvec3 home{0.0};     // spawner monsters: centre of their area (leash)
  float leash = 0.0f;       // spawner monsters: max distance from home
  // Lag compensation: position at tick t is posHistory[t % kPosHistory].
  static constexpr Tick kPosHistory = 32;
  std::array<glm::dvec3, kPosHistory> posHistory{};
  Tick historySince = 0;    // first tick recorded (older ticks use the oldest)
  bool historyValid = false;

  // Projectiles / dropped items
  EntityId owner = kNoEntity;
  uint16_t damage = 0;
  bool critical = false;
  Skill style = Skill::Ranged;
  Tick expireTick = 0;
  float gravity = 0.0f;
  ItemStack stack;
  Tick publicTick = 0;    // dropped items: anyone may pick up after this tick
};

// Interest-grid record: what proximity scans need without touching the
// (large) Entity. Entity pointers are stable: unordered_map nodes don't move,
// and entities are only erased in removeDead(), before the grid is rebuilt.
struct GridRef {
  float x = 0.0f, z = 0.0f;
  uint32_t index = 0; // dense index (Entity::gridIndex)
  EntityId id = kNoEntity;
  uint32_t appearanceVersion = 0; // players (Entity::appearanceVersion)
  EntityKind kind = EntityKind::Player;
  Entity *e = nullptr;
};

struct SnapCandidate {
  float priority = 0.0f;
  Entity *e = nullptr;
};

// Per network shard scratch for the parallel receive pass.
struct NetShardScratch {
  std::vector<atm::net2::Event> deferred; // events handled serially, in order
  proto::InputBatch batch;
};

// Per-thread scratch and counters for the parallel snapshot pass.
struct SnapWorker {
  std::vector<SnapCandidate> candidates;
  std::vector<std::pair<double, uint32_t>> nearby; // (ranking distance², grid index)
  std::vector<uint32_t> stamp; // per grid index: == stampGen when tracked by the current client
  uint32_t stampGen = 0;
  std::vector<uint8_t> measureBuf, msgBuf;
  proto::SnapshotMsg snapMsg;
  uint64_t scanned = 0, appearance = 0, entities = 0;
  std::array<double, 5> phaseMs{}; // snap.* sub-phases (ServerStats::kPhaseNames[10..14])
};

// Resource-node regrowth: the blocks a depleted node changed, restored at `at`.
struct Regrowth {
  Tick at = 0;
  std::vector<std::pair<atm::voxel::BlockPos, atm::voxel::BlockId>> blocks;
};

// One NPC slot of a fixed spawner (mapSpawners()).
struct SpawnerSlot {
  EntityId id = kNoEntity; // alive (or queued) NPC
  Tick readyTick = 0;      // empty slot respawns from this tick
};

struct ServerState {
  ServerConfig cfg;
  bool running = false;
  // Overworld rules: players can't break or place blocks, only harvest
  // resource nodes. True for future instances / player and clan plots.
  bool editableWorld = false;
  std::vector<Regrowth> regrowths;
  std::vector<std::vector<SpawnerSlot>> spawnerSlots; // per mapSpawners() entry
  // Network shards: hosts[k] listens on cfg.port + k. Player PeerIds are
  // global: the peer index is offset by k * shardPeers (see toGlobal()).
  std::vector<std::unique_ptr<atm::net2::Host>> hosts;
  uint16_t shardPeers = 0;
  JobPool jobs;
  std::vector<SnapWorker> snapWorkers;
  std::vector<std::vector<Player *>> shardPlayers; // scratch: welcomed players per shard
  std::vector<Player *> snapshotOrder;              // scratch: players to build snapshots for
  std::vector<Player *> simOrder;                   // scratch: players to simulate
  std::vector<NetShardScratch> netScratch;       // per shard
  std::vector<uint32_t> shardLoad, shardPending;  // per shard: peers, redirects since the last pump
  atm::voxel::BlockRegistry blocks;
  std::unique_ptr<atm::voxel::VoxelWorld> world;
  atm::model::ModelLibrary models;
  std::vector<atm::model::PieceId> itemPiece; // ItemId -> equipment piece

  Tick tick = 0;
  glm::dvec3 spawnPoint{0.0};
  std::mt19937_64 rng;

  std::unordered_map<EntityId, Entity> entities;
  std::vector<Entity> spawnQueue;
  EntityId nextEntityId = 1;
  std::unordered_map<uint32_t, Player> players; // key: PeerId value
  std::unordered_map<uint64_t, std::shared_ptr<const atm::voxel::Chunk>> editedStore;
  std::vector<uint64_t> editedOrder;
  Tick nextMonsterSpawnTick = 0;

  // Interest grid (cell -> entities), rebuilt every tick after removeDead().
  std::unordered_map<uint64_t, std::vector<GridRef>> grid;
  uint32_t gridCount = 0; // entities in the grid (dense indices 0..gridCount-1)
  std::vector<GridRef> gridRefs; // by dense index
  // EntityId -> dense index (open addressing, rebuilt with the grid): lock-free
  // reads for the parallel snapshot pass, no Entity cache misses.
  std::vector<std::pair<EntityId, uint32_t>> gridIds;
  uint32_t gridLookup(EntityId id) const {
    if (gridIds.empty()) return 0xFFFFFFFFu;
    const size_t mask = gridIds.size() - 1;
    for (size_t i = size_t(uint64_t(id) * 0x9E3779B97F4A7C15ull >> 40) & mask;; i = (i + 1) & mask) {
      if (gridIds[i].first == id) return gridIds[i].second;
      if (gridIds[i].first == kNoEntity) return 0xFFFFFFFFu;
    }
  }

  // Scratch (reused; no per-tick allocation once warmed up)
  std::vector<uint8_t> msgBuf;
  proto::InputBatch inBatch;
  std::vector<std::pair<float, EntityId>> candidates;
  std::vector<uint64_t> keyScratch;
  std::vector<EntityId> idScratch;

  // Stats
  std::array<float, 300> tickMs{};
  size_t tickMsCount = 0, tickMsHead = 0;
  uint32_t monsterCount = 0;
  std::array<double, ServerStats::kPhaseCount> phaseMs{};
  uint64_t profiledTicks = 0;
  uint64_t snapScanned = 0, snapAppearance = 0, snapEntities = 0;

  // ---- ZoneServer.cpp ----
  bool start(const ServerConfig &config, std::string *error);
  void stop();
  void step();
  void pumpNetwork();
  void onConnected(PeerId peer);
  void onDisconnected(PeerId peer);
  void onMessage(Player &pl, std::span<const uint8_t> data);
  void handleHello(Player &pl, const proto::Hello &m);
  void handleInput(Player &pl, const proto::InputBatch &m);
  void handleBlockAction(Player &pl, const proto::BlockAction &m);
  void handleAttack(Player &pl, const proto::Attack &m);
  void handleEquip(Player &pl, const proto::Equip &m);
  void handlePickup(Player &pl, const proto::Pickup &m);
  void handleChat(Player &pl, const proto::ChatSend &m);

  // ---- network shards (PeerIds held by the game are global) ----
  PeerId toGlobal(size_t shard, PeerId local) const {
    PeerId g;
    g.value = (local.value & 0xFFFF0000u) | uint32_t(shard * shardPeers + local.index());
    return g;
  }
  size_t shardOf(PeerId g) const { return shardPeers ? g.index() / shardPeers : 0; }
  PeerId toLocal(PeerId g) const {
    PeerId l;
    l.value = (g.value & 0xFFFF0000u) | uint32_t(g.index() % std::max<uint16_t>(shardPeers, 1));
    return l;
  }
  bool netSend(PeerId g, Channel channel, std::span<const uint8_t> data) {
    const size_t k = shardOf(g);
    return k < hosts.size() && hosts[k]->send(toLocal(g), channel, data);
  }
  void netDisconnect(PeerId g) {
    if (const size_t k = shardOf(g); k < hosts.size()) hosts[k]->disconnect(toLocal(g));
  }
  atm::net2::PeerStats netStats(PeerId g) const {
    const size_t k = shardOf(g);
    return k < hosts.size() ? hosts[k]->stats(toLocal(g)) : atm::net2::PeerStats{};
  }

  template <class M> bool sendTo(Player &pl, const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return false;
    return netSend(pl.peer, channel, msgBuf);
  }
  // Calls fn(Entity &) for every live entity within `radius` blocks
  // (horizontal) of `pos`, using the interest grid (rebuilt every tick, so
  // entities spawned this tick are not in it yet). fn returns false to stop.
  template <class F> void forEachNear(const glm::dvec3 &pos, double radius, F &&fn) {
    const int32_t x0 = gridCellOf(pos.x - radius), x1 = gridCellOf(pos.x + radius);
    const int32_t z0 = gridCellOf(pos.z - radius), z1 = gridCellOf(pos.z + radius);
    const double r2 = radius * radius;
    for (int32_t cz = z0; cz <= z1; ++cz) {
      for (int32_t cx = x0; cx <= x1; ++cx) {
        auto cit = grid.find(gridCellKey(cx, cz));
        if (cit == grid.end()) continue;
        for (const GridRef &g : cit->second) {
          const double dx = double(g.x) - pos.x, dz = double(g.z) - pos.z;
          if (dx * dx + dz * dz > r2 || g.e->remove) continue;
          if (!fn(*g.e)) return;
        }
      }
    }
  }
  // Sends to every welcomed player within `radius` blocks (horizontal) of `pos`.
  template <class M> void sendNear(const glm::dvec3 &pos, double radius, const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return;
    forEachNear(pos, radius, [&](Entity &e) {
      if (e.kind == EntityKind::Player)
        if (Player *pl = playerOf(e); pl && pl->welcomed) netSend(pl->peer, channel, msgBuf);
      return true;
    });
  }
  template <class M> void sendAll(const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return;
    for (auto &[key, pl] : players)
      if (pl.welcomed) netSend(pl.peer, channel, msgBuf);
  }

  // ---- ServerGameplay.cpp ----
  Entity *findEntity(EntityId id);
  Player *playerOf(const Entity &e);
  Entity &queueSpawn(EntityKind kind);
  void flushSpawns();
  void removeDead();
  void startAction(Entity &e, uint8_t anim);

  void simulatePlayers();
  void simulateMonsters();
  void simulateProjectiles();
  void simulateItems();
  void spawnMonsters();
  bool findGround(int32_t x, int32_t z, int32_t &groundY) const;

  uint32_t skillLevel(const Player &pl, Skill s) const;
  void addXp(Player &pl, Skill s, double amount);
  void flushXp(Player &pl);
  uint16_t maxHpFor(const Player &pl) const;
  uint16_t armourOf(const Player &pl) const;

  uint16_t addItem(Player &pl, ItemId item, uint16_t count); // returns leftover
  bool removeFromSlot(Player &pl, int slot, uint16_t count);
  void giveOrDrop(Player &pl, ItemId item, uint16_t count, const glm::dvec3 &at, EntityId from, bool rare);
  // Monster loot: pops out onto the ground (personal to `pl` for 60 s).
  void dropLoot(Player &pl, ItemId item, uint16_t count, const glm::dvec3 &at, EntityId from, bool rare);
  void sendInventory(Player &pl);
  void refreshAppearance(Player &pl);
  void giveStartingKit(Player &pl);

  uint16_t rollDamage(const Player &pl, WeaponType weapon, Skill style, bool &critical);
  void damageMonster(Entity &m, Player &attacker, uint16_t amount, bool critical, Skill style);
  void killMonster(Entity &m);
  void damagePlayer(Entity &victim, uint16_t amount, EntityId source);
  void meleeAttack(Player &pl, Entity &pe, const glm::vec3 &dir, Skill style, Tick viewTick);
  glm::dvec3 historicPos(const Entity &e, Tick at) const; // lag compensation
  ItemId heldWeapon(const Player &pl) const; // selected hotbar weapon/tool, else main hand
  void consumeItem(Player &pl, uint8_t slot);
  void rangedAttack(Player &pl, Entity &pe, const glm::vec3 &dir, WeaponType weapon);
  float randf(); // [0,1)
  int randi(int lo, int hi); // inclusive

  // ---- ServerWorld.cpp ----
  // Harvests a resource node (tool, level, gather speed checked); false if refused.
  bool harvestNode(Player &pl, Entity &pe, atm::voxel::BlockPos p, atm::voxel::BlockId block);
  void depleteNode(const ResourceDef &r, atm::voxel::BlockPos p);
  void regrowNodes();
  void updateSpawners();

  // ---- ServerReplication.cpp ----
  void buildGrid();
  void sendSnapshots();
  void buildSnapshot(Player &pl, Entity &self, SnapWorker &w); // thread-safe per player (outbox)
  void flushOutbox(Player &pl);
  void onSnapshotAck(Player &pl, Tick ackTick);
  void syncChunks(Player &pl, bool force);
  void onBlockEdited(atm::voxel::BlockPos p, atm::voxel::BlockId id);
  Quant quantize(const Entity &e) const;
};

// Aim direction for yaw/pitch (GAME_DESIGN convention: yaw 0 faces -Z,
// positive yaw counter-clockwise from above, +Y up).
glm::vec3 aimDirection(float yaw, float pitch);
glm::dvec3 eyePosition(const MoveState &m);

} // namespace ao::server
