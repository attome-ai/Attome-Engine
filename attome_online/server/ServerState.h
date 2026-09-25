#pragma once

// Internal state of the zone server (not part of the public API). The
// implementation is split by topic:
//   ZoneServer.cpp          lifecycle, network pump, message handlers
//   ServerGameplay.cpp      players, monsters, combat, loot, inventory, skills
//   ServerReplication.cpp   interest grid, priority, delta snapshots, chunk sync

#include "ZoneServer.h"

#include "../shared/GameTypes.h"
#include "../shared/Movement.h"
#include "../shared/Protocol.h"
#include "../shared/Snapshot.h"

#include "../../engine/model/Character.h"
#include "../../engine/net2/Net.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/voxel/VoxelWorld.h"

#include <array>
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
  std::unordered_map<EntityId, RelTrack> rel;
  std::vector<std::pair<EntityId, Tick>> pendingRemoved; // id, first sent
  std::array<SnapRecord, kSnapRecords> records;

  // Edited-chunk sync
  std::unordered_set<uint64_t> announced;                 // told via EditedChunks
  std::unordered_map<uint64_t, uint64_t> chunkSentMs;     // ChunkData queued (ms)
  Tick nextChunkSyncTick = 0;
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

  // Monsters
  EntityId target = kNoEntity;
  Tick attackReadyTick = 0, wanderUntil = 0, despawnTick = 0;
  glm::dvec3 wanderTarget{0.0};
  float stuckTime = 0.0f;
  std::vector<std::pair<EntityId, uint32_t>> damageBy; // player entity -> damage dealt
  Tick pendingHitTick = 0;  // telegraphed attack lands at this tick (0 = none)
  Tick staggerUntil = 0;    // hit reaction: no moving / attacking until then
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

struct ServerState {
  ServerConfig cfg;
  bool running = false;
  std::unique_ptr<atm::net2::Host> host;
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
  double snapshotAccum = 0.0;

  // Interest grid (cell -> entities), rebuilt per snapshot pass.
  std::unordered_map<uint64_t, std::vector<EntityId>> grid;

  // Scratch (reused; no per-tick allocation once warmed up)
  std::vector<uint8_t> msgBuf;
  std::vector<uint8_t> measureBuf;
  proto::InputBatch inBatch;
  proto::SnapshotMsg snapMsg;
  std::vector<std::pair<float, EntityId>> candidates;
  std::vector<uint64_t> keyScratch;
  std::vector<EntityId> idScratch;

  // Stats
  std::array<float, 300> tickMs{};
  size_t tickMsCount = 0, tickMsHead = 0;
  uint32_t monsterCount = 0;

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
  void handleChat(Player &pl, const proto::ChatSend &m);

  template <class M> bool sendTo(Player &pl, const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return false;
    return host->send(pl.peer, channel, msgBuf);
  }
  // Sends to every welcomed player within `radius` blocks (horizontal) of `pos`.
  template <class M> void sendNear(const glm::dvec3 &pos, double radius, const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return;
    for (auto &[key, pl] : players) {
      if (!pl.welcomed) continue;
      const Entity *e = findEntity(pl.entity);
      if (!e) continue;
      const double dx = e->move.pos.x - pos.x, dz = e->move.pos.z - pos.z;
      if (dx * dx + dz * dz <= radius * radius) host->send(pl.peer, channel, msgBuf);
    }
  }
  template <class M> void sendAll(const M &msg, Channel channel) {
    atm::net2::encodeMessage(msgBuf, msg);
    if (msgBuf.empty()) return;
    for (auto &[key, pl] : players)
      if (pl.welcomed) host->send(pl.peer, channel, msgBuf);
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

  // ---- ServerReplication.cpp ----
  void buildGrid();
  void sendSnapshots();
  void buildSnapshot(Player &pl, Entity &self);
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
