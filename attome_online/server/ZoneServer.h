#pragma once

// Authoritative zone server (docs/DEMO_PLAN.md, GAME_DESIGN §8-11, §25).
// Used by ao_server (server_main.cpp) and in-process by the client's --local
// mode. Single-threaded: start(), then tick() at kSimHz or run() on a thread
// of its own (the voxel world's generation workers run in the background).

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace ao::server {

struct ServerConfig {
  uint16_t port = 27015;           // 0 = any free port (see ZoneServer::port())
  uint64_t seed = 12345;
  uint16_t maxPlayers = 256;
  int viewRadiusChunks = 8;        // client view radius: edited-chunk sync + block edits
  int simRadiusChunks = 3;         // chunks the server keeps loaded around each player
  int workerThreads = 0;           // voxel generation workers (0 = auto)
  // Scaling (10k+ players): UDP port shards (ports port..port+netShards-1,
  // each pumped on its own thread; clients may connect to any of them) and
  // worker threads for the parallel tick phases (0 = auto).
  int netShards = 1;
  int jobThreads = 1;
  int monstersPerPlayer = 6;
  int maxMonsters = 300;
  uint32_t bandwidthBytesPerSec = 256u * 1024u; // per client
  uint32_t snapshotBudgetBytes = 1100;          // one unreliable packet
  // Interest cap: each client tracks at most this many entities (nearest
  // first). Keeps replication O(players * cap) instead of O(players^2) when
  // crowds gather.
  int maxRelevantEntities = 128;
  uint32_t timeoutMs = 10000;
  bool verbose = true;             // log joins/leaves

  // Reads a JSON config (engine/ATMJson.h). Missing keys keep defaults; a
  // missing or malformed file returns the defaults and sets *error.
  static ServerConfig load(const std::string &path, std::string *error = nullptr);
};

struct ServerStats {
  uint32_t players = 0, entities = 0, monsters = 0;
  uint32_t tick = 0;
  double tickMsAvg = 0.0, tickMsP99 = 0.0, tickMsMax = 0.0; // over the last ~10 s
  uint64_t bytesSent = 0, bytesReceived = 0;                // totals
  uint64_t malformedPackets = 0;
  size_t loadedChunks = 0, editedChunks = 0;
  // Tick profiler: total ms spent per phase since start (diff two samples).
  static constexpr int kPhaseCount = 17;
  static constexpr const char *kPhaseNames[kPhaseCount] = {
      "netIn", "players", "monsters", "otherSim", "spawn", "world", "perPlayer", "grid", "snapshots", "netOut",
      // inside "snapshots":
      "snap.scan", "snap.appearance", "snap.removed", "snap.select", "snap.send", // snap.*: wall ms (CPU / threads)
      "netEvents", // inside netIn: handling received messages (serial)
      "snap.track"}; // re-ranking the tracked set + updating per-entity state
  // Work counters (totals): candidate entities scanned, appearance messages, entities sent.
  uint64_t snapScanned = 0, snapAppearance = 0, snapEntities = 0, relTracks = 0;
  std::array<double, kPhaseCount> phaseMsTotal{};
  uint64_t profiledTicks = 0;
};

struct ServerState; // internal (ServerState.h)

class ZoneServer {
public:
  ZoneServer();
  ~ZoneServer();
  ZoneServer(const ZoneServer &) = delete;
  ZoneServer &operator=(const ZoneServer &) = delete;

  bool start(const ServerConfig &config, std::string *error = nullptr);
  // One fixed simulation step (network in, simulation, snapshots, network out).
  void tick();
  // Runs tick() at kSimHz until `stop` becomes true (then calls stop()).
  void run(std::atomic<bool> &stop);
  // Disconnects everyone and releases the socket and the world.
  void stop();

  bool running() const;
  uint16_t port() const;        // bound UDP port (valid after start)
  ServerStats stats() const;

private:
  std::unique_ptr<ServerState> s_;
};

} // namespace ao::server
