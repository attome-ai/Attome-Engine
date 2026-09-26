// ao_bots: headless bot clients for load tests (NETWORK_PLAN §8).
//
//   ao_bots --host 127.0.0.1 --port 27015 --count 200 --behaviour wander|mine|fight
//           [--ramp 50] [--duration 0]
//
// All bots run on one thread (no per-bot threads): each bot is an
// atm::net2::Host client with a small memory footprint. Movement is predicted
// locally with ao::stepMovement against a flat approximation of the terrain
// (heightmap from WorldGenerator::surfaceHeight, shared by all bots) and
// corrected from the server's SelfState. Linux: raise `ulimit -n` for more
// than ~1000 bots (one UDP socket per bot).

#include "../shared/GameTypes.h"
#include "../shared/Movement.h"
#include "../shared/Protocol.h"
#include "../shared/Snapshot.h"

#include "../../engine/ATMConfig.h"
#include "../../engine/net2/Net.h"
#include "../../engine/net2/Schema.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/voxel/Chunk.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

namespace net = atm::net2;
namespace vx = atm::voxel;
using namespace ao;
using namespace ao::proto;

std::atomic<bool> g_stop{false};
void onSignal(int) { g_stop.store(true); }

enum class Behaviour { Wander, Mine, Fight };

// ---- flat terrain approximation (shared, cached heightmap) -------------------
class TerrainAccess final : public vx::IBlockAccess {
public:
  void setSeed(uint64_t seed) {
    if (gen_ && gen_->seed() == seed) return;
    gen_ = std::make_unique<vx::WorldGenerator>(seed);
    for (auto &c : cache_) c.valid = false;
  }
  bool ready() const { return gen_ != nullptr; }
  int height(int32_t x, int32_t z) const {
    const uint64_t k = (uint64_t(uint32_t(x)) << 32) | uint32_t(z);
    Cell &c = cache_[size_t((k * 0x9E3779B97F4A7C15ull) >> 52) & (kCache - 1)];
    if (!c.valid || c.x != x || c.z != z) {
      c.x = x;
      c.z = z;
      c.h = gen_ ? gen_->surfaceHeight(x, z) : 64;
      c.valid = true;
    }
    return c.h;
  }
  vx::BlockId blockAt(vx::BlockPos p) const override {
    if (p.y < 0) return vx::blocks::Bedrock;
    return p.y <= height(p.x, p.z) ? vx::blocks::Stone : vx::kAir;
  }
  bool isLoaded(vx::ChunkCoord) const override { return true; }

private:
  static constexpr size_t kCache = 1u << 12;
  struct Cell {
    int32_t x = 0, z = 0;
    int h = 0;
    bool valid = false;
  };
  std::unique_ptr<vx::WorldGenerator> gen_;
  mutable std::array<Cell, kCache> cache_{};
};

struct BotEntity {
  EntityKind kind = EntityKind::Player;
  glm::dvec3 pos{0.0};
  bool dead = false;
};

struct Bot {
  int index = 0;
  std::unique_ptr<net::Host> host;
  net::PeerId server;
  bool connected = false, welcomed = false;
  uint64_t reconnectAtMs = 0;
  EntityId self = kNoEntity;
  MoveState move;
  std::array<MoveInput, 3> recent{};
  int recentCount = 0;
  uint32_t nextSeq = 1;
  Tick newestSnapshot = 0;
  float yaw = 0.0f;
  int nextActionTick = 0, turnTick = 0;
  int dirtSlot = -1;
  std::unordered_map<EntityId, BotEntity> seen;
  uint64_t bytesIn = 0, bytesOut = 0;
};

struct Totals {
  int connected = 0, welcomed = 0;
  double rttSum = 0.0;
  int rttCount = 0;
  uint64_t bytesIn = 0, bytesOut = 0;
  uint64_t disconnects = 0, snapshots = 0, messages = 0;
};

class BotRunner {
public:
  BotRunner(std::string host, uint16_t port, Behaviour b) : host_(std::move(host)), port_(port), behaviour_(b) {
    blocks_.registerDefaults();
    rng_.seed(std::random_device{}());
  }

  void addBot() {
    auto bot = std::make_unique<Bot>();
    bot->index = int(bots_.size());
    bot->yaw = randf() * 6.2831853f - 3.1415927f;
    connectBot(*bot);
    bots_.push_back(std::move(bot));
  }
  size_t count() const { return bots_.size(); }

  void update(uint64_t nowMs, bool simTick) {
    for (auto &bp : bots_) {
      Bot &b = *bp;
      if (!b.host) {
        if (nowMs >= b.reconnectAtMs) connectBot(b);
        continue;
      }
      b.host->update(nowMs);
      net::Event ev;
      while (b.host && b.host->poll(ev)) handleEvent(b, ev, nowMs); // host may be dropped on disconnect
      if (!b.host) continue;
      if (simTick && b.welcomed) think(b);
    }
    if (simTick) ++tick_;
  }

  Totals totals() {
    Totals t = totals_;
    for (auto &bp : bots_) {
      Bot &b = *bp;
      if (!b.host) continue;
      const net::HostStats hs = b.host->hostStats();
      t.bytesIn += hs.bytesReceived;
      t.bytesOut += hs.bytesSent;
      if (b.connected) {
        ++t.connected;
        if (b.welcomed) ++t.welcomed;
        const net::PeerStats ps = b.host->stats(b.server);
        if (ps.rttMs > 0.0f) {
          t.rttSum += ps.rttMs;
          ++t.rttCount;
        }
      }
    }
    return t;
  }

private:
  float randf() { return float(double(rng_() >> 11) * (1.0 / 9007199254740992.0)); }
  int randi(int lo, int hi) { return hi <= lo ? lo : lo + int(rng_() % uint64_t(hi - lo + 1)); }

  void connectBot(Bot &b) {
    // Keep the dead host's byte counters in the totals.
    if (b.host) {
      const net::HostStats hs = b.host->hostStats();
      totals_.bytesIn += hs.bytesReceived;
      totals_.bytesOut += hs.bytesSent;
    }
    net::HostConfig hc;
    hc.maxPeers = 1;
    hc.protocolId = kTransportProtocolId;
    hc.initialPoolBlocks = 16;
    hc.maxPoolBlocks = 2048;
    hc.maxBulkBytes = 1u << 20;
    b.host = std::make_unique<net::Host>(hc);
    b.connected = b.welcomed = false;
    b.server = {};
    b.self = kNoEntity;
    b.newestSnapshot = 0; // a restarted server counts ticks from 0 again
    b.seen.clear();
    b.recentCount = 0;
    std::string err;
    if (!b.host->connect(host_, port_, &err)) {
      std::fprintf(stderr, "[bots] bot %d: connect failed: %s\n", b.index, err.c_str());
      b.host.reset();
      b.reconnectAtMs = net::nowMs() + 5000;
    }
  }

  template <class M> void send(Bot &b, const M &m, net::Channel ch) {
    net::encodeMessage(buf_, m);
    if (!buf_.empty()) b.host->send(b.server, ch, buf_);
  }

  void handleEvent(Bot &b, const net::Event &ev, uint64_t nowMs) {
    switch (ev.type) {
    case net::EventType::Connected: {
      b.connected = true;
      b.server = ev.peer;
      Hello h;
      h.schemaHash = kSchemaHash;
      h.name = "bot" + std::to_string(b.index);
      h.appearance.skinTone = uint8_t(b.index % 4);
      h.appearance.hairStyle = uint8_t(b.index % 3);
      h.appearance.hairColor = uint8_t(b.index % 5);
      send(b, h, net::Channel::ReliableOrdered);
      break;
    }
    case net::EventType::Disconnected:
      ++totals_.disconnects;
      b.connected = b.welcomed = false;
      b.reconnectAtMs = nowMs + 5000;
      // Dropping the host here is safe: update() re-checks b.host before polling again.
      {
        const net::HostStats hs = b.host->hostStats();
        totals_.bytesIn += hs.bytesReceived;
        totals_.bytesOut += hs.bytesSent;
      }
      b.host.reset();
      break;
    case net::EventType::Message: handleMessage(b, ev.data); break;
    }
  }

  void handleMessage(Bot &b, std::span<const uint8_t> data) {
    ++totals_.messages;
    net::BitReader r(data);
    uint16_t id = 0;
    if (!net::peekMessageId(data, id, r)) return;
    switch (id) {
    case Welcome::kId: {
      Welcome w;
      if (!net::decodeMessage(r, w)) return;
      terrain_.setSeed(w.worldSeed);
      b.self = w.playerEntity;
      b.move = MoveState{};
      b.move.pos = w.spawn;
      b.welcomed = true;
      break;
    }
    case SnapshotMsg::kId: {
      if (!net::decodeMessage(r, snap_)) return;
      ++totals_.snapshots;
      const Snapshot &s = snap_.snapshot;
      if (b.newestSnapshot != 0 && s.tick <= b.newestSnapshot) return; // out of order
      b.newestSnapshot = s.tick;
      // Crude reconciliation: follow the server when prediction drifts.
      const glm::dvec3 d = s.self.move.pos - b.move.pos;
      if (glm::dot(d, d) > 4.0) b.move = s.self.move;
      for (const EntityState &e : s.entities) {
        BotEntity &be = b.seen[e.id];
        if (e.mask & field::Type) be.kind = e.kind;
        if (e.mask & field::Pos) be.pos = e.pos;
        if (e.mask & field::Flags) be.dead = (e.flags & eflag::Dead) != 0;
      }
      for (EntityId gone : s.removed) b.seen.erase(gone);
      break;
    }
    case InventoryMsg::kId: {
      InventoryMsg m;
      if (!net::decodeMessage(r, m)) return;
      b.dirtSlot = -1;
      for (size_t i = 0; i < m.slots.size() && i < 9; ++i)
        if (itemDef(m.slots[i].item).kind == ItemKind::Block && m.slots[i].count > 0) {
          b.dirtSlot = int(i);
          break;
        }
      break;
    }
    default: break; // chunk data, chat, xp, loot, damage: not needed by bots
    }
  }

  void think(Bot &b) {
    // Steering
    if (tick_ >= b.turnTick) {
      b.yaw += (randf() - 0.5f) * 2.0f;
      b.turnTick = tick_ + randi(15, 90);
    }
    const BotEntity *target = nullptr;
    if (behaviour_ == Behaviour::Fight) {
      double best = 30.0 * 30.0;
      for (const auto &[id, e] : b.seen) {
        if (e.kind != EntityKind::Monster || e.dead) continue;
        const glm::dvec3 d = e.pos - b.move.pos;
        const double d2 = d.x * d.x + d.z * d.z;
        if (d2 < best) {
          best = d2;
          target = &e;
        }
      }
      if (target) {
        const glm::dvec3 d = target->pos - b.move.pos;
        b.yaw = float(std::atan2(-d.x, -d.z));
      }
    }

    MoveInput in;
    in.tick = Tick(tick_);
    in.seq = b.nextSeq++;
    in.yaw = b.yaw;
    in.pitch = 0.0f;
    in.moveZ = behaviour_ == Behaviour::Mine ? 0.4f : 1.0f;
    if (target) {
      const glm::dvec3 d = target->pos - b.move.pos;
      if (d.x * d.x + d.z * d.z < 4.0) in.moveZ = 0.0f;
    }
    if (randf() < 0.03f) in.buttons |= button::Jump;
    if (randf() < 0.005f) in.buttons |= button::Dash;
    if (randf() < 0.2f) in.buttons |= button::Sprint;
    if (terrain_.ready()) stepMovement(b.move, in, terrain_, blocks_, kSimDt);

    // Last 3 inputs, oldest first.
    if (b.recentCount < 3) {
      b.recent[size_t(b.recentCount++)] = in;
    } else {
      b.recent[0] = b.recent[1];
      b.recent[1] = b.recent[2];
      b.recent[2] = in;
    }
    batch_.inputs.assign(b.recent.begin(), b.recent.begin() + b.recentCount);
    batch_.lastSnapshotTick = b.newestSnapshot;
    send(b, batch_, net::Channel::Unreliable);

    // Occasional actions (all behaviours do a little of everything).
    if (tick_ < b.nextActionTick) return;
    const bool mine = behaviour_ == Behaviour::Mine || randf() < 0.1f;
    const bool fight = behaviour_ == Behaviour::Fight || randf() < 0.1f;
    if (fight) {
      Attack a;
      a.tick = Tick(tick_);
      a.yaw = b.yaw;
      a.pitch = target ? 0.0f : -0.2f;
      a.ability = uint8_t(randi(0, 2));
      send(b, a, net::Channel::ReliableOrdered);
    }
    if (mine && terrain_.ready()) {
      const int32_t x = int32_t(std::floor(b.move.pos.x)) + randi(-2, 2);
      const int32_t z = int32_t(std::floor(b.move.pos.z)) + randi(-2, 2);
      const int32_t y = terrain_.height(x, z);
      BlockAction ba;
      const bool place = b.dirtSlot >= 0 && randf() < 0.4f;
      ba.action = place ? 1 : 0;
      ba.x = x;
      ba.y = y;
      ba.z = z;
      ba.face = uint8_t(vx::FaceDir::PosY);
      ba.hotbarSlot = uint8_t(std::max(0, b.dirtSlot));
      send(b, ba, net::Channel::ReliableOrdered);
    }
    b.nextActionTick = tick_ + randi(20, 45);
  }

  std::string host_;
  uint16_t port_;
  Behaviour behaviour_;
  vx::BlockRegistry blocks_;
  TerrainAccess terrain_;
  std::vector<std::unique_ptr<Bot>> bots_;
  std::mt19937_64 rng_;
  int tick_ = 0;
  std::vector<uint8_t> buf_;
  SnapshotMsg snap_;
  InputBatch batch_;
  Totals totals_;
};

} // namespace

int main(int argc, char **argv) {
  {
    std::string dataError;
    if (!ao::loadGameData(atm::resolve_path("data"), &dataError))
      std::fprintf(stderr, "[bots] %s\n", dataError.c_str());
  }
  std::string host = "127.0.0.1";
  int port = 27015, count = 200, ramp = 50;
  double duration = 0.0;
  Behaviour behaviour = Behaviour::Wander;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
    if (!std::strcmp(argv[i], "--host")) host = next();
    else if (!std::strcmp(argv[i], "--port")) port = std::atoi(next());
    else if (!std::strcmp(argv[i], "--count")) count = std::atoi(next());
    else if (!std::strcmp(argv[i], "--ramp")) ramp = std::atoi(next());
    else if (!std::strcmp(argv[i], "--duration")) duration = std::atof(next());
    else if (!std::strcmp(argv[i], "--behaviour") || !std::strcmp(argv[i], "--behavior")) {
      const std::string b = next();
      if (b == "wander") behaviour = Behaviour::Wander;
      else if (b == "mine") behaviour = Behaviour::Mine;
      else if (b == "fight") behaviour = Behaviour::Fight;
      else {
        std::fprintf(stderr, "unknown behaviour '%s' (wander|mine|fight)\n", b.c_str());
        return 2;
      }
    } else {
      std::fprintf(stderr,
                   "usage: ao_bots --host 127.0.0.1 --port 27015 --count 200 --behaviour wander|mine|fight "
                   "[--ramp bots_per_s] [--duration seconds]\n");
      return std::strcmp(argv[i], "--help") ? 2 : 0;
    }
  }
  if (port <= 0 || port > 65535 || count <= 0) {
    std::fprintf(stderr, "invalid --port/--count\n");
    return 2;
  }
  ramp = std::max(1, ramp);

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  BotRunner runner(host, uint16_t(port), behaviour);
  std::printf("[bots] %d bots -> %s:%d (ramp %d/s)\n", count, host.c_str(), port, ramp);

  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / kSimHz));
  auto next = start;
  auto lastReport = start;
  double spawnCredit = 0.0;
  Totals prev;
  while (!g_stop.load()) {
    const auto now = clock::now();
    if (duration > 0.0 && std::chrono::duration<double>(now - start).count() >= duration) break;
    const bool simTick = now >= next;
    if (simTick) {
      next += dt;
      if (now - next > dt * 10) next = now;
      spawnCredit += double(ramp) / kSimHz;
      while (spawnCredit >= 1.0 && runner.count() < size_t(count)) {
        runner.addBot();
        spawnCredit -= 1.0;
      }
      if (runner.count() >= size_t(count)) spawnCredit = 0.0;
    }
    runner.update(net::nowMs(), simTick);

    if (now - lastReport >= std::chrono::seconds(5)) {
      const double secs = std::chrono::duration<double>(now - lastReport).count();
      const Totals t = runner.totals();
      std::printf("[bots] bots %zu | connected %d (in game %d) | rtt avg %.1f ms | in %.1f KB/s out %.1f KB/s | "
                  "snapshots %llu | disconnects %llu\n",
                  runner.count(), t.connected, t.welcomed, t.rttCount ? t.rttSum / t.rttCount : 0.0,
                  double(t.bytesIn - std::min(t.bytesIn, prev.bytesIn)) / 1024.0 / secs,
                  double(t.bytesOut - std::min(t.bytesOut, prev.bytesOut)) / 1024.0 / secs,
                  (unsigned long long)t.snapshots, (unsigned long long)t.disconnects);
      std::fflush(stdout);
      prev = t;
      lastReport = now;
    }
    // Pump sockets ~250 times/s between sim ticks without burning a core.
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
  std::printf("[bots] stopping\n");
  return 0;
}
