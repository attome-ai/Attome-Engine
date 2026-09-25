// AttomeNet 2 transport (docs/NETWORK_PLAN.md §4, §5A, §6A).
//
// Wire format (all integers little-endian, byte aligned):
//
//   byte 0: packet type (low 4 bits) | flags (high 4 bits)
//           0x10 = header carries acks, 0x80 = encrypted (reserved, TODO(N2);
//           packets with it set are dropped today)
//
//   Handshake (stateless challenge cookie, D2):
//     C->S ConnectRequest     [1][protocolId u32][clientSalt u64] padded to 64 bytes
//     S->C Challenge          [2][clientSalt u64][timestamp u32][cookie u64]      (21 B)
//     C->S ChallengeResponse  [3][protocolId u32][clientSalt u64][timestamp u32][cookie u64] padded to 64
//     S->C ConnectAccept      [4][clientSalt u64][connectionId u32]               (13 B)
//     S->C ConnectDenied      [5][clientSalt u64][reason u8]
//   The cookie is SipHash-2-4(server secret, address, salt, timestamp,
//   protocolId). The server allocates peer state only for a valid echoed
//   cookie; every reply is smaller than the request (no amplification).
//
//   Data  [6|flags][connectionId u32][seq u16][ack u16][ackBits 16 bytes] frames...
//     ackBits bit i = packet (ack - 1 - i) received: 129 packets acknowledged
//     per header (the ack window equals the in-flight window, D4).
//     Frames:
//       [1][len u16][records]                          unreliable: records = {[len u16][bytes]}*
//       [2|3][unitSeq u16][kind u8][len u16][payload]  reliable (2) / bulk (3) stream unit
//         kind 0 = packed records {[len u16][bytes]}*, 1 = first fragment
//         ([totalLen u32] + data), 2 = middle fragment, 3 = last fragment
//   Disconnect  [7][connectionId u32][reason u8]  (sent 3 times)
//
// Reliability is per *unit* (a pool block ≤ one packet): units are resent
// in new packets until one of the packets carrying them is acked, so a lost
// ack never stalls a stream. Units are delivered in order per stream; the
// receiver acks a packet only after every unit in it is stored in the
// receive ring (from which it is delivered to the event queue; D6).
// Sequence numbers for packets are assigned when the packet is finished and
// recorded (D5); "nothing received" is a flag (D11).

#include "Net.h"
#include "Socket.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>

namespace atm::net2 {

uint64_t nowMs() {
  using namespace std::chrono;
  return uint64_t(duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

namespace {

using detail::Address;
using detail::UdpSocket;

constexpr uint8_t kPktConnectRequest = 1, kPktChallenge = 2, kPktChallengeResponse = 3,
                  kPktAccept = 4, kPktDenied = 5, kPktData = 6, kPktDisconnect = 7;
constexpr uint8_t kFlagHasAck = 0x10;
constexpr uint8_t kFlagEncrypted = 0x80; // reserved: TODO(N2) ChaCha20-Poly1305 per-direction keys

constexpr uint8_t kFrameUnreliable = 1, kFrameReliable = 2, kFrameBulk = 3;
constexpr uint8_t kUnitPacked = 0, kUnitStart = 1, kUnitMid = 2, kUnitEnd = 3;

constexpr size_t kHeaderSize = 1 + 4 + 2 + 2 + 16;
constexpr size_t kUnitFrameOverhead = 1 + 2 + 1 + 2;
constexpr size_t kUnrelFrameOverhead = 1 + 2;
constexpr size_t kBlockData = 1400;
constexpr size_t kMaxDatagram = 1500;
constexpr size_t kHandshakeSize = 64;
constexpr uint32_t kCookieLifetimeSec = 10;
constexpr int kSentRing = 256;
constexpr int kAckWindow = 128; // bits in the ack field
constexpr uint32_t kNoBlock = 0xFFFFFFFFu;
constexpr int kMaxUnitsPerPacket = 256;
constexpr int kMaxPacketsPerFlush = 64;
constexpr int kMaxReceivesPerUpdate = 16384;
constexpr uint32_t kMaxUnreliableBlocks = 128;
constexpr uint64_t kHandshakeResendMs = 250;
constexpr uint64_t kResponseRestartMs = 4000;
constexpr int kDisconnectRepeats = 3;

inline int16_t seqDiff(uint16_t a, uint16_t b) { return int16_t(uint16_t(a - b)); }

// ---- byte-aligned little-endian writer/reader for headers -------------------
struct Wr {
  uint8_t *p;
  size_t cap;
  size_t n = 0;
  bool ok = true;
  void u8(uint8_t v) { if (n + 1 > cap) { ok = false; return; } p[n++] = v; }
  void u16(uint16_t v) { u8(uint8_t(v)); u8(uint8_t(v >> 8)); }
  void u32(uint32_t v) { u16(uint16_t(v)); u16(uint16_t(v >> 16)); }
  void u64(uint64_t v) { u32(uint32_t(v)); u32(uint32_t(v >> 32)); }
  void bytes(const uint8_t *d, size_t len) {
    if (n + len > cap) { ok = false; return; }
    if (len) std::memcpy(p + n, d, len);
    n += len;
  }
};
struct Rd {
  const uint8_t *p;
  size_t n;
  size_t pos = 0;
  bool ok = true;
  size_t remaining() const { return pos < n ? n - pos : 0; }
  uint8_t u8() { if (pos + 1 > n) { ok = false; return 0; } return p[pos++]; }
  uint16_t u16() { const uint16_t a = u8(); return uint16_t(a | (uint16_t(u8()) << 8)); }
  uint32_t u32() { const uint32_t a = u16(); return a | (uint32_t(u16()) << 16); }
  uint64_t u64() { const uint64_t a = u32(); return a | (uint64_t(u32()) << 32); }
  const uint8_t *skip(size_t len) {
    if (len > remaining()) { ok = false; return nullptr; }
    const uint8_t *r = p + pos;
    pos += len;
    return r;
  }
};

// ---- SipHash-2-4 (keyed hash for challenge cookies) --------------------------
inline uint64_t rotl64(uint64_t x, int b) { return (x << b) | (x >> (64 - b)); }
uint64_t sipHash24(const uint8_t key[16], const uint8_t *in, size_t len) {
  uint64_t k0 = 0, k1 = 0;
  for (int i = 0; i < 8; ++i) k0 |= uint64_t(key[i]) << (8 * i);
  for (int i = 0; i < 8; ++i) k1 |= uint64_t(key[8 + i]) << (8 * i);
  uint64_t v0 = 0x736f6d6570736575ull ^ k0, v1 = 0x646f72616e646f6dull ^ k1;
  uint64_t v2 = 0x6c7967656e657261ull ^ k0, v3 = 0x7465646279746573ull ^ k1;
  auto round = [&]() {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
  };
  const size_t full = len & ~size_t(7);
  for (size_t i = 0; i < full; i += 8) {
    uint64_t m = 0;
    for (int j = 0; j < 8; ++j) m |= uint64_t(in[i + size_t(j)]) << (8 * j);
    v3 ^= m; round(); round(); v0 ^= m;
  }
  uint64_t b = uint64_t(len) << 56;
  for (size_t j = 0; j < (len & 7); ++j) b |= uint64_t(in[full + j]) << (8 * j);
  v3 ^= b; round(); round(); v0 ^= b;
  v2 ^= 0xff;
  round(); round(); round(); round();
  return v0 ^ v1 ^ v2 ^ v3;
}

inline uint64_t splitmix64(uint64_t &s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// ---- slab pool of packet-sized blocks (§5A.2) --------------------------------
struct Block {
  uint32_t next = kNoBlock;
  uint16_t len = 0;
  uint16_t unitSeq = 0;
  uint16_t lastSeq = 0, prevSeq = 0; // packets that carried this unit (last two sends)
  uint16_t retries = 0;
  uint16_t msgCount = 0;             // reliable messages completed by this unit
  uint8_t kind = 0;
  uint8_t sendCount = 0;             // saturating
  bool acked = false;
  uint64_t sentMs = 0;
  uint8_t data[kBlockData];
};

class BlockPool {
public:
  void init(uint32_t initial, uint32_t maxBlocks, uint32_t slab) {
    maxBlocks_ = std::max<uint32_t>(maxBlocks, 16);
    slab_ = std::max<uint32_t>(slab, 8);
    slabs_.reserve(maxBlocks_ / slab_ + 2);
    while (count_ < initial && count_ < maxBlocks_) grow();
  }
  uint32_t alloc() {
    if (free_ == kNoBlock && !grow()) return kNoBlock;
    const uint32_t idx = free_;
    Block &b = at(idx);
    free_ = b.next;
    b.next = kNoBlock;
    b.len = 0;
    b.unitSeq = 0;
    b.lastSeq = b.prevSeq = 0;
    b.retries = 0;
    b.msgCount = 0;
    b.kind = 0;
    b.sendCount = 0;
    b.acked = false;
    b.sentMs = 0;
    ++inUse_;
    return idx;
  }
  void release(uint32_t idx) {
    if (idx == kNoBlock) return;
    Block &b = at(idx);
    b.next = free_;
    free_ = idx;
    --inUse_;
  }
  Block &at(uint32_t idx) { return slabs_[idx / slab_][idx % slab_]; }
  uint32_t count() const { return count_; }
  uint32_t inUse() const { return inUse_; }
  uint32_t maxBlocks() const { return maxBlocks_; }

private:
  bool grow() {
    if (count_ >= maxBlocks_) return false;
    slabs_.push_back(std::unique_ptr<Block[]>(new Block[slab_]));
    const uint32_t base = uint32_t(slabs_.size() - 1) * slab_;
    // Push in reverse so allocation order is ascending (cache friendly).
    for (uint32_t i = slab_; i-- > 0;) {
      Block &b = slabs_.back()[i];
      b.next = free_;
      free_ = base + i;
    }
    count_ += slab_;
    return true;
  }
  std::vector<std::unique_ptr<Block[]>> slabs_;
  uint32_t slab_ = 64;
  uint32_t free_ = kNoBlock;
  uint32_t count_ = 0, inUse_ = 0, maxBlocks_ = 0;
};

// ---- open-addressing address -> peer index table ------------------------------
class AddrTable {
public:
  void init(size_t peers) {
    size_t cap = 16;
    while (cap < peers * 2) cap <<= 1;
    keys_.assign(cap, kEmpty);
    vals_.assign(cap, 0);
    mask_ = cap - 1;
  }
  int find(uint64_t key) const {
    if (keys_.empty()) return -1;
    for (size_t i = slot(key);; i = (i + 1) & mask_) {
      if (keys_[i] == kEmpty) return -1;
      if (keys_[i] == key) return int(vals_[i]);
    }
  }
  void insert(uint64_t key, uint16_t val) {
    for (size_t i = slot(key);; i = (i + 1) & mask_) {
      if (keys_[i] == kEmpty || keys_[i] == key) { keys_[i] = key; vals_[i] = val; return; }
    }
  }
  void erase(uint64_t key) { // backward-shift deletion (no tombstones)
    if (keys_.empty()) return;
    size_t i = slot(key);
    for (;; i = (i + 1) & mask_) {
      if (keys_[i] == kEmpty) return;
      if (keys_[i] == key) break;
    }
    size_t j = i;
    for (;;) {
      j = (j + 1) & mask_;
      if (keys_[j] == kEmpty) break;
      const size_t home = slot(keys_[j]);
      // Move j back to i if home is not cyclically in (i, j].
      const bool inRange = (i <= j) ? (home > i && home <= j) : (home > i || home <= j);
      if (!inRange) {
        keys_[i] = keys_[j];
        vals_[i] = vals_[j];
        i = j;
      }
    }
    keys_[i] = kEmpty;
  }

private:
  static constexpr uint64_t kEmpty = ~0ull;
  size_t slot(uint64_t key) const { return size_t((key * 0x9E3779B97F4A7C15ull) >> 20) & mask_; }
  std::vector<uint64_t> keys_;
  std::vector<uint16_t> vals_;
  size_t mask_ = 0;
};

struct SentPacket {
  uint64_t sentMs = 0;
  uint16_t seq = 0;
  bool valid = false;
  bool acked = false;
};

struct SendStream {
  uint32_t head = kNoBlock, tail = kNoBlock;
  uint16_t nextUnitSeq = 0;
  uint32_t messages = 0; // queued reliable messages
  uint32_t bytes = 0;    // queued payload bytes (sum of block lengths)
};

struct RecvStream {
  uint16_t nextSeq = 0;
  bool assembling = false;
  uint32_t expected = 0;
  std::vector<uint8_t> assembly; // grows lazily for multi-unit messages
};

struct Peer {
  bool active = false;
  uint16_t gen = 0;
  uint32_t connId = 0;
  Address addr;
  uint64_t clientSalt = 0;
  uint64_t lastRecvMs = 0, lastSendMs = 0;

  uint16_t nextSendSeq = 0;
  bool hasRecv = false;
  uint16_t recvLatest = 0;
  uint64_t recvBits[2] = {0, 0};
  bool needAck = false;
  bool ackDirty = false;
  std::array<SentPacket, kSentRing> sent{};

  bool hasRtt = false;
  float srtt = 0.0f, rttvar = 0.0f, rto = 200.0f;
  float loss = 0.0f;
  double tokens = 0.0;
  uint64_t lastRefillMs = 0;

  SendStream streams[2];
  RecvStream recv[2];
  uint32_t unrelHead = kNoBlock, unrelTail = kNoBlock, unrelCount = 0;

  uint64_t bytesSent = 0, bytesReceived = 0;
  bool deliveryBlocked = false;
  DisconnectReason pendingDisconnect = DisconnectReason::None;
  int activeIndex = -1;
};

} // namespace

// =============================================================================
// Host::Impl
// =============================================================================

struct Host::Impl {
  enum class Mode : uint8_t { None, Server, Client };
  enum class ClientState : uint8_t { Idle, SendingRequest, SendingResponse, Connected };

  HostConfig cfg;
  Mode mode = Mode::None;
  UdpSocket socket;
  BlockPool pool;
  std::vector<Peer> peers;
  std::vector<uint32_t> recvRing; // per peer: 2 streams x window block indices
  uint32_t window = 128;          // units per stream in flight / reorder window
  std::vector<uint16_t> freeRing; // FIFO free list of peer indices (D7)
  size_t freeHead = 0, freeCount = 0;
  std::vector<uint16_t> active;
  AddrTable addrTable;

  std::vector<Event> events;
  size_t eventRead = 0;
  std::vector<uint8_t> arena;
  size_t arenaUsed = 0;
  std::vector<uint32_t> blocked; // PeerId values with deferred delivery

  uint8_t secret[16] = {};
  uint64_t rng = 0;
  uint64_t startMs = 0;
  uint64_t now = 0;
  size_t unitMax = 0; // payload bytes per unit

  // client handshake
  ClientState cstate = ClientState::Idle;
  Address serverAddr;
  uint64_t clientSalt = 0;
  uint32_t chTimestamp = 0;
  uint64_t chCookie = 0;
  uint64_t connectStartMs = 0, responseStartMs = 0, lastHandshakeMs = 0;

  // packet under construction
  uint8_t recvBuf[kMaxDatagram + 64];
  uint8_t pkt[kMaxDatagram];
  size_t pktLen = 0;
  std::array<uint32_t, kMaxUnitsPerPacket> pktUnits{};
  int pktUnitCount = 0;
  int packetsThisFlush = 0;

  HostStats hs;

  explicit Impl(const HostConfig &c) : cfg(c) {
    if (cfg.maxPeers == 0) cfg.maxPeers = 1;
    if (cfg.maxPeers == 0xFFFFu) cfg.maxPeers = 0xFFFEu; // index 0xFFFF reserved
    cfg.mtu = uint16_t(std::clamp<int>(cfg.mtu, 256, int(kHeaderSize + kUnitFrameOverhead + kBlockData)));
    if (cfg.maxRetries == 0) cfg.maxRetries = 1;
    if (cfg.keepaliveMs == 0) cfg.keepaliveMs = 250;
    if (cfg.timeoutMs < 1000) cfg.timeoutMs = 1000;
    if (cfg.bandwidthBytesPerSec < 8192) cfg.bandwidthBytesPerSec = 8192;
    unitMax = std::min<size_t>(kBlockData, size_t(cfg.mtu) - kHeaderSize - kUnitFrameOverhead);
    uint32_t w = 16;
    while (w < cfg.maxInFlight && w < 256) w <<= 1;
    window = w;

    std::random_device rd;
    for (auto &b : secret) b = uint8_t(rd());
    rng = (uint64_t(rd()) << 32) ^ uint64_t(rd()) ^ nowMs();
    startMs = nowMs();
    now = startMs;
  }

  void allocate(uint16_t peerCount) {
    peers.clear();
    peers.resize(peerCount);
    recvRing.assign(size_t(peerCount) * 2 * window, kNoBlock);
    freeRing.resize(peerCount);
    for (uint16_t i = 0; i < peerCount; ++i) freeRing[i] = i;
    freeHead = 0;
    freeCount = peerCount;
    active.clear();
    active.reserve(peerCount);
    addrTable.init(peerCount);
    uint32_t maxBlocks = cfg.maxPoolBlocks;
    if (maxBlocks == 0) maxBlocks = std::clamp<uint32_t>(uint32_t(peerCount) * 1024u, 4096u, 131072u);
    uint32_t initial = cfg.initialPoolBlocks;
    if (initial == 0) initial = std::min<uint32_t>(uint32_t(peerCount) * 64u, 4096u);
    pool.init(std::min(initial, maxBlocks), maxBlocks, peerCount > 1 ? 256u : 32u);
    events.reserve(peerCount > 1 ? 4096 : 256);
    arena.resize(peerCount > 1 ? (1u << 20) : (64u << 10)); // grows (when empty) for larger messages
    blocked.reserve(peerCount);
  }

  // ---- peer helpers --------------------------------------------------------
  static PeerId idOf(uint16_t index, const Peer &p) {
    PeerId id;
    id.value = (uint32_t(p.gen) << 16) | index;
    return id;
  }
  uint16_t indexOf(const Peer &p) const { return uint16_t(&p - peers.data()); }
  Peer *lookup(PeerId id) {
    if (!id.valid() || id.index() >= peers.size()) return nullptr;
    Peer &p = peers[id.index()];
    if (!p.active || p.gen != id.generation()) return nullptr;
    return &p;
  }
  const Peer *lookup(PeerId id) const { return const_cast<Impl *>(this)->lookup(id); }
  uint32_t *ring(Peer &p, int stream) {
    return recvRing.data() + (size_t(indexOf(p)) * 2 + size_t(stream)) * window;
  }

  void pushEvent(EventType t, PeerId id, DisconnectReason reason = DisconnectReason::None) {
    Event e;
    e.type = t;
    e.peer = id;
    e.reason = reason;
    e.data = {};
    events.push_back(e);
  }

  double burst() const {
    return std::max<double>(double(cfg.mtu) * 8.0, double(cfg.bandwidthBytesPerSec) / 10.0);
  }

  Peer *activate(uint16_t index, const Address &addr, uint32_t connId, uint64_t salt) {
    Peer &p = peers[index];
    const uint16_t gen = p.gen;
    std::vector<uint8_t> a0 = std::move(p.recv[0].assembly), a1 = std::move(p.recv[1].assembly);
    p = Peer{};
    p.gen = gen;
    p.recv[0].assembly = std::move(a0);
    p.recv[1].assembly = std::move(a1);
    p.recv[0].assembly.clear();
    p.recv[1].assembly.clear();
    p.active = true;
    p.addr = addr;
    p.connId = connId;
    p.clientSalt = salt;
    p.lastRecvMs = p.lastSendMs = now;
    p.lastRefillMs = now;
    p.tokens = burst();
    p.activeIndex = int(active.size());
    active.push_back(index);
    if (mode == Mode::Server) addrTable.insert(addr.key(), index);
    return &p;
  }

  void releaseChain(uint32_t head) {
    while (head != kNoBlock) {
      const uint32_t next = pool.at(head).next;
      pool.release(head);
      head = next;
    }
  }

  void sendControl(const Address &to, const uint8_t *data, size_t len) {
    if (socket.send(to, data, int(len))) {
      hs.bytesSent += len;
      ++hs.packetsSent;
    }
  }

  void sendDisconnectPackets(const Peer &p, DisconnectReason reason) {
    uint8_t buf[8];
    Wr w{buf, sizeof(buf)};
    w.u8(kPktDisconnect);
    w.u32(p.connId);
    w.u8(uint8_t(reason));
    for (int i = 0; i < kDisconnectRepeats; ++i) sendControl(p.addr, buf, w.n);
  }

  // Frees every resource of a peer and emits exactly one Disconnected event.
  void freePeer(Peer &p, DisconnectReason reason, bool notifyRemote) {
    if (!p.active) return;
    const uint16_t index = indexOf(p);
    const PeerId id = idOf(index, p);
    if (notifyRemote) sendDisconnectPackets(p, reason);
    for (int s = 0; s < 2; ++s) {
      releaseChain(p.streams[s].head);
      p.streams[s] = SendStream{};
      uint32_t *r = ring(p, s);
      for (uint32_t i = 0; i < window; ++i) {
        if (r[i] != kNoBlock) {
          pool.release(r[i]);
          r[i] = kNoBlock;
        }
      }
      std::vector<uint8_t>().swap(p.recv[s].assembly);
      p.recv[s].assembling = false;
    }
    releaseChain(p.unrelHead);
    p.unrelHead = p.unrelTail = kNoBlock;
    p.unrelCount = 0;
    if (mode == Mode::Server) addrTable.erase(p.addr.key());
    if (p.activeIndex >= 0 && size_t(p.activeIndex) < active.size()) { // swap-remove
      const uint16_t last = active.back();
      active[size_t(p.activeIndex)] = last;
      peers[last].activeIndex = p.activeIndex;
      active.pop_back();
    }
    p.activeIndex = -1;
    p.active = false;
    p.deliveryBlocked = false;
    p.gen = uint16_t(p.gen + 1); // stale PeerIds/packets are rejected from now on
    if (mode == Mode::Server) {
      freeRing[(freeHead + freeCount) % freeRing.size()] = index;
      ++freeCount;
    } else {
      cstate = ClientState::Idle;
    }
    pushEvent(EventType::Disconnected, id, reason);
  }

  // ---- event arena -----------------------------------------------------------
  bool deliver(Peer &p, Channel ch, const uint8_t *a, size_t an, const uint8_t *b = nullptr,
               size_t bn = 0) {
    const size_t need = an + bn;
    if (arenaUsed + need > arena.size()) {
      if (arenaUsed != 0) return false; // wait for the app to drain
      arena.resize(need);               // rare: one message larger than the arena
    }
    uint8_t *dst = arena.data() + arenaUsed;
    if (an) std::memcpy(dst, a, an);
    if (bn) std::memcpy(dst + an, b, bn);
    Event e;
    e.type = EventType::Message;
    e.peer = idOf(indexOf(p), p);
    e.channel = ch;
    e.data = std::span<const uint8_t>(dst, need);
    events.push_back(e);
    arenaUsed += need;
    return true;
  }

  // Validates packed records {[len u16][bytes]}*; returns total payload bytes or -1.
  static long long packedTotal(const uint8_t *d, size_t len) {
    Rd r{d, len};
    long long total = 0;
    while (r.remaining() > 0) {
      const uint16_t n = r.u16();
      if (!r.ok || !r.skip(n)) return -1;
      total += n;
    }
    return r.ok ? total : -1;
  }

  void markBlocked(Peer &p) {
    if (p.deliveryBlocked) return;
    p.deliveryBlocked = true;
    blocked.push_back(idOf(indexOf(p), p).value);
  }

  // Delivers contiguous stored units of one stream. Returns false on a
  // protocol violation (caller disconnects the peer).
  bool deliverStream(Peer &p, int s) {
    RecvStream &rs = p.recv[s];
    uint32_t *r = ring(p, s);
    const Channel ch = s == 0 ? Channel::ReliableOrdered : Channel::Bulk;
    const size_t maxMessage = std::max<size_t>(cfg.maxBulkBytes, 64u * 1024u);
    for (;;) {
      const uint32_t slot = rs.nextSeq & (window - 1);
      const uint32_t bi = r[slot];
      if (bi == kNoBlock) return true;
      Block &b = pool.at(bi);
      switch (b.kind) {
      case kUnitPacked: {
        if (rs.assembling) return false;
        const long long total = packedTotal(b.data, b.len);
        if (total < 0) return false;
        // Room for every record first, so a unit is delivered whole.
        if (arenaUsed + size_t(total) > arena.size()) {
          if (arenaUsed != 0) {
            markBlocked(p);
            return true;
          }
          arena.resize(size_t(total)); // no outstanding spans while the arena is empty
        }
        Rd rd{b.data, b.len};
        while (rd.remaining() > 0) {
          const uint16_t n = rd.u16();
          const uint8_t *d = rd.skip(n);
          deliver(p, ch, d, n); // fits: checked above (or the arena was empty and grows)
        }
        break;
      }
      case kUnitStart: {
        if (rs.assembling || b.len < 4) return false;
        const uint32_t total = uint32_t(b.data[0]) | (uint32_t(b.data[1]) << 8) |
                               (uint32_t(b.data[2]) << 16) | (uint32_t(b.data[3]) << 24);
        if (total > maxMessage || total <= uint32_t(b.len - 4)) return false;
        rs.assembly.clear();
        rs.assembly.insert(rs.assembly.end(), b.data + 4, b.data + b.len);
        rs.expected = total;
        rs.assembling = true;
        break;
      }
      case kUnitMid:
        if (!rs.assembling || rs.assembly.size() + b.len >= rs.expected) return false;
        rs.assembly.insert(rs.assembly.end(), b.data, b.data + b.len);
        break;
      case kUnitEnd:
        if (!rs.assembling || rs.assembly.size() + b.len != rs.expected) return false;
        if (!deliver(p, ch, rs.assembly.data(), rs.assembly.size(), b.data, b.len)) {
          markBlocked(p);
          return true;
        }
        rs.assembling = false;
        rs.assembly.clear();
        break;
      default:
        return false;
      }
      pool.release(bi);
      r[slot] = kNoBlock;
      rs.nextSeq = uint16_t(rs.nextSeq + 1);
    }
  }

  void deliverAll(Peer &p) {
    for (int s = 0; s < 2 && p.active; ++s) {
      if (!deliverStream(p, s)) {
        ++hs.malformedPackets;
        p.pendingDisconnect = DisconnectReason::Rejected;
        return;
      }
    }
  }

  // ---- receive path ------------------------------------------------------------
  // 0 = new, 1 = duplicate or too old
  int classify(const Peer &p, uint16_t seq) const {
    if (!p.hasRecv) return 0;
    const int d = seqDiff(seq, p.recvLatest);
    if (d > 0) return 0;
    if (d == 0) return 1;
    const int idx = -d - 1;
    if (idx >= kAckWindow) return 1;
    return ((p.recvBits[idx >> 6] >> (idx & 63)) & 1u) ? 1 : 0;
  }

  static void shiftBits(uint64_t bits[2], int d) {
    if (d <= 0) return;
    if (d >= 128) {
      bits[0] = bits[1] = 0;
      return;
    }
    if (d >= 64) {
      bits[1] = bits[0] << (d - 64);
      bits[0] = 0;
      return;
    }
    bits[1] = (bits[1] << d) | (bits[0] >> (64 - d));
    bits[0] <<= d;
  }

  void markReceived(Peer &p, uint16_t seq) {
    if (!p.hasRecv) {
      p.hasRecv = true;
      p.recvLatest = seq;
      p.recvBits[0] = p.recvBits[1] = 0;
      return;
    }
    const int d = seqDiff(seq, p.recvLatest);
    if (d > 0) {
      shiftBits(p.recvBits, d);
      const int idx = d - 1; // the previous latest
      if (idx < kAckWindow) p.recvBits[idx >> 6] |= 1ull << (idx & 63);
      p.recvLatest = seq;
    } else if (d < 0) {
      const int idx = -d - 1;
      if (idx < kAckWindow) p.recvBits[idx >> 6] |= 1ull << (idx & 63);
    }
  }

  void processAcks(Peer &p, uint16_t ack, const uint64_t bits[2]) {
    if (seqDiff(ack, p.nextSendSeq) >= 0) return; // ack for a packet never sent: bogus
    float newestSample = -1.0f;
    for (int i = 0; i <= kAckWindow; ++i) {
      if (i > 0) {
        const int bit = i - 1;
        if (!((bits[bit >> 6] >> (bit & 63)) & 1u)) continue;
      }
      const uint16_t seq = uint16_t(ack - uint16_t(i));
      SentPacket &sp = p.sent[seq & (kSentRing - 1)];
      if (!sp.valid || sp.seq != seq || sp.acked) continue;
      sp.acked = true;
      p.ackDirty = true;
      if (newestSample < 0.0f) newestSample = float(now - sp.sentMs);
    }
    if (newestSample >= 0.0f) {
      const float r = newestSample;
      if (!p.hasRtt) {
        p.srtt = r;
        p.rttvar = r * 0.5f;
        p.hasRtt = true;
      } else {
        p.rttvar = 0.75f * p.rttvar + 0.25f * std::fabs(p.srtt - r);
        p.srtt = 0.875f * p.srtt + 0.125f * r;
      }
      p.rto = std::clamp(p.srtt + std::max(4.0f * p.rttvar, 20.0f), 50.0f, 3000.0f);
    }
  }

  // Stores a stream unit in the receive ring. False = could not store (the
  // packet is then not acked and the sender resends).
  bool storeUnit(Peer &p, int s, uint16_t unitSeq, uint8_t kind, const uint8_t *d, uint16_t len) {
    RecvStream &rs = p.recv[s];
    const int diff = seqDiff(unitSeq, rs.nextSeq);
    if (diff < 0) return true;             // already delivered (duplicate)
    if (diff >= int(window)) return false; // outside the window
    uint32_t *r = ring(p, s);
    const uint32_t slot = unitSeq & (window - 1);
    if (r[slot] != kNoBlock) return true;  // already stored
    const uint32_t bi = pool.alloc();
    if (bi == kNoBlock) return false;      // pool exhausted: no ack, sender retries
    Block &b = pool.at(bi);
    b.kind = kind;
    b.len = len;
    b.unitSeq = unitSeq;
    if (len) std::memcpy(b.data, d, len);
    r[slot] = bi;
    return true;
  }

  // Parses frames. Returns 0 ok, 1 = something not storable (don't ack), 2 = malformed.
  int processFrames(Peer &p, Rd &rd, bool &hadReliable) {
    int result = 0;
    while (rd.remaining() > 0) {
      const uint8_t type = rd.u8();
      if (type == kFrameUnreliable) {
        const uint16_t len = rd.u16();
        const uint8_t *d = rd.skip(len);
        if (!rd.ok || packedTotal(d, len) < 0) return 2;
        Rd rr{d, len};
        while (rr.remaining() > 0) {
          const uint16_t n = rr.u16();
          const uint8_t *m = rr.skip(n);
          deliver(p, Channel::Unreliable, m, n); // dropped if the app is behind
        }
      } else if (type == kFrameReliable || type == kFrameBulk) {
        const uint16_t unitSeq = rd.u16();
        const uint8_t kind = rd.u8();
        const uint16_t len = rd.u16();
        const uint8_t *d = rd.skip(len);
        if (!rd.ok || kind > kUnitEnd || len > kBlockData) return 2;
        hadReliable = true;
        if (!storeUnit(p, int(type - kFrameReliable), unitSeq, kind, d, len)) result = 1;
      } else {
        return 2;
      }
    }
    return result;
  }

  void handleData(const Address &from, const uint8_t *data, size_t len) {
    Rd rd{data, len};
    const uint8_t flags = rd.u8();
    const uint32_t connId = rd.u32();
    const uint16_t seq = rd.u16();
    const uint16_t ack = rd.u16();
    uint64_t bits[2];
    bits[0] = rd.u64();
    bits[1] = rd.u64();
    if (!rd.ok) {
      ++hs.malformedPackets;
      return;
    }
    const uint16_t index = mode == Mode::Server ? uint16_t(connId & 0xFFFFu) : 0;
    if (index >= peers.size()) {
      ++hs.malformedPackets;
      return;
    }
    Peer &p = peers[index];
    // TODO(N2): connection migration (address change) needs authenticated packets.
    if (!p.active || p.connId != connId || !(p.addr == from)) {
      ++hs.malformedPackets;
      return;
    }
    p.lastRecvMs = now;
    p.bytesReceived += len;
    if (flags & kFlagHasAck) processAcks(p, ack, bits);
    if (classify(p, seq) != 0) {
      if (rd.remaining() > 0) p.needAck = true; // our ack may have been lost
      return;
    }
    bool hadReliable = false;
    const int res = processFrames(p, rd, hadReliable);
    if (res == 2) {
      ++hs.malformedPackets;
      return; // not marked received; units already stored are harmless
    }
    if (res == 0) markReceived(p, seq);
    if (hadReliable) {
      p.needAck = true;
      if (!p.deliveryBlocked) deliverAll(p);
    }
  }

  uint64_t cookieFor(const Address &a, uint64_t salt, uint32_t ts) const {
    uint8_t buf[32];
    Wr w{buf, sizeof(buf)};
    w.u32(a.ip);
    w.u16(a.port);
    w.u64(salt);
    w.u32(ts);
    w.u32(cfg.protocolId);
    return sipHash24(secret, buf, w.n);
  }
  uint32_t nowSec() const { return uint32_t((now - startMs) / 1000u) + 1u; }

  void sendDenied(const Address &to, uint64_t salt, DisconnectReason reason) {
    uint8_t buf[16];
    Wr w{buf, sizeof(buf)};
    w.u8(kPktDenied);
    w.u64(salt);
    w.u8(uint8_t(reason));
    sendControl(to, buf, w.n);
  }
  void sendAccept(const Peer &p) {
    uint8_t buf[16];
    Wr w{buf, sizeof(buf)};
    w.u8(kPktAccept);
    w.u64(p.clientSalt);
    w.u32(p.connId);
    sendControl(p.addr, buf, w.n);
  }

  void serverHandshake(uint8_t type, const Address &from, const uint8_t *data, size_t len) {
    if (len < kHandshakeSize) {
      ++hs.malformedPackets;
      return;
    }
    Rd rd{data, len};
    rd.u8();
    const uint32_t protocolId = rd.u32();
    const uint64_t salt = rd.u64();
    if (protocolId != cfg.protocolId) {
      sendDenied(from, salt, DisconnectReason::ProtocolMismatch);
      return;
    }
    if (type == kPktConnectRequest) {
      const uint32_t ts = nowSec();
      uint8_t buf[32];
      Wr w{buf, sizeof(buf)};
      w.u8(kPktChallenge);
      w.u64(salt);
      w.u32(ts);
      w.u64(cookieFor(from, salt, ts));
      sendControl(from, buf, w.n);
      return;
    }
    // ChallengeResponse: only now is any per-peer state allocated.
    const uint32_t ts = rd.u32();
    const uint64_t cookie = rd.u64();
    const uint32_t t = nowSec();
    if (!rd.ok || ts > t || t - ts > kCookieLifetimeSec || cookie != cookieFor(from, salt, ts)) {
      ++hs.malformedPackets;
      return;
    }
    const int existing = addrTable.find(from.key());
    if (existing >= 0) {
      Peer &old = peers[size_t(existing)];
      if (old.active && old.clientSalt == salt) { // our accept was lost
        sendAccept(old);
        return;
      }
      if (old.active) freePeer(old, DisconnectReason::Requested, false); // client restarted
    }
    if (freeCount == 0) {
      sendDenied(from, salt, DisconnectReason::ServerFull);
      return;
    }
    const uint16_t index = freeRing[freeHead];
    freeHead = (freeHead + 1) % freeRing.size();
    --freeCount;
    uint32_t hi = 0;
    while (hi == 0) hi = uint32_t(splitmix64(rng)) & 0xFFFFu;
    Peer *p = activate(index, from, (hi << 16) | index, salt);
    sendAccept(*p);
    pushEvent(EventType::Connected, idOf(index, *p));
  }

  void clientSendHandshake() {
    uint8_t buf[kHandshakeSize];
    std::memset(buf, 0, sizeof(buf));
    Wr w{buf, sizeof(buf)};
    if (cstate == ClientState::SendingRequest) {
      w.u8(kPktConnectRequest);
      w.u32(cfg.protocolId);
      w.u64(clientSalt);
    } else {
      w.u8(kPktChallengeResponse);
      w.u32(cfg.protocolId);
      w.u64(clientSalt);
      w.u32(chTimestamp);
      w.u64(chCookie);
    }
    sendControl(serverAddr, buf, sizeof(buf)); // padded: replies are never larger
    lastHandshakeMs = now;
  }

  void clientFail(DisconnectReason reason) {
    cstate = ClientState::Idle;
    Peer &p = peers[0];
    pushEvent(EventType::Disconnected, idOf(0, p), reason);
    p.gen = uint16_t(p.gen + 1);
  }

  void clientHandshake(uint8_t type, const uint8_t *data, size_t len) {
    Rd rd{data, len};
    rd.u8();
    const uint64_t salt = rd.u64();
    if (!rd.ok || salt != clientSalt) {
      ++hs.malformedPackets;
      return;
    }
    if (type == kPktChallenge) {
      const uint32_t ts = rd.u32();
      const uint64_t cookie = rd.u64();
      if (!rd.ok || cstate != ClientState::SendingRequest) return;
      chTimestamp = ts;
      chCookie = cookie;
      cstate = ClientState::SendingResponse;
      responseStartMs = now;
      clientSendHandshake();
    } else if (type == kPktAccept) {
      const uint32_t connId = rd.u32();
      if (!rd.ok || cstate != ClientState::SendingResponse) return;
      cstate = ClientState::Connected;
      Peer *p = activate(0, serverAddr, connId, salt);
      pushEvent(EventType::Connected, idOf(0, *p));
    } else if (type == kPktDenied) {
      const uint8_t reason = rd.u8();
      if (!rd.ok || cstate == ClientState::Connected || cstate == ClientState::Idle) return;
      DisconnectReason r = DisconnectReason::Rejected;
      if (reason == uint8_t(DisconnectReason::ServerFull)) r = DisconnectReason::ServerFull;
      if (reason == uint8_t(DisconnectReason::ProtocolMismatch)) r = DisconnectReason::ProtocolMismatch;
      clientFail(r);
    }
  }

  void handleDisconnectPacket(const Address &from, const uint8_t *data, size_t len) {
    Rd rd{data, len};
    rd.u8();
    const uint32_t connId = rd.u32();
    if (!rd.ok) return;
    const uint16_t index = mode == Mode::Server ? uint16_t(connId & 0xFFFFu) : 0;
    if (index >= peers.size()) return;
    Peer &p = peers[index];
    if (!p.active || p.connId != connId || !(p.addr == from)) return;
    freePeer(p, DisconnectReason::Requested, false);
  }

  void handlePacket(const Address &from, const uint8_t *data, size_t len) {
    ++hs.packetsReceived;
    hs.bytesReceived += len;
    if (len < 1) return;
    const uint8_t first = data[0];
    if (first & kFlagEncrypted) { // TODO(N2): encryption
      ++hs.malformedPackets;
      return;
    }
    const uint8_t type = first & 0x0F;
    if (mode == Mode::Server) {
      switch (type) {
      case kPktConnectRequest:
      case kPktChallengeResponse: serverHandshake(type, from, data, len); return;
      case kPktData: handleData(from, data, len); return;
      case kPktDisconnect: handleDisconnectPacket(from, data, len); return;
      default: ++hs.malformedPackets; return;
      }
    }
    if (!(from == serverAddr)) {
      ++hs.malformedPackets;
      return;
    }
    switch (type) {
    case kPktChallenge:
    case kPktAccept:
    case kPktDenied: clientHandshake(type, data, len); return;
    case kPktData: handleData(from, data, len); return;
    case kPktDisconnect: handleDisconnectPacket(from, data, len); return;
    default: ++hs.malformedPackets; return;
    }
  }

  // ---- send path ---------------------------------------------------------------
  void beginPacket() {
    pktLen = kHeaderSize;
    pktUnitCount = 0;
  }

  // Assigns the sequence, records the packet, then sends it (D5).
  void finishPacket(Peer &p) {
    const uint16_t seq = p.nextSendSeq;
    p.nextSendSeq = uint16_t(seq + 1);
    SentPacket &sp = p.sent[seq & (kSentRing - 1)];
    if (sp.valid) { // the old entry's fate is final now: feed the loss estimate
      const float sample = sp.acked ? 0.0f : 100.0f;
      p.loss += (sample - p.loss) * 0.02f;
    }
    sp.valid = true;
    sp.acked = false;
    sp.seq = seq;
    sp.sentMs = now;

    Wr w{pkt, kHeaderSize};
    w.u8(uint8_t(kPktData | (p.hasRecv ? kFlagHasAck : 0)));
    w.u32(p.connId);
    w.u16(seq);
    w.u16(p.hasRecv ? p.recvLatest : uint16_t(0));
    w.u64(p.hasRecv ? p.recvBits[0] : 0);
    w.u64(p.hasRecv ? p.recvBits[1] : 0);

    for (int i = 0; i < pktUnitCount; ++i) {
      Block &b = pool.at(pktUnits[size_t(i)]);
      b.prevSeq = b.lastSeq;
      b.lastSeq = seq;
      if (b.sendCount < 255) ++b.sendCount;
      b.sentMs = now;
    }
    if (socket.send(p.addr, pkt, int(pktLen))) {
      hs.bytesSent += pktLen;
      ++hs.packetsSent;
    } // a failed send is just a lost packet
    p.bytesSent += pktLen;
    p.lastSendMs = now;
    p.needAck = false;
    ++packetsThisFlush;
    beginPacket();
  }

  bool ensureRoom(Peer &p, size_t frameLen) {
    if (pktLen + frameLen <= cfg.mtu && pktUnitCount < kMaxUnitsPerPacket) return true;
    if (pktLen > kHeaderSize) finishPacket(p);
    return packetsThisFlush < kMaxPacketsPerFlush && pktLen + frameLen <= cfg.mtu;
  }

  static bool isAcked(const Peer &p, uint16_t seq) {
    const SentPacket &sp = p.sent[seq & (kSentRing - 1)];
    return sp.valid && sp.seq == seq && sp.acked;
  }

  void processAckedUnits(Peer &p) {
    for (int s = 0; s < 2; ++s) {
      SendStream &ss = p.streams[s];
      if (ss.head == kNoBlock) continue;
      const uint16_t base = pool.at(ss.head).unitSeq;
      for (uint32_t bi = ss.head; bi != kNoBlock;) {
        Block &b = pool.at(bi);
        if (seqDiff(b.unitSeq, base) >= int(window)) break;
        if (b.sendCount > 0 && !b.acked &&
            (isAcked(p, b.lastSeq) || (b.sendCount > 1 && isAcked(p, b.prevSeq))))
          b.acked = true;
        bi = b.next;
      }
      while (ss.head != kNoBlock && pool.at(ss.head).acked) {
        Block &b = pool.at(ss.head);
        const uint32_t next = b.next;
        ss.messages -= std::min<uint32_t>(ss.messages, b.msgCount);
        ss.bytes -= std::min<uint32_t>(ss.bytes, b.len);
        pool.release(ss.head);
        ss.head = next;
      }
      if (ss.head == kNoBlock) ss.tail = kNoBlock;
    }
  }

  // False when the peer must be disconnected (retry cap reached).
  bool sendStream(Peer &p, int s) {
    SendStream &ss = p.streams[s];
    if (ss.head == kNoBlock) return true;
    const uint16_t base = pool.at(ss.head).unitSeq;
    const uint8_t frameType = s == 0 ? kFrameReliable : kFrameBulk;
    for (uint32_t bi = ss.head; bi != kNoBlock;) {
      Block &b = pool.at(bi);
      const uint32_t next = b.next;
      if (seqDiff(b.unitSeq, base) >= int(window)) break; // receiver window
      if (!b.acked) {
        bool need = false, resend = false;
        if (b.sendCount == 0) {
          need = true;
        } else {
          const uint32_t backoff = 1u << std::min<uint32_t>(b.retries, 3u);
          const bool timedOut = double(now - b.sentMs) >= double(p.rto) * backoff;
          const bool ackUnreachable = seqDiff(p.nextSendSeq, b.lastSeq) > kAckWindow;
          need = resend = timedOut || ackUnreachable;
        }
        if (need) {
          if (p.tokens <= 0.0) return true;
          const size_t frameLen = kUnitFrameOverhead + b.len;
          if (!ensureRoom(p, frameLen)) return true;
          if (resend && ++b.retries > cfg.maxRetries) {
            p.pendingDisconnect = DisconnectReason::TooManyRetries;
            return false;
          }
          Wr w{pkt + pktLen, frameLen};
          w.u8(frameType);
          w.u16(b.unitSeq);
          w.u8(b.kind);
          w.u16(b.len);
          w.bytes(b.data, b.len);
          pktLen += frameLen;
          pktUnits[size_t(pktUnitCount++)] = bi;
          p.tokens -= double(frameLen);
        }
      }
      bi = next;
    }
    return true;
  }

  void sendUnreliable(Peer &p) {
    uint32_t bi = p.unrelHead;
    while (bi != kNoBlock) {
      Block &b = pool.at(bi);
      const uint32_t next = b.next;
      const size_t frameLen = kUnrelFrameOverhead + b.len;
      if (p.tokens > 0.0 && ensureRoom(p, frameLen)) {
        Wr w{pkt + pktLen, frameLen};
        w.u8(kFrameUnreliable);
        w.u16(b.len);
        w.bytes(b.data, b.len);
        pktLen += frameLen;
        p.tokens -= double(frameLen);
      } // else: dropped (budget exhausted) - unreliable by contract
      pool.release(bi);
      bi = next;
    }
    p.unrelHead = p.unrelTail = kNoBlock;
    p.unrelCount = 0;
  }

  void flushPeer(Peer &p) {
    const double dt = double(now - p.lastRefillMs);
    p.lastRefillMs = now;
    p.tokens = std::min(burst(), p.tokens + dt * double(cfg.bandwidthBytesPerSec) / 1000.0);
    packetsThisFlush = 0;
    beginPacket();
    // Order within the budget (NETWORK_PLAN §6.5): reliable -> unreliable -> bulk.
    if (!sendStream(p, 0)) return;
    sendUnreliable(p);
    if (!sendStream(p, 1)) return;
    if (pktLen > kHeaderSize) finishPacket(p);
    if (packetsThisFlush == 0 && (p.needAck || now - p.lastSendMs >= cfg.keepaliveMs))
      finishPacket(p); // ack-only / keepalive (header only)
  }

  static void writeRecord(Block &b, const uint8_t *d, size_t n) {
    b.data[b.len] = uint8_t(n);
    b.data[b.len + 1] = uint8_t(n >> 8);
    if (n) std::memcpy(b.data + b.len + 2, d, n);
    b.len = uint16_t(b.len + 2 + n);
  }

  bool queueUnreliable(Peer &p, const uint8_t *d, size_t n) {
    if (n + 2 > unitMax) return false; // unreliable messages must fit one packet
    if (p.unrelTail != kNoBlock) {
      Block &t = pool.at(p.unrelTail);
      if (t.len + 2 + n <= unitMax) {
        writeRecord(t, d, n);
        return true;
      }
    }
    if (p.unrelCount >= kMaxUnreliableBlocks) return false;
    const uint32_t bi = pool.alloc();
    if (bi == kNoBlock) return false;
    Block &b = pool.at(bi);
    b.kind = kUnitPacked;
    writeRecord(b, d, n);
    if (p.unrelTail == kNoBlock) p.unrelHead = bi;
    else pool.at(p.unrelTail).next = bi;
    p.unrelTail = bi;
    ++p.unrelCount;
    return true;
  }

  void linkUnit(SendStream &ss, uint32_t bi) {
    Block &b = pool.at(bi);
    b.unitSeq = ss.nextUnitSeq;
    ss.nextUnitSeq = uint16_t(ss.nextUnitSeq + 1);
    b.next = kNoBlock;
    if (ss.tail == kNoBlock) ss.head = bi;
    else pool.at(ss.tail).next = bi;
    ss.tail = bi;
  }

  bool queueStream(Peer &p, int s, const uint8_t *d, size_t n) {
    SendStream &ss = p.streams[s];
    if (ss.messages >= cfg.maxReliableBacklog) return false;
    const size_t cap = s == 1 ? size_t(cfg.maxBulkBytes) : std::max<size_t>(cfg.maxBulkBytes, 64u * 1024u);
    if (size_t(ss.bytes) + n > cap || n > 0x7FFFFFFFu) return false;
    if (n + 2 <= unitMax) { // small: pack into the open (never sent) tail unit
      if (ss.tail != kNoBlock) {
        Block &t = pool.at(ss.tail);
        if (t.kind == kUnitPacked && t.sendCount == 0 && t.len + 2 + n <= unitMax) {
          writeRecord(t, d, n);
          ++t.msgCount;
          ++ss.messages;
          ss.bytes += uint32_t(n + 2);
          return true;
        }
      }
      const uint32_t bi = pool.alloc();
      if (bi == kNoBlock) return false;
      Block &b = pool.at(bi);
      b.kind = kUnitPacked;
      writeRecord(b, d, n);
      b.msgCount = 1;
      linkUnit(ss, bi);
      ++ss.messages;
      ss.bytes += b.len;
      return true;
    }
    // Large: fragment into Start / Mid... / End units (always >= 2 units).
    uint32_t first = kNoBlock, last = kNoBlock;
    size_t off = 0;
    while (off < n) {
      const uint32_t bi = pool.alloc();
      if (bi == kNoBlock) {
        releaseChain(first); // all or nothing
        return false;
      }
      Block &b = pool.at(bi);
      size_t take;
      if (first == kNoBlock) {
        take = std::min(unitMax - 4, n - off);
        b.data[0] = uint8_t(n);
        b.data[1] = uint8_t(n >> 8);
        b.data[2] = uint8_t(n >> 16);
        b.data[3] = uint8_t(n >> 24);
        std::memcpy(b.data + 4, d + off, take);
        b.len = uint16_t(4 + take);
        b.kind = kUnitStart;
        first = bi;
      } else {
        take = std::min(unitMax, n - off);
        std::memcpy(b.data, d + off, take);
        b.len = uint16_t(take);
        b.kind = kUnitMid;
        pool.at(last).next = bi;
      }
      last = bi;
      off += take;
    }
    Block &end = pool.at(last);
    end.kind = kUnitEnd;
    end.msgCount = 1;
    for (uint32_t bi = first; bi != kNoBlock;) {
      const uint32_t next = pool.at(bi).next;
      ss.bytes += pool.at(bi).len;
      linkUnit(ss, bi);
      bi = next;
    }
    ++ss.messages;
    return true;
  }

  // ---- update ---------------------------------------------------------------------
  void update(uint64_t t) {
    now = std::max(now, t);
    if (mode == Mode::None) return;

    // Events: drop the consumed ones; keep undrained ones (their arena bytes stay).
    if (eventRead >= events.size()) {
      events.clear();
      arenaUsed = 0;
    } else if (eventRead > 0) {
      events.erase(events.begin(), events.begin() + std::ptrdiff_t(eventRead));
    }
    eventRead = 0;

    // Deliveries deferred because the arena was full.
    const size_t nBlocked = blocked.size();
    for (size_t i = 0; i < nBlocked; ++i) {
      PeerId id;
      id.value = blocked[i];
      if (Peer *p = lookup(id)) {
        p->deliveryBlocked = false;
        deliverAll(*p);
      }
    }
    blocked.erase(blocked.begin(), blocked.begin() + std::ptrdiff_t(nBlocked));

    for (int i = 0; i < kMaxReceivesPerUpdate; ++i) {
      Address from;
      const int n = socket.receive(recvBuf, int(sizeof(recvBuf)), from);
      if (n <= 0) break;
      handlePacket(from, recvBuf, size_t(n));
    }

    if (mode == Mode::Client &&
        (cstate == ClientState::SendingRequest || cstate == ClientState::SendingResponse)) {
      if (now - connectStartMs > cfg.timeoutMs) {
        clientFail(DisconnectReason::Timeout);
      } else {
        if (cstate == ClientState::SendingResponse && now - responseStartMs > kResponseRestartMs)
          cstate = ClientState::SendingRequest; // cookie may have expired: start over
        if (now - lastHandshakeMs >= kHandshakeResendMs) clientSendHandshake();
      }
    }

    // Backwards: freePeer swap-removes with the (already processed) last entry.
    for (size_t i = active.size(); i-- > 0;) {
      if (i >= active.size()) continue;
      Peer &p = peers[active[i]];
      if (p.ackDirty) {
        processAckedUnits(p);
        p.ackDirty = false;
      }
      if (p.pendingDisconnect != DisconnectReason::None) {
        freePeer(p, p.pendingDisconnect, true);
        continue;
      }
      if (now - p.lastRecvMs > cfg.timeoutMs) {
        freePeer(p, DisconnectReason::Timeout, true);
        continue;
      }
      flushPeer(p);
      if (p.pendingDisconnect != DisconnectReason::None) freePeer(p, p.pendingDisconnect, true);
    }
  }
};

// =============================================================================
// Host
// =============================================================================

Host::Host(const HostConfig &config) : impl_(std::make_unique<Impl>(config)) {}

Host::~Host() {
  if (!impl_) return;
  for (uint16_t index : impl_->active) {
    const Peer &p = impl_->peers[index];
    if (p.active) impl_->sendDisconnectPackets(p, DisconnectReason::Requested);
  }
}

bool Host::listen(uint16_t port, std::string *error) {
  Impl &m = *impl_;
  if (m.mode != Impl::Mode::None) {
    if (error) *error = "host already started";
    return false;
  }
  m.mode = Impl::Mode::Server;
  m.allocate(m.cfg.maxPeers);
  if (!m.socket.open(port, error)) {
    m.mode = Impl::Mode::None;
    return false;
  }
  return true;
}

bool Host::connect(const std::string &host, uint16_t port, std::string *error) {
  Impl &m = *impl_;
  if (m.mode != Impl::Mode::None) {
    if (error) *error = "host already started";
    return false;
  }
  detail::Address addr;
  if (!detail::UdpSocket::resolve(host, port, addr)) {
    if (error) *error = "cannot resolve " + host;
    return false;
  }
  m.mode = Impl::Mode::Client;
  m.allocate(1);
  if (!m.socket.open(0, error)) {
    m.mode = Impl::Mode::None;
    return false;
  }
  m.now = std::max(m.now, nowMs());
  m.serverAddr = addr;
  m.clientSalt = splitmix64(m.rng);
  m.cstate = Impl::ClientState::SendingRequest;
  m.connectStartMs = m.now;
  m.clientSendHandshake();
  return true;
}

void Host::update(uint64_t now) { impl_->update(now); }

bool Host::poll(Event &out) {
  Impl &m = *impl_;
  if (m.eventRead >= m.events.size()) return false;
  out = m.events[m.eventRead++];
  return true;
}

bool Host::send(PeerId peer, Channel channel, std::span<const uint8_t> data) {
  Impl &m = *impl_;
  Peer *p = m.lookup(peer);
  if (!p || p->pendingDisconnect != DisconnectReason::None) return false;
  switch (channel) {
  case Channel::Unreliable: return m.queueUnreliable(*p, data.data(), data.size());
  case Channel::ReliableOrdered: return m.queueStream(*p, 0, data.data(), data.size());
  case Channel::Bulk: return m.queueStream(*p, 1, data.data(), data.size());
  }
  return false;
}

void Host::disconnect(PeerId peer) {
  Impl &m = *impl_;
  if (Peer *p = m.lookup(peer)) {
    m.freePeer(*p, DisconnectReason::Requested, true);
    return;
  }
  if (m.mode == Impl::Mode::Client && (m.cstate == Impl::ClientState::SendingRequest ||
                                       m.cstate == Impl::ClientState::SendingResponse))
    m.clientFail(DisconnectReason::Requested);
}

PeerId Host::serverPeer() const {
  const Impl &m = *impl_;
  if (m.mode != Impl::Mode::Client || m.cstate != Impl::ClientState::Connected || m.peers.empty() ||
      !m.peers[0].active)
    return PeerId{};
  return Impl::idOf(0, m.peers[0]);
}

bool Host::isConnected(PeerId peer) const { return impl_->lookup(peer) != nullptr; }

PeerStats Host::stats(PeerId peer) const {
  PeerStats s;
  Impl &m = *impl_;
  Peer *p = m.lookup(peer);
  if (!p) return s;
  s.rttMs = p->srtt;
  s.lossPercent = p->loss;
  s.bytesSent = p->bytesSent;
  s.bytesReceived = p->bytesReceived;
  s.reliableQueued = p->streams[0].messages;
  s.bulkQueuedBytes = p->streams[1].bytes;
  uint32_t inFlight = 0;
  for (int st = 0; st < 2; ++st) {
    uint32_t count = 0;
    for (uint32_t bi = p->streams[st].head; bi != kNoBlock && count < m.window; ++count) {
      const Block &b = m.pool.at(bi);
      if (b.sendCount > 0 && !b.acked) ++inFlight;
      bi = b.next;
    }
  }
  s.inFlight = inFlight;
  return s;
}

uint16_t Host::localPort() const { return impl_->socket.localPort(); }
size_t Host::peerCount() const { return impl_->active.size(); }

HostStats Host::hostStats() const {
  HostStats s = impl_->hs;
  s.poolBlocks = impl_->pool.count();
  s.poolBlocksInUse = impl_->pool.inUse();
  s.poolBlocksMax = impl_->pool.maxBlocks();
  s.peers = uint32_t(impl_->active.size());
  return s;
}

} // namespace atm::net2
