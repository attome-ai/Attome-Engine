#pragma once

// AttomeNet 2 transport (docs/NETWORK_PLAN.md). This is the demo-scale core:
// correct and memory-bounded first, then the multi-threaded batched I/O of
// milestone N1 goes behind the same API.
//
// Fixes for the defects found in the old library (NETWORK_PLAN §3.1) are
// part of the contract:
//   - PeerId = 16-bit index + 16-bit generation (no stale-slot delivery, D7)
//   - stateless challenge cookie before any per-peer state (D2)
//   - idle timeout + explicit disconnect (D2)
//   - ack window == max in-flight window (128) (D4)
//   - sequence assigned only after the packet is recorded (D5)
//   - reliable data acked only after it is queued for the application (D6)
//   - bounded memory: per-peer in-flight cap, retry cap -> disconnect (D3)
//   - "nothing received yet" is a flag, not a sequence value (D11)
//
// Threading: single-threaded; call update() from one thread. Non-blocking.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace atm::net2 {

struct PeerId {
  uint32_t value = 0xFFFFFFFFu; // (generation << 16) | index
  uint16_t index() const { return uint16_t(value & 0xFFFFu); }
  uint16_t generation() const { return uint16_t(value >> 16); }
  bool valid() const { return value != 0xFFFFFFFFu; }
  friend bool operator==(PeerId a, PeerId b) { return a.value == b.value; }
};

enum class Channel : uint8_t {
  Unreliable = 0,       // snapshots, input (loss handled by the game protocol)
  ReliableOrdered = 1,  // game events, inventory, chat, block edits
  Bulk = 2,             // reliable ordered, lowest priority (chunk data); may exceed MTU
};

struct HostConfig {
  uint16_t maxPeers = 64;            // server: up to 65535; client: 1
  uint32_t protocolId = 0;           // game protocol hash; mismatches are rejected
  uint32_t timeoutMs = 10000;        // silence before disconnect
  uint32_t keepaliveMs = 250;        // send something at least this often
  uint16_t mtu = 1200;               // max datagram payload
  uint16_t maxInFlight = 128;        // reliable packets awaiting ack, per peer
  uint16_t maxRetries = 20;          // then disconnect (peer is gone or stalled)
  uint32_t maxBulkBytes = 4u << 20;  // queued bulk data per peer
  uint32_t maxReliableBacklog = 1024;// queued reliable messages per peer
  uint32_t bandwidthBytesPerSec = 256u * 1024; // per-peer send budget
  // Shared packet-block pool cap (blocks of ~1.4 KB, grown lazily in small
  // slabs, never shrunk). 0 = auto: clamp(maxPeers * 1024, 4096, 131072).
  uint32_t maxPoolBlocks = 0;
  uint32_t initialPoolBlocks = 0;    // 0 = auto (64 per peer, max 4096)
};

enum class EventType : uint8_t { Connected, Disconnected, Message };

enum class DisconnectReason : uint8_t {
  None, Timeout, Requested, Rejected, ProtocolMismatch, ServerFull, TooManyRetries, Backlog,
};

struct Event {
  EventType type = EventType::Message;
  PeerId peer;
  Channel channel = Channel::Unreliable;
  DisconnectReason reason = DisconnectReason::None;
  // Message payload. Valid until the next call to update(). Drain poll()
  // before calling update() again (undrained events are kept, but their
  // storage then blocks further reliable delivery until drained).
  std::span<const uint8_t> data;
};

struct PeerStats {
  float rttMs = 0.0f;
  float lossPercent = 0.0f;
  uint64_t bytesSent = 0, bytesReceived = 0;
  uint32_t inFlight = 0, reliableQueued = 0, bulkQueuedBytes = 0;
};

// Whole-host counters (for server stats / load tests).
struct HostStats {
  uint64_t bytesSent = 0, bytesReceived = 0;
  uint64_t packetsSent = 0, packetsReceived = 0;
  uint64_t malformedPackets = 0;     // dropped: bad header/frames/cookie
  uint32_t poolBlocks = 0, poolBlocksInUse = 0, poolBlocksMax = 0;
  uint32_t peers = 0;
};

class Host {
public:
  explicit Host(const HostConfig &config);
  ~Host();
  Host(const Host &) = delete;
  Host &operator=(const Host &) = delete;

  // Server: listen on a UDP port (0 = any). Client: connect to host:port.
  bool listen(uint16_t port, std::string *error = nullptr);
  bool connect(const std::string &host, uint16_t port, std::string *error = nullptr);

  // Pumps the socket: receive, acks, resends, timeouts, sends queued data.
  void update(uint64_t nowMs);
  // Retrieves the next event produced by update(); false when none remain.
  bool poll(Event &out);

  // Queues a message. Unreliable messages must fit in one packet; reliable
  // and bulk messages may be larger (fragmented). Returns false when the
  // peer's queue is full (caller decides: drop, retry, or disconnect).
  bool send(PeerId peer, Channel channel, std::span<const uint8_t> data);
  void disconnect(PeerId peer);

  // Client convenience: the single server peer (invalid until connected).
  PeerId serverPeer() const;
  bool isConnected(PeerId peer) const;
  PeerStats stats(PeerId peer) const;
  uint16_t localPort() const;
  size_t peerCount() const;
  HostStats hostStats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

uint64_t nowMs(); // monotonic milliseconds

} // namespace atm::net2
