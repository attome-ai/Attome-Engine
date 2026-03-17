#pragma once

#include "NetTypes.h"
#include "PacketPool.h"

#include <cstdint>
#include <span>

namespace attome::net {

struct SimSend {
  PoolIdx buf_idx{kInvalidPoolIdx};
  uint16_t len{};
  NetEndpoint endpoint{};
};

#if NET_SIM

class SimLayer {
public:
  struct Params {
    float send_loss{NET_SIM_SEND_LOSS};
    float recv_loss{NET_SIM_RECV_LOSS};
    int32_t latency_ms{NET_SIM_LATENCY_MS};
    int32_t jitter_ms{NET_SIM_JITTER_MS};
  };

  void init(uint32_t seed = 0);
  void set_params(const Params &p);
  Params params() const { return params_; }

  // Returns true if the packet was consumed by the sim layer (delayed or
  // dropped). Returns false if caller should send/process immediately.
  bool enqueue_send(PacketPool &pool, PoolIdx buf_idx, uint16_t len,
                    const NetEndpoint &endpoint, uint32_t now_ms);
  bool enqueue_recv(PacketPool &pool, PoolIdx buf_idx, uint16_t len,
                    const NetEndpoint &endpoint, uint32_t now_ms);

  // Drains entries with deliver_at_ms <= now_ms into the provided output spans.
  void drain_ready(PacketPool &pool, uint32_t now_ms, std::span<SimSend> send_out,
                   uint32_t &send_count, std::span<RawRecv> recv_out,
                   uint32_t &recv_count);

  // Removes all queued entries targeting this endpoint (disconnect cleanup).
  void purge_endpoint(PacketPool &pool, const NetEndpoint &endpoint);

private:
  enum : uint8_t { kKindSend = 0, kKindRecv = 1 };

  struct SimEntry {
    PoolIdx buf_idx{kInvalidPoolIdx};
    uint16_t len{};
    uint32_t deliver_at_ms{};
    NetEndpoint endpoint{};
    uint8_t kind{};
    uint8_t _pad0{};
    uint16_t _pad1{};
  };

  SimEntry ring_[NET_SIM_DELAY_QUEUE_SIZE]{};
  uint32_t head_{0};
  uint32_t tail_{0};
  uint32_t rng_{0xA1B2C3D4u};

  uint32_t next_u32_();
  float next_f01_();
  bool should_drop_(float p);
  int32_t jitter_ms_sample_();
  void apply_env_overrides_();

  void drop_entry_(PacketPool &pool, const SimEntry &e);
  void push_entry_(PacketPool &pool, const SimEntry &e);
  bool endpoint_eq_(const NetEndpoint &a, const NetEndpoint &b) const;

  Params params_{};
};

#else

class SimLayer {
public:
  void init(uint32_t = 0) {}
  bool enqueue_send(PacketPool &, PoolIdx, uint16_t, const NetEndpoint &,
                    uint32_t) {
    return false;
  }
  bool enqueue_recv(PacketPool &, PoolIdx, uint16_t, const NetEndpoint &,
                    uint32_t) {
    return false;
  }
  void drain_ready(PacketPool &, uint32_t, std::span<SimSend>,
                   uint32_t &send_count, std::span<RawRecv>,
                   uint32_t &recv_count) {
    send_count = 0;
    recv_count = 0;
  }
  void purge_endpoint(PacketPool &, const NetEndpoint &) {}
};

#endif

} // namespace attome::net
