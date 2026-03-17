#include "SimLayer.h"

#if NET_SIM

#include <cerrno>
#include <cstring>
#include <cstdlib>

namespace attome::net {

void SimLayer::init(uint32_t seed) {
  std::memset(ring_, 0, sizeof(ring_));
  head_ = 0;
  tail_ = 0;
  rng_ = (seed != 0) ? seed : 0xA1B2C3D4u;

  params_ = Params{};
  apply_env_overrides_();
}

void SimLayer::set_params(const Params &p) {
  params_ = p;
  if (params_.send_loss < 0.0f) {
    params_.send_loss = 0.0f;
  }
  if (params_.send_loss > 1.0f) {
    params_.send_loss = 1.0f;
  }
  if (params_.recv_loss < 0.0f) {
    params_.recv_loss = 0.0f;
  }
  if (params_.recv_loss > 1.0f) {
    params_.recv_loss = 1.0f;
  }
  if (params_.latency_ms < 0) {
    params_.latency_ms = 0;
  }
  if (params_.jitter_ms < 0) {
    params_.jitter_ms = 0;
  }
}

uint32_t SimLayer::next_u32_() {
  // xorshift32
  uint32_t x = rng_;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  rng_ = x;
  return x;
}

float SimLayer::next_f01_() {
  // 24-bit mantissa -> [0,1)
  const uint32_t v = next_u32_() >> 8;
  return static_cast<float>(v) * (1.0f / 16777216.0f);
}

bool SimLayer::should_drop_(float p) {
  if (p <= 0.0f) {
    return false;
  }
  if (p >= 1.0f) {
    return true;
  }
  return next_f01_() < p;
}

int32_t SimLayer::jitter_ms_sample_() {
  if (params_.jitter_ms <= 0) {
    return 0;
  }
  const uint32_t span = static_cast<uint32_t>(params_.jitter_ms * 2 + 1);
  const int32_t r = static_cast<int32_t>(next_u32_() % span);
  return r - params_.jitter_ms;
}

void SimLayer::apply_env_overrides_() {
  // These are test/dev-only knobs (NET_SIM builds). They allow cranking loss /
  // latency without creating separate build directories.
  auto parse_f = [](const char *s, float &out) -> bool {
    if (s == nullptr || s[0] == '\0') {
      return false;
    }
    errno = 0;
    char *end = nullptr;
    const float v = std::strtof(s, &end);
    if (end == s || errno != 0) {
      return false;
    }
    out = v;
    return true;
  };

  auto parse_i = [](const char *s, int32_t &out) -> bool {
    if (s == nullptr || s[0] == '\0') {
      return false;
    }
    errno = 0;
    char *end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || errno != 0) {
      return false;
    }
    out = static_cast<int32_t>(v);
    return true;
  };

  Params p = params_;
  (void)parse_f(std::getenv("ATTOME_NET_SIM_SEND_LOSS"), p.send_loss);
  (void)parse_f(std::getenv("ATTOME_NET_SIM_RECV_LOSS"), p.recv_loss);
  (void)parse_i(std::getenv("ATTOME_NET_SIM_LATENCY_MS"), p.latency_ms);
  (void)parse_i(std::getenv("ATTOME_NET_SIM_JITTER_MS"), p.jitter_ms);
  set_params(p);
}

bool SimLayer::endpoint_eq_(const NetEndpoint &a, const NetEndpoint &b) const {
  if (a.port != b.port) {
    return false;
  }
  if (a.is_v6 != b.is_v6) {
    return false;
  }
  if (a.is_v6) {
    if (a.v6_scope_id != b.v6_scope_id) {
      return false;
    }
    return std::memcmp(a.addr.v6, b.addr.v6, 16) == 0;
  }
  return a.addr.v4 == b.addr.v4;
}

void SimLayer::drop_entry_(PacketPool &pool, const SimEntry &e) {
  if (e.buf_idx == kInvalidPoolIdx) {
    return;
  }

  if (e.kind == kKindRecv) {
    pool.release(e.buf_idx);
    return;
  }

  // Send entries: drop the sim reference, and only release if unreliable (CH0)
  // or an untracked ACK-only datagram. Reliable buffers are owned by retransmit
  // rings.
  const uint8_t ch = pool.ptr(e.buf_idx)[0];
  const bool is_ack_only =
      (e.len == NET_HEADER_SIZE)
#if NET_ENCRYPTION
      || (e.len == (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE))
#endif
      ;

  pool.sim_send_ref_dec(e.buf_idx);
  if (ch == NET_CHANNEL_UNRELIABLE || is_ack_only) {
    pool.release(e.buf_idx);
  }
}

void SimLayer::push_entry_(PacketPool &pool, const SimEntry &e) {
  const uint32_t used = tail_ - head_;
  const uint32_t cap = NET_SIM_DELAY_QUEUE_SIZE;
  const uint32_t mask = cap - 1;

  if (used >= cap) {
    const SimEntry &old = ring_[head_ & mask];
    drop_entry_(pool, old);
    head_ += 1;
  }

  ring_[tail_ & mask] = e;
  tail_ += 1;
}

bool SimLayer::enqueue_send(PacketPool &pool, PoolIdx buf_idx, uint16_t len,
                            const NetEndpoint &endpoint, uint32_t now_ms) {
  if (buf_idx == kInvalidPoolIdx) {
    return true;
  }
  if (should_drop_(params_.send_loss)) {
    // Only release if CH0 or an untracked ACK-only datagram.
    const uint8_t ch = pool.ptr(buf_idx)[0];
    const bool is_ack_only =
        (len == NET_HEADER_SIZE)
#if NET_ENCRYPTION
        || (len == (NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE))
#endif
        ;
    if (ch == NET_CHANNEL_UNRELIABLE || is_ack_only) {
      pool.release(buf_idx);
    }
    return true;
  }

  const int32_t delay = params_.latency_ms + jitter_ms_sample_();
  uint32_t deliver_at = now_ms;
  if (delay > 0) {
    deliver_at = static_cast<uint32_t>(now_ms + static_cast<uint32_t>(delay));
  }
  if (deliver_at <= now_ms) {
    return false; // immediate
  }

  pool.sim_send_ref_inc(buf_idx);
  push_entry_(pool, SimEntry{
                        .buf_idx = buf_idx,
                        .len = len,
                        .deliver_at_ms = deliver_at,
                        .endpoint = endpoint,
                        .kind = kKindSend,
                    });
  return true;
}

bool SimLayer::enqueue_recv(PacketPool &pool, PoolIdx buf_idx, uint16_t len,
                            const NetEndpoint &endpoint, uint32_t now_ms) {
  if (buf_idx == kInvalidPoolIdx) {
    return true;
  }
  if (should_drop_(params_.recv_loss)) {
    pool.release(buf_idx);
    return true;
  }

  const int32_t delay = params_.latency_ms + jitter_ms_sample_();
  uint32_t deliver_at = now_ms;
  if (delay > 0) {
    deliver_at = static_cast<uint32_t>(now_ms + static_cast<uint32_t>(delay));
  }
  if (deliver_at <= now_ms) {
    return false; // immediate
  }

  push_entry_(pool, SimEntry{
                        .buf_idx = buf_idx,
                        .len = len,
                        .deliver_at_ms = deliver_at,
                        .endpoint = endpoint,
                        .kind = kKindRecv,
                    });
  return true;
}

void SimLayer::drain_ready(PacketPool &pool, uint32_t now_ms,
                           std::span<SimSend> send_out,
                           uint32_t &send_count, std::span<RawRecv> recv_out,
                           uint32_t &recv_count) {
  send_count = 0;
  recv_count = 0;

  const uint32_t cap = NET_SIM_DELAY_QUEUE_SIZE;
  const uint32_t mask = cap - 1;

  while (head_ != tail_) {
    const SimEntry &e = ring_[head_ & mask];
    if (e.deliver_at_ms > now_ms) {
      break;
    }

    if (e.kind == kKindSend) {
      if (send_count < send_out.size()) {
        send_out[send_count++] = SimSend{
            .buf_idx = e.buf_idx,
            .len = e.len,
            .endpoint = e.endpoint,
        };
      } else {
        drop_entry_(pool, e);
      }
    } else {
      if (recv_count < recv_out.size()) {
        recv_out[recv_count++] = RawRecv{
            .buf_idx = e.buf_idx,
            .len = e.len,
            .endpoint = e.endpoint,
        };
      } else {
        drop_entry_(pool, e);
      }
    }

    head_ += 1;
  }
}

void SimLayer::purge_endpoint(PacketPool &pool, const NetEndpoint &endpoint) {
  // Rebuild ring in-place (N is small, O(N) is fine on cold path).
  const uint32_t cap = NET_SIM_DELAY_QUEUE_SIZE;
  const uint32_t mask = cap - 1;

  uint32_t new_head = 0;
  uint32_t new_tail = 0;
  SimEntry tmp[NET_SIM_DELAY_QUEUE_SIZE]{};

  for (uint32_t i = head_; i != tail_; ++i) {
    const SimEntry &e = ring_[i & mask];
    if (endpoint_eq_(e.endpoint, endpoint)) {
      drop_entry_(pool, e);
      continue;
    }
    tmp[new_tail & mask] = e;
    new_tail += 1;
  }

  std::memcpy(ring_, tmp, sizeof(ring_));
  head_ = new_head;
  tail_ = new_tail;
}

} // namespace attome::net

#endif
