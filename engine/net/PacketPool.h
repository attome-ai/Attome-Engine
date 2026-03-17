#pragma once

#include "NetTypes.h"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace attome::net {

class PacketPool {
public:
  void init();

  PoolIdx acquire();
  void release(PoolIdx idx);

  uint8_t *ptr(PoolIdx idx);
  const uint8_t *ptr(PoolIdx idx) const;

  // Returns the pool index for a pointer into the internal buffer array, or
  // kInvalidPoolIdx if out of range. Used to release CH0 send buffers.
  PoolIdx idx_from_ptr(const uint8_t *p) const;

#if NET_SIM
  // Tracks buffers queued for delayed send in SimLayer. Without this, a buffer
  // can be ACKed/freed while a delayed send entry still references it.
  void sim_send_ref_inc(PoolIdx idx);
  void sim_send_ref_dec(PoolIdx idx);
#endif

private:
  void release_now_(PoolIdx idx);

  alignas(64) uint8_t buffers_[NET_PACKET_POOL_SIZE][NET_MTU];
  alignas(64) uint16_t free_stack_[NET_PACKET_POOL_SIZE];
  // Tagged head index to avoid ABA in the lock-free freelist.
  // Layout: [tag:48][idx:16], idx=0xFFFF means empty.
  std::atomic<uint64_t> free_top_{0};

#if NET_SIM
  // Debug in-use tracking to catch double-free / use-after-free under stress.
  static_assert((NET_PACKET_POOL_SIZE % 32) == 0,
                "NET_PACKET_POOL_SIZE must be a multiple of 32 for in_use_bits_.");
  alignas(64) std::atomic<uint32_t> in_use_bits_[NET_PACKET_POOL_SIZE / 32]{};

  // Reference count per pool buffer for pending delayed sends (SimLayer).
  alignas(64) std::atomic<uint32_t> sim_send_refs_[NET_PACKET_POOL_SIZE]{};
  // Deferred free flag set when release() is called while sim_send_refs_ > 0.
  alignas(64) std::atomic<uint8_t> sim_send_defer_free_[NET_PACKET_POOL_SIZE]{};
#endif
};

} // namespace attome::net
