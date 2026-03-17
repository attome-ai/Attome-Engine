#include "PacketPool.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace attome::net {

namespace {

inline constexpr uint64_t kPoolHeadIdxMask = 0xFFFFull;
inline constexpr uint32_t kPoolHeadTagShift = 16;

constexpr uint64_t pack_pool_head(uint64_t tag, PoolIdx idx) {
  return (tag << kPoolHeadTagShift) | static_cast<uint64_t>(idx);
}

constexpr PoolIdx pool_head_idx(uint64_t head) {
  return static_cast<PoolIdx>(head & kPoolHeadIdxMask);
}

constexpr uint64_t pool_head_tag(uint64_t head) { return (head >> kPoolHeadTagShift); }

} // namespace

void PacketPool::init() {
  static_assert(NET_PACKET_POOL_SIZE <= UINT16_MAX,
                "PacketPool freelist uses uint16 indices.");

  for (uint16_t i = 0; i < NET_PACKET_POOL_SIZE; ++i) {
    free_stack_[i] = (i == 0) ? kInvalidPoolIdx : static_cast<uint16_t>(i - 1);
  }

  free_top_.store(pack_pool_head(0, static_cast<PoolIdx>(NET_PACKET_POOL_SIZE - 1)),
                  std::memory_order_release);

#if NET_SIM
  for (uint32_t i = 0; i < (NET_PACKET_POOL_SIZE / 32); ++i) {
    in_use_bits_[i].store(0, std::memory_order_relaxed);
  }
  for (uint32_t i = 0; i < NET_PACKET_POOL_SIZE; ++i) {
    sim_send_refs_[i].store(0, std::memory_order_relaxed);
    sim_send_defer_free_[i].store(0, std::memory_order_relaxed);
  }
#endif
}

PoolIdx PacketPool::acquire() {
  uint64_t head = free_top_.load(std::memory_order_acquire);
  while (true) {
    const PoolIdx idx = pool_head_idx(head);
    if (idx == kInvalidPoolIdx) {
      return kInvalidPoolIdx;
    }
    if (idx >= NET_PACKET_POOL_SIZE) {
      std::fprintf(stderr, "PacketPool corrupted head idx=%u\n",
                   static_cast<unsigned>(idx));
      std::abort();
    }

    const PoolIdx next = free_stack_[idx];
    const uint64_t desired = pack_pool_head(pool_head_tag(head) + 1, next);

    if (free_top_.compare_exchange_weak(head, desired, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
#if NET_SIM
      const uint32_t word = static_cast<uint32_t>(idx) / 32u;
      const uint32_t bit = static_cast<uint32_t>(idx) % 32u;
      const uint32_t mask = (1u << bit);
      const uint32_t prev =
          in_use_bits_[word].fetch_or(mask, std::memory_order_relaxed);
      if ((prev & mask) != 0u) {
        std::fprintf(stderr, "PacketPool double-acquire idx=%u\n",
                     static_cast<unsigned>(idx));
        std::abort();
      }
#endif
      return idx;
    }
  }
}

void PacketPool::release_now_(PoolIdx idx) {
#if NET_SIM
  {
    const uint32_t word = static_cast<uint32_t>(idx) / 32u;
    const uint32_t bit = static_cast<uint32_t>(idx) % 32u;
    const uint32_t mask = (1u << bit);
    const uint32_t prev =
        in_use_bits_[word].fetch_and(~mask, std::memory_order_relaxed);
    if ((prev & mask) == 0u) {
      std::fprintf(stderr, "PacketPool double-free idx=%u\n",
                   static_cast<unsigned>(idx));
      std::abort();
    }
  }
#endif

  uint64_t head = free_top_.load(std::memory_order_acquire);
  while (true) {
    free_stack_[idx] = pool_head_idx(head);

    const uint64_t desired = pack_pool_head(pool_head_tag(head) + 1, idx);

    if (free_top_.compare_exchange_weak(head, desired, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
      return;
    }
  }
}

void PacketPool::release(PoolIdx idx) {
  if (idx == kInvalidPoolIdx) {
    return;
  }
  if (idx >= NET_PACKET_POOL_SIZE) {
    std::fprintf(stderr, "PacketPool release out of range idx=%u\n",
                 static_cast<unsigned>(idx));
    std::abort();
  }

#if NET_SIM
  if (sim_send_refs_[idx].load(std::memory_order_relaxed) != 0u) {
    sim_send_defer_free_[idx].store(1, std::memory_order_relaxed);
    return;
  }
#endif

  release_now_(idx);
}

#if NET_SIM

void PacketPool::sim_send_ref_inc(PoolIdx idx) {
  if (idx == kInvalidPoolIdx) {
    return;
  }
  if (idx >= NET_PACKET_POOL_SIZE) {
    std::fprintf(stderr, "PacketPool sim_send_ref_inc out of range idx=%u\n",
                 static_cast<unsigned>(idx));
    std::abort();
  }

  sim_send_refs_[idx].fetch_add(1, std::memory_order_relaxed);
}

void PacketPool::sim_send_ref_dec(PoolIdx idx) {
  if (idx == kInvalidPoolIdx) {
    return;
  }
  if (idx >= NET_PACKET_POOL_SIZE) {
    std::fprintf(stderr, "PacketPool sim_send_ref_dec out of range idx=%u\n",
                 static_cast<unsigned>(idx));
    std::abort();
  }

  const uint32_t prev = sim_send_refs_[idx].fetch_sub(1, std::memory_order_relaxed);
  if (prev == 0u) {
    std::fprintf(stderr, "PacketPool sim_send_ref_dec underflow idx=%u\n",
                 static_cast<unsigned>(idx));
    std::abort();
  }

  if (prev == 1u) {
    if (sim_send_defer_free_[idx].exchange(0, std::memory_order_relaxed) != 0u) {
      release_now_(idx);
    }
  }
}

#endif

uint8_t *PacketPool::ptr(PoolIdx idx) {
  assert(idx != kInvalidPoolIdx);
  assert(idx < NET_PACKET_POOL_SIZE);
  return &buffers_[idx][0];
}

const uint8_t *PacketPool::ptr(PoolIdx idx) const {
  assert(idx != kInvalidPoolIdx);
  assert(idx < NET_PACKET_POOL_SIZE);
  return &buffers_[idx][0];
}

PoolIdx PacketPool::idx_from_ptr(const uint8_t *p) const {
  if (p == nullptr) {
    return kInvalidPoolIdx;
  }

  const uint8_t *base = &buffers_[0][0];
  const uint8_t *end = base + (static_cast<size_t>(NET_PACKET_POOL_SIZE) *
                               static_cast<size_t>(NET_MTU));
  if (p < base || p >= end) {
    return kInvalidPoolIdx;
  }

  const size_t diff = static_cast<size_t>(p - base);
  const size_t idx = diff / static_cast<size_t>(NET_MTU);
  if (idx >= static_cast<size_t>(NET_PACKET_POOL_SIZE)) {
    return kInvalidPoolIdx;
  }
  return static_cast<PoolIdx>(idx);
}

} // namespace attome::net
