#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace attome::net {

template <typename T, uint32_t N> class SpscQueue {
  static_assert(N > 0, "SpscQueue N must be > 0.");
  static_assert((N & (N - 1)) == 0, "SpscQueue N must be a power of 2.");

public:
  bool push(const T &value) {
    const uint32_t tail = tail_.load(std::memory_order_relaxed);
    const uint32_t head = head_.load(std::memory_order_acquire);
    if ((tail - head) == N) {
      return false;
    }

    ring_[tail & (N - 1)] = value;
    tail_.store(tail + 1, std::memory_order_release);
    return true;
  }

  bool pop(T &out) {
    const uint32_t head = head_.load(std::memory_order_relaxed);
    const uint32_t tail = tail_.load(std::memory_order_acquire);
    if (head == tail) {
      return false;
    }

    out = ring_[head & (N - 1)];
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  void clear() {
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
  }

private:
  alignas(64) std::atomic<uint32_t> head_{0};
  alignas(64) std::atomic<uint32_t> tail_{0};
  T ring_[N];
};

} // namespace attome::net
