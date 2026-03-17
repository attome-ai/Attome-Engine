#pragma once

#include "NetTypes.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

namespace attome::net {

struct ConnectionSlots {
  static_assert(NET_MAX_CONNECTIONS <= UINT16_MAX,
                "ConnId is uint16; NET_MAX_CONNECTIONS must fit.");
  static_assert(NET_RETRANSMIT_SLOTS > 0,
                "NET_RETRANSMIT_SLOTS must be > 0.");
  static_assert((NET_RETRANSMIT_SLOTS & (NET_RETRANSMIT_SLOTS - 1)) == 0,
                "NET_RETRANSMIT_SLOTS must be a power of 2.");
  static_assert(NET_RETRANSMIT_SLOTS <= 256,
                "Retransmit head/tail are uint8; NET_RETRANSMIT_SLOTS must be <= 256.");
  static_assert(NET_REORDER_BUF_SIZE > 0, "NET_REORDER_BUF_SIZE must be > 0.");
  static_assert((NET_REORDER_BUF_SIZE & (NET_REORDER_BUF_SIZE - 1)) == 0,
                "NET_REORDER_BUF_SIZE must be a power of 2.");
  static_assert(NET_CONTEXT_SIZE > 0, "NET_CONTEXT_SIZE must be > 0.");

  static constexpr uint8_t kReliableChannelCount = 2; // CH1 + CH2

  static constexpr uint32_t kReliableUnordRecvWindowBits =
      (NET_RETRANSMIT_SLOTS < 256) ? (NET_RETRANSMIT_SLOTS * 2u)
                                   : static_cast<uint32_t>(NET_RETRANSMIT_SLOTS);
  static_assert(kReliableUnordRecvWindowBits >= NET_ACK_BITS_COUNT,
                "Reliable receive window must be >= ACK window.");
  static_assert(kReliableUnordRecvWindowBits <= 256,
                "Reliable receive window must stay <= 256 bits.");
  static_assert((kReliableUnordRecvWindowBits % 32u) == 0,
                "Reliable receive window must be a multiple of 32 bits.");
  static constexpr uint32_t kReliableUnordRecvWindowWords =
      kReliableUnordRecvWindowBits / 32u;

  void init();

  ConnId alloc_slot(const NetEndpoint &endpoint,
                    uint64_t connect_time_ms = 0);
  void free_slot(ConnId conn);

  inline bool is_alive(ConnId conn) const {
    return (conn != kInvalidConnId) && (conn < NET_MAX_CONNECTIONS) &&
           (alive[conn] != 0);
  }

  template <typename T, typename... Args>
  T *emplace_context(ConnId conn, Args &&...args) {
    static_assert(sizeof(T) <= NET_CONTEXT_SIZE,
                  "Context type too large for NET_CONTEXT_SIZE.");
    static_assert(alignof(T) <= 16,
                  "Context alignment > 16 not supported by context_buf.");

    if (!is_alive(conn)) {
      return nullptr;
    }

    clear_context(conn);

    void *p = static_cast<void *>(&context_buf[conn][0]);
    T *obj = new (p) T(std::forward<Args>(args)...);
    context_dtor[conn] = [](void *q) { static_cast<T *>(q)->~T(); };
    return obj;
  }

  template <typename T> T *get_context(ConnId conn) {
    if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
      return nullptr;
    }
    if (context_dtor[conn] == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<T *>(&context_buf[conn][0]);
  }

  void clear_context(ConnId conn) {
    if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
      return;
    }
    auto *dtor = context_dtor[conn];
    if (dtor != nullptr) {
      dtor(static_cast<void *>(&context_buf[conn][0]));
      context_dtor[conn] = nullptr;
    }
    std::memset(&context_buf[conn][0], 0, NET_CONTEXT_SIZE);
  }

  static inline bool is_reliable_channel(uint8_t ch) {
    return ch == NET_CHANNEL_RELIABLE_UNORD || ch == NET_CHANNEL_RELIABLE_ORD;
  }

  static inline uint8_t reliable_index(uint8_t ch) {
    if (ch == NET_CHANNEL_RELIABLE_UNORD) {
      return 0;
    }
    if (ch == NET_CHANNEL_RELIABLE_ORD) {
      return 1;
    }
    return UINT8_MAX;
  }

  // HOT (alignas(64))
  alignas(64) uint8_t alive[NET_MAX_CONNECTIONS]{};
  alignas(64) uint16_t send_seq[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]{};
  alignas(64) uint16_t recv_ack[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]{};
  alignas(64) uint32_t recv_ack_bits[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]{};
  // Extended receive history for CH1 (reliable unordered) duplicate suppression.
  // recv_ack_bits holds only the low 32 bits used on-wire; this holds the full
  // receive window (NET_RETRANSMIT_SLOTS bits) so late retransmits are dropped.
  alignas(64) uint32_t recv_rel_unord_bits[NET_MAX_CONNECTIONS]
                                         [kReliableUnordRecvWindowWords]{};
  alignas(64) uint16_t reorder_next_exp[NET_MAX_CONNECTIONS]{};
  alignas(64) uint8_t retransmit_head[NET_MAX_CONNECTIONS][kReliableChannelCount]{};
  alignas(64) uint8_t retransmit_tail[NET_MAX_CONNECTIONS][kReliableChannelCount]{};

  // COLD
  NetEndpoint endpoints[NET_MAX_CONNECTIONS]{};
  uint64_t connect_time_ms[NET_MAX_CONNECTIONS]{};

#if NET_ENCRYPTION
  uint8_t encryption_key[NET_MAX_CONNECTIONS][32]{};
  uint32_t encryption_conn_id[NET_MAX_CONNECTIONS]{};
  uint8_t encryption_ready[NET_MAX_CONNECTIONS]{};
#endif

  alignas(16) uint8_t context_buf[NET_MAX_CONNECTIONS][NET_CONTEXT_SIZE]{};
  void (*context_dtor[NET_MAX_CONNECTIONS])(void *){};

  RetransmitSlot retransmit_slots[NET_MAX_CONNECTIONS][kReliableChannelCount]
                                 [NET_RETRANSMIT_SLOTS]{};
  ReorderEntry reorder_buf[NET_MAX_CONNECTIONS][NET_REORDER_BUF_SIZE]{};

  uint16_t free_slot_stack[NET_MAX_CONNECTIONS]{};
  int32_t free_slot_top{-1};
};

} // namespace attome::net
