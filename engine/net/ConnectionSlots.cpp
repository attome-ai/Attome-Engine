#include "ConnectionSlots.h"

#include <cassert>

namespace attome::net {

void ConnectionSlots::init() {
  std::memset(alive, 0, sizeof(alive));
  std::memset(send_seq, 0, sizeof(send_seq));
  std::memset(recv_ack, 0xFF, sizeof(recv_ack)); // 0xFFFF = none received yet
  std::memset(recv_ack_bits, 0, sizeof(recv_ack_bits));
  std::memset(recv_rel_unord_bits, 0, sizeof(recv_rel_unord_bits));
  std::memset(reorder_next_exp, 0, sizeof(reorder_next_exp));
  std::memset(retransmit_head, 0, sizeof(retransmit_head));
  std::memset(retransmit_tail, 0, sizeof(retransmit_tail));
  std::memset(endpoints, 0, sizeof(endpoints));
  std::memset(connect_time_ms, 0, sizeof(connect_time_ms));

#if NET_ENCRYPTION
  std::memset(encryption_key, 0, sizeof(encryption_key));
  std::memset(encryption_conn_id, 0, sizeof(encryption_conn_id));
  std::memset(encryption_ready, 0, sizeof(encryption_ready));
#endif

  std::memset(context_buf, 0, sizeof(context_buf));
  std::memset(context_dtor, 0, sizeof(context_dtor));

  for (uint32_t c = 0; c < NET_MAX_CONNECTIONS; ++c) {
    for (uint32_t i = 0; i < kReliableChannelCount; ++i) {
      for (uint32_t s = 0; s < NET_RETRANSMIT_SLOTS; ++s) {
        retransmit_slots[c][i][s].buf_idx = kInvalidPoolIdx;
        retransmit_slots[c][i][s].conn = kInvalidConnId;
        retransmit_slots[c][i][s].seq = 0;
        retransmit_slots[c][i][s].sent_at_ms = 0;
        retransmit_slots[c][i][s].len = 0;
      }
    }

    for (uint32_t s = 0; s < NET_REORDER_BUF_SIZE; ++s) {
      reorder_buf[c][s].buf_idx = kInvalidPoolIdx;
      reorder_buf[c][s].seq = 0;
      reorder_buf[c][s].len = 0;
      reorder_buf[c][s].occupied = false;
    }
  }

  for (uint16_t i = 0; i < NET_MAX_CONNECTIONS; ++i) {
    free_slot_stack[i] =
        static_cast<uint16_t>((NET_MAX_CONNECTIONS - 1) - i);
  }
  free_slot_top = static_cast<int32_t>(NET_MAX_CONNECTIONS - 1);
}

ConnId ConnectionSlots::alloc_slot(const NetEndpoint &endpoint,
                                   uint64_t connect_time) {
  if (free_slot_top < 0) {
    return kInvalidConnId;
  }

  const ConnId conn =
      static_cast<ConnId>(free_slot_stack[free_slot_top--]);
  assert(conn != kInvalidConnId);
  assert(conn < NET_MAX_CONNECTIONS);

  alive[conn] = 1;
  endpoints[conn] = endpoint;
  connect_time_ms[conn] = connect_time;

  std::memset(send_seq[conn], 0, sizeof(send_seq[conn]));
  std::memset(recv_ack[conn], 0xFF, sizeof(recv_ack[conn]));
  std::memset(recv_ack_bits[conn], 0, sizeof(recv_ack_bits[conn]));
  std::memset(recv_rel_unord_bits[conn], 0, sizeof(recv_rel_unord_bits[conn]));
  reorder_next_exp[conn] = 0;
  std::memset(retransmit_head[conn], 0, sizeof(retransmit_head[conn]));
  std::memset(retransmit_tail[conn], 0, sizeof(retransmit_tail[conn]));

#if NET_ENCRYPTION
  std::memset(encryption_key[conn], 0, sizeof(encryption_key[conn]));
  encryption_conn_id[conn] = 0;
  encryption_ready[conn] = 0;
#endif

  clear_context(conn);

  for (uint32_t i = 0; i < kReliableChannelCount; ++i) {
    for (uint32_t s = 0; s < NET_RETRANSMIT_SLOTS; ++s) {
      retransmit_slots[conn][i][s].buf_idx = kInvalidPoolIdx;
      retransmit_slots[conn][i][s].conn = conn;
      retransmit_slots[conn][i][s].seq = 0;
      retransmit_slots[conn][i][s].sent_at_ms = 0;
      retransmit_slots[conn][i][s].len = 0;
    }
  }

  for (uint32_t s = 0; s < NET_REORDER_BUF_SIZE; ++s) {
    reorder_buf[conn][s].buf_idx = kInvalidPoolIdx;
    reorder_buf[conn][s].seq = 0;
    reorder_buf[conn][s].len = 0;
    reorder_buf[conn][s].occupied = false;
  }

  return conn;
}

void ConnectionSlots::free_slot(ConnId conn) {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return;
  }
  if (alive[conn] == 0) {
    return;
  }

  clear_context(conn);

  alive[conn] = 0;
  endpoints[conn] = NetEndpoint{};
  connect_time_ms[conn] = 0;

  std::memset(send_seq[conn], 0, sizeof(send_seq[conn]));
  std::memset(recv_ack[conn], 0xFF, sizeof(recv_ack[conn]));
  std::memset(recv_ack_bits[conn], 0, sizeof(recv_ack_bits[conn]));
  std::memset(recv_rel_unord_bits[conn], 0, sizeof(recv_rel_unord_bits[conn]));
  reorder_next_exp[conn] = 0;
  std::memset(retransmit_head[conn], 0, sizeof(retransmit_head[conn]));
  std::memset(retransmit_tail[conn], 0, sizeof(retransmit_tail[conn]));

#if NET_ENCRYPTION
  std::memset(encryption_key[conn], 0, sizeof(encryption_key[conn]));
  encryption_conn_id[conn] = 0;
  encryption_ready[conn] = 0;
#endif

  for (uint32_t i = 0; i < kReliableChannelCount; ++i) {
    for (uint32_t s = 0; s < NET_RETRANSMIT_SLOTS; ++s) {
      retransmit_slots[conn][i][s].buf_idx = kInvalidPoolIdx;
      retransmit_slots[conn][i][s].conn = kInvalidConnId;
      retransmit_slots[conn][i][s].seq = 0;
      retransmit_slots[conn][i][s].sent_at_ms = 0;
      retransmit_slots[conn][i][s].len = 0;
    }
  }

  for (uint32_t s = 0; s < NET_REORDER_BUF_SIZE; ++s) {
    reorder_buf[conn][s].buf_idx = kInvalidPoolIdx;
    reorder_buf[conn][s].seq = 0;
    reorder_buf[conn][s].len = 0;
    reorder_buf[conn][s].occupied = false;
  }

  const int32_t next = free_slot_top + 1;
  assert(next < static_cast<int32_t>(NET_MAX_CONNECTIONS));
  if (next >= static_cast<int32_t>(NET_MAX_CONNECTIONS)) {
    return;
  }

  free_slot_stack[next] = conn;
  free_slot_top = next;
}

} // namespace attome::net
