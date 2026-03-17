#pragma once

#include "ConnectionSlots.h"
#include "PacketPool.h"
#include "SpscQueue.h"

#include <cstdint>

namespace attome::net::reorder {

inline void clear_conn(ConnectionSlots &slots, PacketPool &pool, ConnId conn) {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return;
  }

  for (uint32_t i = 0; i < NET_REORDER_BUF_SIZE; ++i) {
    ReorderEntry &e = slots.reorder_buf[conn][i];
    if (e.occupied && e.buf_idx != kInvalidPoolIdx) {
      pool.release(e.buf_idx);
    }
    e.occupied = false;
    e.buf_idx = kInvalidPoolIdx;
    e.seq = 0;
    e.len = 0;
  }
  slots.reorder_next_exp[conn] = 0;
}

template <uint32_t QN>
inline bool push_inbound(SpscQueue<InboundMsg, QN> &out_queue, ConnId conn,
                         PoolIdx buf_idx, uint16_t len) {
  if (len < NET_HEADER_SIZE) {
    return false;
  }
  const uint16_t payload_offset = NET_HEADER_SIZE;
  const uint16_t payload_len = static_cast<uint16_t>(len - payload_offset);
  return out_queue.push(InboundMsg{
      .conn = conn,
      .channel = NET_CHANNEL_RELIABLE_ORD,
      .buf_idx = buf_idx,
      .payload_offset = payload_offset,
      .payload_len = payload_len,
  });
}

template <uint32_t QN>
inline void try_drain(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                      SpscQueue<InboundMsg, QN> &out_queue) {
  if (!slots.is_alive(conn)) {
    return;
  }

  const uint16_t mask = static_cast<uint16_t>(NET_REORDER_BUF_SIZE - 1);
  while (true) {
    const uint16_t expected = slots.reorder_next_exp[conn];
    ReorderEntry &ent = slots.reorder_buf[conn][expected & mask];
    if (!ent.occupied) {
      break;
    }
    if (ent.seq != expected) {
      break;
    }

    if (!push_inbound(out_queue, conn, ent.buf_idx, ent.len)) {
      break;
    }

    ent.occupied = false;
    ent.buf_idx = kInvalidPoolIdx;
    ent.len = 0;

    slots.reorder_next_exp[conn] = static_cast<uint16_t>(expected + 1);
  }

  (void)pool;
}

template <uint32_t QN>
inline bool insert(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                   uint16_t seq, PoolIdx buf_idx, uint16_t len,
                   SpscQueue<InboundMsg, QN> &out_queue) {
  if (!slots.is_alive(conn)) {
    pool.release(buf_idx);
    return false;
  }
  if (buf_idx == kInvalidPoolIdx) {
    return false;
  }
  if (len < NET_HEADER_SIZE) {
    pool.release(buf_idx);
    return false;
  }

  const uint16_t expected = slots.reorder_next_exp[conn];
  const uint16_t dist = static_cast<uint16_t>(seq - expected); // wrapping
  if (dist >= NET_REORDER_BUF_SIZE) {
    pool.release(buf_idx);
    return false;
  }

  const uint16_t mask = static_cast<uint16_t>(NET_REORDER_BUF_SIZE - 1);
  const uint16_t ring_idx = static_cast<uint16_t>(seq & mask);
  ReorderEntry &slot = slots.reorder_buf[conn][ring_idx];

  if (dist == 0) {
    if (push_inbound(out_queue, conn, buf_idx, len)) {
      slots.reorder_next_exp[conn] = static_cast<uint16_t>(expected + 1);
      try_drain(slots, pool, conn, out_queue);
      return true;
    }

    if (slot.occupied) {
      pool.release(buf_idx);
      return false;
    }
    slot.buf_idx = buf_idx;
    slot.seq = seq;
    slot.len = len;
    slot.occupied = true;
    return true;
  }

  if (slot.occupied) {
    pool.release(buf_idx);
    return false;
  }

  slot.buf_idx = buf_idx;
  slot.seq = seq;
  slot.len = len;
  slot.occupied = true;
  return true;
}

} // namespace attome::net::reorder
