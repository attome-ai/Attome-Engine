#include "ReliabilityLayer.h"

namespace attome::net::reliability {

bool record_sent(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                 uint8_t ch, uint16_t seq, PoolIdx buf_idx, uint16_t len,
                 uint32_t now_ms) {
  (void)pool;

  if (!slots.is_alive(conn)) {
    return false;
  }
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  if (ridx == UINT8_MAX) {
    return false;
  }
  if (buf_idx == kInvalidPoolIdx) {
    return false;
  }

  const uint8_t head = slots.retransmit_head[conn][ridx];
  const uint8_t tail = slots.retransmit_tail[conn][ridx];
  const uint8_t used = static_cast<uint8_t>(tail - head);
  if (used >= NET_RETRANSMIT_SLOTS) {
    return false;
  }

  const uint8_t mask = static_cast<uint8_t>(NET_RETRANSMIT_SLOTS - 1);
  const uint8_t slotIdx = static_cast<uint8_t>(tail & mask);

  RetransmitSlot &slot = slots.retransmit_slots[conn][ridx][slotIdx];
  slot.buf_idx = buf_idx;
  slot.seq = seq;
  slot.conn = conn;
  slot.sent_at_ms = now_ms;
  slot.len = len;

  slots.retransmit_tail[conn][ridx] = static_cast<uint8_t>(tail + 1);
  return true;
}

void process_ack(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                 uint8_t ch, uint16_t ack, uint32_t ack_bits) {
  if (!slots.is_alive(conn)) {
    return;
  }
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  if (ridx == UINT8_MAX) {
    return;
  }

  const uint8_t mask = static_cast<uint8_t>(NET_RETRANSMIT_SLOTS - 1);

  uint8_t head = slots.retransmit_head[conn][ridx];
  const uint8_t tail = slots.retransmit_tail[conn][ridx];

  for (uint8_t cur = head; cur != tail; cur = static_cast<uint8_t>(cur + 1)) {
    RetransmitSlot &slot = slots.retransmit_slots[conn][ridx][cur & mask];
    if (slot.buf_idx == kInvalidPoolIdx) {
      continue;
    }

    if (!seq_acked(slot.seq, ack, ack_bits)) {
      continue;
    }

    pool.release(slot.buf_idx);
    slot.buf_idx = kInvalidPoolIdx;
    slot.len = 0;
  }

  while (head != tail) {
    RetransmitSlot &slot = slots.retransmit_slots[conn][ridx][head & mask];
    if (slot.buf_idx != kInvalidPoolIdx) {
      break;
    }
    head = static_cast<uint8_t>(head + 1);
  }

  slots.retransmit_head[conn][ridx] = head;
}

void retransmit_pass(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                     uint8_t ch, uint32_t now_ms, std::span<SendItem> out,
                     uint32_t &out_count) {
  if (!slots.is_alive(conn)) {
    return;
  }
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  if (ridx == UINT8_MAX) {
    return;
  }

  const uint8_t mask = static_cast<uint8_t>(NET_RETRANSMIT_SLOTS - 1);

  const uint8_t head = slots.retransmit_head[conn][ridx];
  const uint8_t tail = slots.retransmit_tail[conn][ridx];

  for (uint8_t cur = head; cur != tail; cur = static_cast<uint8_t>(cur + 1)) {
    RetransmitSlot &slot = slots.retransmit_slots[conn][ridx][cur & mask];
    if (slot.buf_idx == kInvalidPoolIdx) {
      continue;
    }

    const uint32_t age = now_ms - slot.sent_at_ms;
    if (age < NET_RETRANSMIT_TIMEOUT_MS) {
      continue;
    }

    if (out_count >= out.size()) {
      return;
    }

    out[out_count++] = SendItem{
        .data = pool.ptr(slot.buf_idx),
        .len = slot.len,
        .endpoint = &slots.endpoints[conn],
    };

    slot.sent_at_ms = now_ms;
  }
}

void clear_all(ConnectionSlots &slots, PacketPool &pool, ConnId conn) {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return;
  }

  const uint8_t mask = static_cast<uint8_t>(NET_RETRANSMIT_SLOTS - 1);

  for (uint8_t ridx = 0; ridx < ConnectionSlots::kReliableChannelCount; ++ridx) {
    const uint8_t head = slots.retransmit_head[conn][ridx];
    const uint8_t tail = slots.retransmit_tail[conn][ridx];

    for (uint8_t cur = head; cur != tail; cur = static_cast<uint8_t>(cur + 1)) {
      RetransmitSlot &slot = slots.retransmit_slots[conn][ridx][cur & mask];
      if (slot.buf_idx == kInvalidPoolIdx) {
        continue;
      }
      pool.release(slot.buf_idx);
      slot.buf_idx = kInvalidPoolIdx;
      slot.len = 0;
    }

    slots.retransmit_head[conn][ridx] = tail;
  }
}

} // namespace attome::net::reliability
