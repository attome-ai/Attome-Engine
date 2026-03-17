#include "MessageBatcher.h"
#include "CryptoLayer.h"

#include <cassert>
#include <cstring>

namespace attome::net {

static inline void write_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
}

void MessageBatcher::init() {
  for (uint16_t conn = 0; conn < NET_MAX_CONNECTIONS; ++conn) {
    for (uint8_t ch = 0; ch < NET_CHANNEL_COUNT; ++ch) {
      cur_buf_[conn][ch] = kInvalidPoolIdx;
      cur_len_[conn][ch] = 0;
      msg_count_[conn][ch] = 0;
    }
  }
}

void MessageBatcher::reset_conn(PacketPool &pool, ConnId conn) {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS)) {
    return;
  }

  for (uint8_t ch = 0; ch < NET_CHANNEL_COUNT; ++ch) {
    PoolIdx &buf_idx = cur_buf_[conn][ch];
    if (buf_idx != kInvalidPoolIdx) {
      pool.release(buf_idx);
    }
    buf_idx = kInvalidPoolIdx;
    cur_len_[conn][ch] = 0;
    msg_count_[conn][ch] = 0;
  }
}

void MessageBatcher::flush_one_(PacketPool &pool, ConnectionSlots &slots,
                                ConnId conn, uint8_t ch, uint32_t now_ms,
                                std::span<SendItem> out,
                                uint32_t &out_count,
                                const CryptoLayer *crypto) {
  PoolIdx &buf_idx = cur_buf_[conn][ch];
  uint16_t &len = cur_len_[conn][ch];
  uint8_t &msg_count = msg_count_[conn][ch];

  if (buf_idx == kInvalidPoolIdx) {
    return;
  }
  if (msg_count == 0 || len <= NET_HEADER_SIZE) {
    pool.release(buf_idx);
    buf_idx = kInvalidPoolIdx;
    len = 0;
    msg_count = 0;
    return;
  }
  if (slots.alive[conn] == 0) {
    pool.release(buf_idx);
    buf_idx = kInvalidPoolIdx;
    len = 0;
    msg_count = 0;
    return;
  }

  auto *hdr = reinterpret_cast<PacketHeader *>(pool.ptr(buf_idx));
  hdr->channel = ch;
  hdr->seq = reliability::assign_seq(slots, conn, ch);
  hdr->ack = slots.recv_ack[conn][ch];
  hdr->ack_bits = slots.recv_ack_bits[conn][ch];

#if NET_ENCRYPTION
  if (crypto != nullptr && slots.encryption_ready[conn] != 0) {
    if (!crypto->encrypt_in_place(pool.ptr(buf_idx), len, conn, slots)) {
      pool.release(buf_idx);
      buf_idx = kInvalidPoolIdx;
      len = 0;
      msg_count = 0;
      return;
    }
  }
#endif

  if (ch == NET_CHANNEL_RELIABLE_UNORD || ch == NET_CHANNEL_RELIABLE_ORD) {
    if (!reliability::record_sent(slots, pool, conn, ch, hdr->seq, buf_idx, len,
                                  now_ms)) {
      pool.release(buf_idx);
      buf_idx = kInvalidPoolIdx;
      len = 0;
      msg_count = 0;
      return;
    }
  }

  if (out_count < out.size()) {
    out[out_count++] =
        SendItem{pool.ptr(buf_idx), len, &slots.endpoints[conn]};
  } else {
    if (ch == NET_CHANNEL_UNRELIABLE) {
      pool.release(buf_idx);
    }
  }

  buf_idx = kInvalidPoolIdx;
  len = 0;
  msg_count = 0;
}

bool MessageBatcher::write(PacketPool &pool, ConnectionSlots &slots, ConnId conn,
                           uint8_t ch, uint16_t opcode, const OpcodeInfo &info,
                           const uint8_t *payload, uint16_t plen,
                           uint32_t now_ms, std::span<SendItem> out,
                           uint32_t &out_count, const CryptoLayer *crypto) {
  assert(conn < NET_MAX_CONNECTIONS);
  assert(ch < NET_CHANNEL_COUNT);

  const bool is_dynamic = (info.payload_size == NET_PAYLOAD_DYNAMIC);
  const uint16_t fixed_size =
      is_dynamic ? 0u : static_cast<uint16_t>(info.payload_size);
  if (!is_dynamic && (plen != fixed_size)) {
    return false;
  }

  const uint16_t wire_size = static_cast<uint16_t>(
      NET_OPCODE_SIZE + (is_dynamic ? (NET_DYNLEN_SIZE + plen) : plen));
  const uint16_t max_packet_len =
      static_cast<uint16_t>(NET_HEADER_SIZE + NET_MAX_PAYLOAD_ENC);
  if (wire_size > NET_MAX_PAYLOAD_ENC || max_packet_len > NET_MTU) {
    return false;
  }

  PoolIdx &buf_idx = cur_buf_[conn][ch];
  uint16_t &len = cur_len_[conn][ch];
  uint8_t &msg_count = msg_count_[conn][ch];

  if (buf_idx == kInvalidPoolIdx) {
    buf_idx = pool.acquire();
    if (buf_idx == kInvalidPoolIdx) {
      return false;
    }
    std::memset(pool.ptr(buf_idx), 0, NET_HEADER_SIZE);
    len = NET_HEADER_SIZE;
    msg_count = 0;
  }

  const bool would_overflow_mtu =
      static_cast<uint32_t>(len) + wire_size > max_packet_len;
  const bool would_overflow_msgs = (msg_count >= NET_MAX_MESSAGES_PER_PACKET);
  if (would_overflow_mtu || would_overflow_msgs) {
    flush_one_(pool, slots, conn, ch, now_ms, out, out_count, crypto);

    buf_idx = pool.acquire();
    if (buf_idx == kInvalidPoolIdx) {
      return false;
    }
    std::memset(pool.ptr(buf_idx), 0, NET_HEADER_SIZE);
    len = NET_HEADER_SIZE;
    msg_count = 0;
  }

  uint8_t *dst = pool.ptr(buf_idx) + len;
  write_u16_le(dst, opcode);
  dst += NET_OPCODE_SIZE;

  if (is_dynamic) {
    write_u16_le(dst, plen);
    dst += NET_DYNLEN_SIZE;
  }

  if (plen > 0) {
    std::memcpy(dst, payload, plen);
    dst += plen;
  }

  len = static_cast<uint16_t>(len + wire_size);
  msg_count = static_cast<uint8_t>(msg_count + 1);
  return true;
}

void MessageBatcher::flush_all(PacketPool &pool, ConnectionSlots &slots,
                               uint32_t now_ms, std::span<SendItem> out,
                               uint32_t &out_count, const CryptoLayer *crypto) {
  for (uint16_t conn = 0; conn < NET_MAX_CONNECTIONS; ++conn) {
    for (uint8_t ch = 0; ch < NET_CHANNEL_COUNT; ++ch) {
      flush_one_(pool, slots, static_cast<ConnId>(conn), ch, now_ms, out,
                 out_count, crypto);
    }
  }
}

} // namespace attome::net
