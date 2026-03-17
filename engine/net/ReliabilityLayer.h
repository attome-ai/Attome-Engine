#pragma once

#include "ConnectionSlots.h"
#include "PacketPool.h"

#include <cstdint>
#include <span>

namespace attome::net::reliability {

inline bool seq_acked(uint16_t seq, uint16_t ack, uint32_t ack_bits) {
  const uint16_t diff = static_cast<uint16_t>(ack - seq); // wrapping
  if (diff == 0) {
    return true;
  }
  if (diff > NET_ACK_BITS_COUNT) {
    return false;
  }
  const uint32_t mask = 1u << (diff - 1);
  return (ack_bits & mask) != 0;
}

inline uint16_t assign_seq(ConnectionSlots &slots, ConnId conn, uint8_t ch) {
  if ((conn == kInvalidConnId) || (conn >= NET_MAX_CONNECTIONS) ||
      (ch >= NET_CHANNEL_COUNT)) {
    return 0;
  }
  const uint16_t cur = slots.send_seq[conn][ch];
  slots.send_seq[conn][ch] = static_cast<uint16_t>(cur + 1);
  return cur;
}

inline void shift_left_words(std::span<uint32_t> words, uint32_t shift_bits) {
  if (words.empty()) {
    return;
  }

  const uint32_t word_shift = shift_bits / 32u;
  const uint32_t bit_shift = shift_bits % 32u;
  const uint32_t n = static_cast<uint32_t>(words.size());

  if (shift_bits == 0) {
    return;
  }
  if (word_shift >= n) {
    for (uint32_t i = 0; i < n; ++i) {
      words[i] = 0;
    }
    return;
  }

  if (bit_shift == 0) {
    for (int32_t i = static_cast<int32_t>(n) - 1; i >= 0; --i) {
      const int32_t src = i - static_cast<int32_t>(word_shift);
      words[static_cast<uint32_t>(i)] =
          (src >= 0) ? words[static_cast<uint32_t>(src)] : 0u;
    }
    return;
  }

  const uint32_t rshift = 32u - bit_shift;
  for (int32_t i = static_cast<int32_t>(n) - 1; i >= 0; --i) {
    const int32_t src0 = i - static_cast<int32_t>(word_shift);
    const int32_t src1 = src0 - 1;

    const uint32_t hi =
        (src0 >= 0) ? (words[static_cast<uint32_t>(src0)] << bit_shift) : 0u;
    const uint32_t lo =
        (src1 >= 0) ? (words[static_cast<uint32_t>(src1)] >> rshift) : 0u;
    words[static_cast<uint32_t>(i)] = hi | lo;
  }
}

// Extended receive window update for reliable-unordered duplicate suppression.
// - `ack`/`ack_bits` remain the on-wire 32-bit ACK representation.
// - `recv_window` tracks a larger history (e.g. NET_RETRANSMIT_SLOTS bits) so
//   late retransmits are still dropped as duplicates.
// Returns false if the sequence number was already seen (duplicate) or is too
// old for the configured receive window.
inline bool update_recv_window_ext(uint16_t &ack, uint32_t &ack_bits,
                                   std::span<uint32_t> recv_window,
                                   uint16_t seq) {
  if (recv_window.empty()) {
    return false;
  }

  const uint32_t window_bits = static_cast<uint32_t>(recv_window.size() * 32u);

  // 0xFFFF is used as "none received yet" initial state.
  if (ack == 0xFFFFu) {
    ack = seq;
    for (uint32_t &w : recv_window) {
      w = 0;
    }
    ack_bits = 0;
    return true;
  }

  if (seq == ack) {
    return false;
  }

  const uint16_t forward = static_cast<uint16_t>(seq - ack);
  if (forward < 32768) {
    if (forward > window_bits) {
      for (uint32_t &w : recv_window) {
        w = 0;
      }
    } else {
      shift_left_words(recv_window, static_cast<uint32_t>(forward));
      const uint32_t bit = static_cast<uint32_t>(forward - 1);
      recv_window[bit / 32u] |= (1u << (bit % 32u));
    }

    ack = seq;
    ack_bits = recv_window[0];
    return true;
  }

  const uint16_t back = static_cast<uint16_t>(ack - seq);
  if (back == 0) {
    return false;
  }
  if (back > window_bits) {
    return false;
  }

  const uint32_t bit = static_cast<uint32_t>(back - 1);
  const uint32_t mask = 1u << (bit % 32u);
  const uint32_t word = bit / 32u;
  if ((recv_window[word] & mask) != 0u) {
    return false;
  }

  recv_window[word] |= mask;
  ack_bits = recv_window[0];
  return true;
}

bool record_sent(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                 uint8_t ch, uint16_t seq, PoolIdx buf_idx, uint16_t len,
                 uint32_t now_ms);

void process_ack(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                 uint8_t ch, uint16_t ack, uint32_t ack_bits);

void retransmit_pass(ConnectionSlots &slots, PacketPool &pool, ConnId conn,
                     uint8_t ch, uint32_t now_ms, std::span<SendItem> out,
                     uint32_t &out_count);

// Releases all in-flight retransmit buffers for a connection (CH1 + CH2).
// Intended for disconnect/cleanup paths to prevent PacketPool leaks.
void clear_all(ConnectionSlots &slots, PacketPool &pool, ConnId conn);

} // namespace attome::net::reliability
