#pragma once

#include "ConnectionSlots.h"
#include "PacketPool.h"
#include "ReliabilityLayer.h"

#include <cstdint>
#include <span>

namespace attome::net {

class CryptoLayer;

class MessageBatcher {
public:
  void init();

  // Releases any in-progress packet buffers for this connection (all channels).
  // Intended for disconnect paths to prevent PacketPool leaks.
  void reset_conn(PacketPool &pool, ConnId conn);

  // Returns false only on fatal failure (pool exhausted or payload invalid).
  bool write(PacketPool &pool, ConnectionSlots &slots, ConnId conn, uint8_t ch,
             uint16_t opcode, const OpcodeInfo &info, const uint8_t *payload,
             uint16_t plen, uint32_t now_ms, std::span<SendItem> out,
             uint32_t &out_count, const CryptoLayer *crypto = nullptr);

  void flush_all(PacketPool &pool, ConnectionSlots &slots, uint32_t now_ms,
                 std::span<SendItem> out, uint32_t &out_count,
                 const CryptoLayer *crypto = nullptr);

private:
  PoolIdx cur_buf_[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT];
  uint16_t cur_len_[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT];
  uint8_t msg_count_[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT];

  void flush_one_(PacketPool &pool, ConnectionSlots &slots, ConnId conn,
                  uint8_t ch, uint32_t now_ms, std::span<SendItem> out,
                  uint32_t &out_count, const CryptoLayer *crypto);
};

} // namespace attome::net
