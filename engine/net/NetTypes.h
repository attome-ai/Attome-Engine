#pragma once

#include "../NetConfig.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace attome::net {

using ConnId = uint16_t;
using PoolIdx = uint16_t;

inline constexpr ConnId kInvalidConnId = UINT16_MAX;
inline constexpr PoolIdx kInvalidPoolIdx = UINT16_MAX;

struct OpcodeInfo {
  const char *name{};
  int16_t payload_size{NET_PAYLOAD_DYNAMIC};
};

#pragma pack(push, 1)
struct PacketHeader {
  uint8_t channel{};
  uint16_t seq{};
  uint16_t ack{};
  uint32_t ack_bits{};
};
#pragma pack(pop)

static_assert(sizeof(PacketHeader) == NET_HEADER_SIZE,
              "PacketHeader layout must match NET_HEADER_SIZE.");

struct NetEndpoint {
  uint16_t port{};
  uint8_t is_v6{}; // 0 = v4, 1 = v6
  uint8_t _pad0{};
  uint32_t v6_scope_id{};
  union {
    uint32_t v4{};
    uint8_t v6[16];
  } addr{};
};

struct RetransmitSlot {
  uint32_t sent_at_ms{};
  PoolIdx buf_idx{kInvalidPoolIdx};
  uint16_t seq{};
  ConnId conn{kInvalidConnId};
  uint16_t len{};
};

static_assert(sizeof(RetransmitSlot) == 12, "RetransmitSlot must stay compact.");

struct ReorderEntry {
  PoolIdx buf_idx{kInvalidPoolIdx};
  uint16_t seq{};
  uint16_t len{};
  bool occupied{};
};

static_assert(sizeof(ReorderEntry) == 8, "ReorderEntry must stay compact.");

struct InboundMsg {
  ConnId conn{kInvalidConnId};
  uint8_t channel{};
  PoolIdx buf_idx{kInvalidPoolIdx};
  uint16_t payload_offset{};
  uint16_t payload_len{};
};

struct OutboundMsg {
  ConnId conn{kInvalidConnId};
  uint8_t channel{};
  uint16_t opcode{};
  uint16_t len{};
  alignas(16) uint8_t staging[NET_MTU];
};

static_assert(alignof(OutboundMsg) >= 16,
              "OutboundMsg must be at least 16-byte aligned.");
static_assert((offsetof(OutboundMsg, staging) % 16) == 0,
              "OutboundMsg::staging must be 16-byte aligned.");

struct RawRecv {
  PoolIdx buf_idx{kInvalidPoolIdx};
  uint16_t len{};
  NetEndpoint endpoint{};
};

struct SendItem {
  const uint8_t *data{};
  uint16_t len{};
  const NetEndpoint *endpoint{};
};

} // namespace attome::net
