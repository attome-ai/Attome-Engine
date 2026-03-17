#include "NetSelfTest.h"

#include "ActionBuilder.h"
#include "Client.h"
#include "ConnectionSlots.h"
#include "CryptoLayer.h"
#include "MessageBatcher.h"
#include "MessageReader.h"
#include "NetSocket.h"
#include "OpcodeRegistry.h"
#include "PacketPool.h"
#include "ReorderBuffer.h"
#include "ReliabilityLayer.h"
#include "Server.h"
#include "SimLayer.h"
#include "SpscQueue.h"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace {

static attome::net::PacketPool g_pool{};
static attome::net::ConnectionSlots g_slots{};
static attome::net::MessageBatcher g_batcher{};

struct TestFailure {
  const char *file{};
  int line{};
  const char *expr{};
};

static void print_failure(const TestFailure &f) {
  std::fprintf(stderr, "FAIL %s:%d: %s\n", f.file, f.line, f.expr);
}

#define CHECK(expr)                                                           \
  do {                                                                        \
    if (!(expr)) {                                                            \
      ::print_failure({__FILE__, __LINE__, #expr});                            \
      return false;                                                           \
    }                                                                         \
  } while (0)

static inline void write_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

static bool test_packet_pool_basic() {
  using namespace attome::net;

  g_pool.init();

  std::array<uint8_t, NET_PACKET_POOL_SIZE> seen{};
  seen.fill(0);

  // Init is a LIFO stack: first acquire should return NET_PACKET_POOL_SIZE-1.
  {
    const PoolIdx first = g_pool.acquire();
    CHECK(first != kInvalidPoolIdx);
    CHECK(first == static_cast<PoolIdx>(NET_PACKET_POOL_SIZE - 1));
    g_pool.release(first);
  }

  std::array<PoolIdx, NET_PACKET_POOL_SIZE> held{};
  for (uint32_t i = 0; i < NET_PACKET_POOL_SIZE; ++i) {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    CHECK(idx < NET_PACKET_POOL_SIZE);
    CHECK(seen[idx] == 0);
    seen[idx] = 1;
    held[i] = idx;

    // ptr() should return a stable pointer within the pool.
    CHECK(g_pool.ptr(idx) != nullptr);
    CHECK(g_pool.ptr(idx) == g_pool.ptr(idx));
  }

  // Beyond capacity should fail.
  CHECK(g_pool.acquire() == kInvalidPoolIdx);

  // Release all (deterministic order) and verify the freelist top resets.
  for (uint32_t i = 0; i < NET_PACKET_POOL_SIZE; ++i) {
    g_pool.release(static_cast<PoolIdx>(i));
  }

  const PoolIdx after_release = g_pool.acquire();
  CHECK(after_release != kInvalidPoolIdx);
  CHECK(after_release == static_cast<PoolIdx>(NET_PACKET_POOL_SIZE - 1));
  g_pool.release(after_release);

  return true;
}

static bool test_connection_slots_alloc_free_and_context() {
  using namespace attome::net;

  g_slots.init();

  NetEndpoint ep{};
  ep.port = 1234;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x01020304u;

  const ConnId c0 = g_slots.alloc_slot(ep, 42);
  CHECK(c0 != kInvalidConnId);
  CHECK(c0 == 0);

  struct Ctx {
    uint32_t *counter{};
    explicit Ctx(uint32_t *c) : counter(c) {
      if (counter != nullptr) {
        *counter += 1;
      }
    }
    ~Ctx() {
      if (counter != nullptr) {
        *counter += 1;
      }
    }
  };

  uint32_t dtor_count = 0;
  CHECK(g_slots.emplace_context<Ctx>(c0, &dtor_count) != nullptr);
  CHECK(dtor_count == 1);
  {
    auto *ctx = g_slots.get_context<Ctx>(c0);
    CHECK(ctx != nullptr);
    CHECK(ctx->counter == &dtor_count);
  }
  CHECK(g_slots.emplace_context<Ctx>(c0, &dtor_count) != nullptr);
  CHECK(dtor_count == 3); // dtor + ctor
  g_slots.clear_context(c0);
  CHECK(dtor_count == 4);
  CHECK(g_slots.get_context<Ctx>(c0) == nullptr);

  // free_slot() should auto-clear any active context (dtor called).
  CHECK(g_slots.emplace_context<Ctx>(c0, &dtor_count) != nullptr);
  CHECK(dtor_count == 5);
  g_slots.free_slot(c0);
  CHECK(dtor_count == 6);
  CHECK(g_slots.alive[c0] == 0);
  CHECK(g_slots.get_context<Ctx>(c0) == nullptr);

  // Fill all slots.
  std::array<uint8_t, NET_MAX_CONNECTIONS> seen{};
  seen.fill(0);

  for (uint32_t i = 0; i < NET_MAX_CONNECTIONS; ++i) {
    NetEndpoint e2 = ep;
    e2.port = static_cast<uint16_t>(1234 + i);
    const ConnId c = g_slots.alloc_slot(e2, i);
    CHECK(c != kInvalidConnId);
    CHECK(c < NET_MAX_CONNECTIONS);
    CHECK(seen[c] == 0);
    seen[c] = 1;
  }

  CHECK(g_slots.alloc_slot(ep, 0) == kInvalidConnId);
  for (uint32_t i = 0; i < NET_MAX_CONNECTIONS; ++i) {
    CHECK(seen[i] == 1);
    CHECK(g_slots.alive[i] != 0);
  }

  // Free in reverse order to restore deterministic 0..N-1 allocation order.
  for (int32_t i = static_cast<int32_t>(NET_MAX_CONNECTIONS) - 1; i >= 0; --i) {
    g_slots.free_slot(static_cast<ConnId>(i));
  }
  CHECK(g_slots.free_slot_top == static_cast<int32_t>(NET_MAX_CONNECTIONS - 1));
  for (uint32_t i = 0; i < NET_MAX_CONNECTIONS; ++i) {
    CHECK(g_slots.alive[i] == 0);
  }

  const ConnId again = g_slots.alloc_slot(ep, 0);
  CHECK(again != kInvalidConnId);
  CHECK(again == 0);
  g_slots.free_slot(again);

  return true;
}

static bool test_spsc_queue_order_and_capacity() {
  using namespace attome::net;

  constexpr uint32_t kN = 8192;
  SpscQueue<uint32_t, kN> q{};

  for (uint32_t i = 0; i < kN; ++i) {
    CHECK(q.push(i));
  }
  CHECK(!q.push(123456u)); // full

  for (uint32_t i = 0; i < kN; ++i) {
    uint32_t v = 0xFFFFFFFFu;
    CHECK(q.pop(v));
    CHECK(v == i);
  }

  uint32_t out = 0;
  CHECK(!q.pop(out)); // empty
  return true;
}

static bool test_reliability_layer_ack_and_retransmit() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9000;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u; // 127.0.0.1

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  constexpr uint8_t ch = NET_CHANNEL_RELIABLE_UNORD;
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  CHECK(ridx != UINT8_MAX);

  const PoolIdx b0 = g_pool.acquire();
  const PoolIdx b1 = g_pool.acquire();
  CHECK(b0 != kInvalidPoolIdx);
  CHECK(b1 != kInvalidPoolIdx);
  CHECK(b0 != b1);

  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 0, b0, 100, 0));
  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 1, b1, 100, 0));

  reliability::process_ack(g_slots, g_pool, conn, ch, 1, 1u);
  CHECK(g_slots.retransmit_head[conn][ridx] ==
        g_slots.retransmit_tail[conn][ridx]);

  const PoolIdx b2 = g_pool.acquire();
  CHECK(b2 != kInvalidPoolIdx);
  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 2, b2, 123, 0));

  SendItem out[8]{};
  uint32_t out_count = 0;
  reliability::retransmit_pass(g_slots, g_pool, conn, ch,
                               NET_RETRANSMIT_TIMEOUT_MS - 1, out, out_count);
  CHECK(out_count == 0);

  out_count = 0;
  reliability::retransmit_pass(g_slots, g_pool, conn, ch,
                               NET_RETRANSMIT_TIMEOUT_MS + 1, out, out_count);

  CHECK(out_count == 1);
  CHECK(out[0].data == g_pool.ptr(b2));
  CHECK(out[0].len == 123);
  CHECK(out[0].endpoint != nullptr);
  CHECK(out[0].endpoint->port == ep.port);

  reliability::process_ack(g_slots, g_pool, conn, ch, 2, 0u);
  CHECK(g_slots.retransmit_head[conn][ridx] ==
        g_slots.retransmit_tail[conn][ridx]);

  // Ring full should fail.
  std::array<PoolIdx, NET_RETRANSMIT_SLOTS> held{};
  for (uint32_t i = 0; i < NET_RETRANSMIT_SLOTS; ++i) {
    held[i] = g_pool.acquire();
    CHECK(held[i] != kInvalidPoolIdx);
    CHECK(reliability::record_sent(g_slots, g_pool, conn, ch,
                                   static_cast<uint16_t>(1000u + i), held[i],
                                   10, 0));
  }

  const PoolIdx extra = g_pool.acquire();
  CHECK(extra != kInvalidPoolIdx);
  CHECK(!reliability::record_sent(g_slots, g_pool, conn, ch, 2000, extra, 10, 0));
  g_pool.release(extra);

  reliability::clear_all(g_slots, g_pool, conn);
  CHECK(g_slots.retransmit_head[conn][ridx] ==
        g_slots.retransmit_tail[conn][ridx]);

  g_slots.free_slot(conn);
  return true;
}

static bool test_reliability_layer_ack_bits_non_contiguous() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9003;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u; // 127.0.0.1

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  constexpr uint8_t ch = NET_CHANNEL_RELIABLE_UNORD;
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  CHECK(ridx != UINT8_MAX);

  const PoolIdx b17 = g_pool.acquire();
  const PoolIdx b18 = g_pool.acquire();
  const PoolIdx b19 = g_pool.acquire();
  const PoolIdx b20 = g_pool.acquire();
  CHECK(b17 != kInvalidPoolIdx);
  CHECK(b18 != kInvalidPoolIdx);
  CHECK(b19 != kInvalidPoolIdx);
  CHECK(b20 != kInvalidPoolIdx);

  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 17, b17, 10, 0));
  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 18, b18, 11, 0));
  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 19, b19, 12, 0));
  CHECK(reliability::record_sent(g_slots, g_pool, conn, ch, 20, b20, 13, 0));

  // ACK 20 (always) + 19 (bit0) + 17 (bit2). Leave 18 unacked.
  const uint32_t ack_bits = (1u << 0) | (1u << 2);
  reliability::process_ack(g_slots, g_pool, conn, ch, 20, ack_bits);

  CHECK(g_slots.retransmit_slots[conn][ridx][0].buf_idx == kInvalidPoolIdx);
  CHECK(g_slots.retransmit_slots[conn][ridx][1].buf_idx == b18);
  CHECK(g_slots.retransmit_slots[conn][ridx][2].buf_idx == kInvalidPoolIdx);
  CHECK(g_slots.retransmit_slots[conn][ridx][3].buf_idx == kInvalidPoolIdx);
  CHECK(g_slots.retransmit_head[conn][ridx] == 1);

  reliability::clear_all(g_slots, g_pool, conn);
  g_slots.free_slot(conn);
  return true;
}

static bool test_reliability_layer_bulk_ack_advances_head() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9004;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  constexpr uint8_t ch = NET_CHANNEL_RELIABLE_UNORD;
  const uint8_t ridx = ConnectionSlots::reliable_index(ch);
  CHECK(ridx != UINT8_MAX);

  std::array<PoolIdx, NET_RETRANSMIT_SLOTS> held{};
  for (uint32_t i = 0; i < NET_RETRANSMIT_SLOTS; ++i) {
    held[i] = g_pool.acquire();
    CHECK(held[i] != kInvalidPoolIdx);
    CHECK(reliability::record_sent(
        g_slots, g_pool, conn, ch, static_cast<uint16_t>(i), held[i], 8, 0));
  }

  CHECK(g_slots.retransmit_head[conn][ridx] == 0);
  CHECK(g_slots.retransmit_tail[conn][ridx] ==
        static_cast<uint8_t>(NET_RETRANSMIT_SLOTS));

  // ACK first 64 sequences: 0..31 then 32..63.
  reliability::process_ack(g_slots, g_pool, conn, ch, 31, 0xFFFFFFFFu);
  reliability::process_ack(g_slots, g_pool, conn, ch, 63, 0xFFFFFFFFu);

  CHECK(g_slots.retransmit_head[conn][ridx] == 64);
  CHECK(g_slots.retransmit_tail[conn][ridx] ==
        static_cast<uint8_t>(NET_RETRANSMIT_SLOTS));

  for (uint32_t i = 0; i < 64; ++i) {
    CHECK(g_slots.retransmit_slots[conn][ridx][i].buf_idx == kInvalidPoolIdx);
  }
  for (uint32_t i = 64; i < NET_RETRANSMIT_SLOTS; ++i) {
    CHECK(g_slots.retransmit_slots[conn][ridx][i].buf_idx != kInvalidPoolIdx);
  }

  reliability::clear_all(g_slots, g_pool, conn);
  g_slots.free_slot(conn);
  return true;
}

static bool test_reorder_buffer_out_of_order_drain() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9001;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SpscQueue<InboundMsg, 8> q{};

  auto make_packet = [&](uint16_t seq) -> PoolIdx {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    uint8_t *buf = g_pool.ptr(idx);
    std::memset(buf, 0, NET_HEADER_SIZE + NET_OPCODE_SIZE);
    auto *hdr = reinterpret_cast<PacketHeader *>(buf);
    hdr->channel = NET_CHANNEL_RELIABLE_ORD;
    hdr->seq = seq;
    return idx;
  };

  const PoolIdx b1 = make_packet(1);
  CHECK(reorder::insert(g_slots, g_pool, conn, 1, b1,
                        static_cast<uint16_t>(NET_HEADER_SIZE + NET_OPCODE_SIZE),
                        q));

  const PoolIdx b0 = make_packet(0);
  CHECK(reorder::insert(g_slots, g_pool, conn, 0, b0,
                        static_cast<uint16_t>(NET_HEADER_SIZE + NET_OPCODE_SIZE),
                        q));

  InboundMsg m{};
  CHECK(q.pop(m));
  CHECK(m.channel == NET_CHANNEL_RELIABLE_ORD);
  CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
        0);
  g_pool.release(m.buf_idx);

  CHECK(q.pop(m));
  CHECK(m.channel == NET_CHANNEL_RELIABLE_ORD);
  CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
        1);
  g_pool.release(m.buf_idx);

  CHECK(!q.pop(m));

  g_slots.free_slot(conn);
  return true;
}

static bool test_reorder_buffer_in_order_delivery() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9005;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SpscQueue<InboundMsg, 128> q{};

  auto make_packet = [&](uint16_t seq) -> PoolIdx {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    uint8_t *buf = g_pool.ptr(idx);
    std::memset(buf, 0, NET_HEADER_SIZE + NET_OPCODE_SIZE);
    auto *hdr = reinterpret_cast<PacketHeader *>(buf);
    hdr->channel = NET_CHANNEL_RELIABLE_ORD;
    hdr->seq = seq;
    return idx;
  };

  for (uint16_t seq = 0; seq < 8; ++seq) {
    const PoolIdx idx = make_packet(seq);
    CHECK(reorder::insert(g_slots, g_pool, conn, seq, idx,
                          static_cast<uint16_t>(NET_HEADER_SIZE +
                                                NET_OPCODE_SIZE),
                          q));
  }

  for (uint16_t seq = 0; seq < 8; ++seq) {
    InboundMsg m{};
    CHECK(q.pop(m));
    CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
          seq);
    g_pool.release(m.buf_idx);
  }

  InboundMsg m{};
  CHECK(!q.pop(m));
  g_slots.free_slot(conn);
  return true;
}

static bool test_reorder_buffer_wraparound_delivery() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9006;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  g_slots.reorder_next_exp[conn] = 65534;

  SpscQueue<InboundMsg, 32> q{};

  auto make_packet = [&](uint16_t seq) -> PoolIdx {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    uint8_t *buf = g_pool.ptr(idx);
    std::memset(buf, 0, NET_HEADER_SIZE + NET_OPCODE_SIZE);
    auto *hdr = reinterpret_cast<PacketHeader *>(buf);
    hdr->channel = NET_CHANNEL_RELIABLE_ORD;
    hdr->seq = seq;
    return idx;
  };

  const uint16_t seqs[] = {65534, 65535, 0, 1};
  for (uint16_t seq : seqs) {
    const PoolIdx idx = make_packet(seq);
    CHECK(reorder::insert(g_slots, g_pool, conn, seq, idx,
                          static_cast<uint16_t>(NET_HEADER_SIZE +
                                                NET_OPCODE_SIZE),
                          q));
  }

  for (uint16_t seq : seqs) {
    InboundMsg m{};
    CHECK(q.pop(m));
    CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
          seq);
    g_pool.release(m.buf_idx);
  }

  InboundMsg m{};
  CHECK(!q.pop(m));
  g_slots.free_slot(conn);
  return true;
}

static bool test_reorder_buffer_window_overflow_and_drain() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9007;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SpscQueue<InboundMsg, 256> q{};

  auto make_packet = [&](uint16_t seq) -> PoolIdx {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    uint8_t *buf = g_pool.ptr(idx);
    std::memset(buf, 0, NET_HEADER_SIZE + NET_OPCODE_SIZE);
    auto *hdr = reinterpret_cast<PacketHeader *>(buf);
    hdr->channel = NET_CHANNEL_RELIABLE_ORD;
    hdr->seq = seq;
    return idx;
  };

  // Insert 1..(NET_REORDER_BUF_SIZE-1) (buffered), then overflow insert at dist
  // == NET_REORDER_BUF_SIZE (dropped), then deliver 0 and drain 1..N-1.
  for (uint16_t seq = 1; seq < NET_REORDER_BUF_SIZE; ++seq) {
    const PoolIdx idx = make_packet(seq);
    CHECK(reorder::insert(g_slots, g_pool, conn, seq, idx,
                          static_cast<uint16_t>(NET_HEADER_SIZE +
                                                NET_OPCODE_SIZE),
                          q));
  }

  {
    const uint16_t overflow_seq = NET_REORDER_BUF_SIZE;
    const PoolIdx idx = make_packet(overflow_seq);
    CHECK(!reorder::insert(g_slots, g_pool, conn, overflow_seq, idx,
                           static_cast<uint16_t>(NET_HEADER_SIZE +
                                                 NET_OPCODE_SIZE),
                           q));
  }

  InboundMsg tmp{};
  CHECK(!q.pop(tmp));

  {
    const PoolIdx idx0 = make_packet(0);
    CHECK(reorder::insert(g_slots, g_pool, conn, 0, idx0,
                          static_cast<uint16_t>(NET_HEADER_SIZE +
                                                NET_OPCODE_SIZE),
                          q));
  }

  for (uint16_t expected = 0; expected < NET_REORDER_BUF_SIZE; ++expected) {
    InboundMsg m{};
    CHECK(q.pop(m));
    CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
          expected);
    g_pool.release(m.buf_idx);
  }

  CHECK(!q.pop(tmp));
  g_slots.free_slot(conn);
  return true;
}

static bool test_reorder_buffer_duplicate_drop() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  NetEndpoint ep{};
  ep.port = 9008;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SpscQueue<InboundMsg, 128> q{};

  auto make_packet = [&](uint16_t seq) -> PoolIdx {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    uint8_t *buf = g_pool.ptr(idx);
    std::memset(buf, 0, NET_HEADER_SIZE + NET_OPCODE_SIZE);
    auto *hdr = reinterpret_cast<PacketHeader *>(buf);
    hdr->channel = NET_CHANNEL_RELIABLE_ORD;
    hdr->seq = seq;
    return idx;
  };

  const PoolIdx first = make_packet(5);
  CHECK(reorder::insert(g_slots, g_pool, conn, 5, first,
                        static_cast<uint16_t>(NET_HEADER_SIZE +
                                              NET_OPCODE_SIZE),
                        q));

  const PoolIdx dup = make_packet(5);
  CHECK(!reorder::insert(g_slots, g_pool, conn, 5, dup,
                         static_cast<uint16_t>(NET_HEADER_SIZE +
                                               NET_OPCODE_SIZE),
                         q));

  // Deliver 0..4; seq 5 drains from the buffered first packet.
  for (uint16_t seq = 0; seq < 5; ++seq) {
    const PoolIdx idx = make_packet(seq);
    CHECK(reorder::insert(g_slots, g_pool, conn, seq, idx,
                          static_cast<uint16_t>(NET_HEADER_SIZE +
                                                NET_OPCODE_SIZE),
                          q));
  }

  for (uint16_t expected = 0; expected <= 5; ++expected) {
    InboundMsg m{};
    CHECK(q.pop(m));
    CHECK(reinterpret_cast<const PacketHeader *>(g_pool.ptr(m.buf_idx))->seq ==
          expected);
    g_pool.release(m.buf_idx);
  }

  InboundMsg tmp{};
  CHECK(!q.pop(tmp));
  g_slots.free_slot(conn);
  return true;
}

static bool test_opcode_registry() {
  using namespace attome::net;

  enum class DummyOp : uint16_t { A = 0, B = 1, C = 2 };

  OpcodeRegistry<DummyOp, 3> reg{};
  reg.table[0] = OpcodeInfo{"A", 4};
  reg.table[1] = OpcodeInfo{"B", NET_PAYLOAD_DYNAMIC};
  reg.table[2] = OpcodeInfo{"C", 0};

  CHECK(std::strcmp(reg.get(DummyOp::A).name, "A") == 0);
  CHECK(reg.fixed_size(DummyOp::A) == 4);
  CHECK(!reg.is_dynamic(DummyOp::A));

  CHECK(reg.is_dynamic(DummyOp::B));
  CHECK(reg.fixed_size(DummyOp::C) == 0);

  return true;
}

static bool test_message_batcher_mtu_overflow_flushes() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();
  g_batcher.init();

  NetEndpoint ep{};
  ep.port = 9010;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SendItem out[8]{};
  uint32_t out_count = 0;

  const OpcodeInfo dyn{"Dyn", NET_PAYLOAD_DYNAMIC};
  static std::array<uint8_t, 1000> payload{};
  payload.fill(0xAB);

  CHECK(g_batcher.write(g_pool, g_slots, conn, NET_CHANNEL_UNRELIABLE, 0x0001,
                        dyn, payload.data(),
                        static_cast<uint16_t>(payload.size()), 100, out,
                        out_count));
  CHECK(out_count == 0);

  // Second write forces MTU flush of the first packet.
  CHECK(g_batcher.write(g_pool, g_slots, conn, NET_CHANNEL_UNRELIABLE, 0x0001,
                        dyn, payload.data(),
                        static_cast<uint16_t>(payload.size()), 100, out,
                        out_count));
  CHECK(out_count == 1);
  CHECK(out[0].len ==
        NET_HEADER_SIZE + NET_OPCODE_SIZE + NET_DYNLEN_SIZE + payload.size());

  g_batcher.flush_all(g_pool, g_slots, 100, out, out_count);
  CHECK(out_count == 2);

  for (uint32_t i = 0; i < out_count; ++i) {
    const PoolIdx idx = g_pool.idx_from_ptr(out[i].data);
    CHECK(idx != kInvalidPoolIdx);
    g_pool.release(idx);
  }

  g_slots.free_slot(conn);
  return true;
}

static bool test_message_batcher_unreliable_wire_layout() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();
  g_batcher.init();

  NetEndpoint ep{};
  ep.port = 9002;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId conn = g_slots.alloc_slot(ep, 0);
  CHECK(conn != kInvalidConnId);

  SendItem out[64]{};
  uint32_t out_count = 0;

  const OpcodeInfo fixed4{"Fixed4", 4};
  const uint8_t p0[4] = {1, 2, 3, 4};
  const uint8_t p1[4] = {5, 6, 7, 8};
  const uint8_t p2[4] = {9, 10, 11, 12};

  CHECK(g_batcher.write(g_pool, g_slots, conn, NET_CHANNEL_UNRELIABLE, 0x1111,
                        fixed4, p0, static_cast<uint16_t>(sizeof(p0)), 100, out,
                        out_count));
  CHECK(g_batcher.write(g_pool, g_slots, conn, NET_CHANNEL_UNRELIABLE, 0x2222,
                        fixed4, p1, static_cast<uint16_t>(sizeof(p1)), 100, out,
                        out_count));
  CHECK(g_batcher.write(g_pool, g_slots, conn, NET_CHANNEL_UNRELIABLE, 0x3333,
                        fixed4, p2, static_cast<uint16_t>(sizeof(p2)), 100, out,
                        out_count));

  g_batcher.flush_all(g_pool, g_slots, 100, out, out_count);
  CHECK(out_count == 1);

  const uint8_t *pkt = out[0].data;
  const uint16_t pkt_len = out[0].len;
  CHECK(pkt != nullptr);
  CHECK(pkt_len ==
        static_cast<uint16_t>(NET_HEADER_SIZE + 3 * (NET_OPCODE_SIZE + 4)));

  const auto *hdr = reinterpret_cast<const PacketHeader *>(pkt);
  CHECK(hdr->channel == NET_CHANNEL_UNRELIABLE);
  CHECK(hdr->seq == 0);

  const uint8_t *p = pkt + NET_HEADER_SIZE;
  CHECK(p[0] == 0x11 && p[1] == 0x11);
  CHECK(std::memcmp(p + NET_OPCODE_SIZE, p0, 4) == 0);

  p += NET_OPCODE_SIZE + 4;
  CHECK(p[0] == 0x22 && p[1] == 0x22);
  CHECK(std::memcmp(p + NET_OPCODE_SIZE, p1, 4) == 0);

  p += NET_OPCODE_SIZE + 4;
  CHECK(p[0] == 0x33 && p[1] == 0x33);
  CHECK(std::memcmp(p + NET_OPCODE_SIZE, p2, 4) == 0);

  const PoolIdx idx = g_pool.idx_from_ptr(pkt);
  CHECK(idx != kInvalidPoolIdx);
  g_pool.release(idx);

  // Dynamic: opcode + u16 length prefix + payload.
  g_pool.init();
  g_slots.init();
  g_batcher.init();
  const ConnId conn_dyn = g_slots.alloc_slot(ep, 0);
  CHECK(conn_dyn != kInvalidConnId);

  out_count = 0;
  const OpcodeInfo dyn{"Dyn", NET_PAYLOAD_DYNAMIC};
  const uint8_t dyn_payload[3] = {0xAA, 0xBB, 0xCC};
  CHECK(g_batcher.write(g_pool, g_slots, conn_dyn, NET_CHANNEL_UNRELIABLE, 0x00FF,
                        dyn, dyn_payload,
                        static_cast<uint16_t>(sizeof(dyn_payload)), 100, out,
                        out_count));
  g_batcher.flush_all(g_pool, g_slots, 100, out, out_count);
  CHECK(out_count == 1);
  CHECK(out[0].len == NET_HEADER_SIZE + NET_OPCODE_SIZE + NET_DYNLEN_SIZE + 3);

  const uint8_t *pd = out[0].data + NET_HEADER_SIZE;
  CHECK(pd[0] == 0xFF && pd[1] == 0x00);
  CHECK(pd[2] == 0x03 && pd[3] == 0x00);
  CHECK(std::memcmp(pd + NET_OPCODE_SIZE + NET_DYNLEN_SIZE, dyn_payload, 3) == 0);

  const PoolIdx idd = g_pool.idx_from_ptr(out[0].data);
  CHECK(idd != kInvalidPoolIdx);
  g_pool.release(idd);

  // Message count flush: NET_MAX_MESSAGES_PER_PACKET + 1 should yield 2 packets.
  g_pool.init();
  g_slots.init();
  g_batcher.init();
  const ConnId conn2 = g_slots.alloc_slot(ep, 0);
  CHECK(conn2 != kInvalidConnId);

  out_count = 0;
  const OpcodeInfo nop{"Nop", 0};
  for (uint32_t i = 0; i < (NET_MAX_MESSAGES_PER_PACKET + 1u); ++i) {
    CHECK(g_batcher.write(g_pool, g_slots, conn2, NET_CHANNEL_UNRELIABLE, 0x0001,
                          nop, nullptr, 0, 100, out, out_count));
  }
  g_batcher.flush_all(g_pool, g_slots, 100, out, out_count);
  CHECK(out_count == 2);

  for (uint32_t i = 0; i < out_count; ++i) {
    const PoolIdx bi = g_pool.idx_from_ptr(out[i].data);
    CHECK(bi != kInvalidPoolIdx);
    g_pool.release(bi);
  }

  g_slots.free_slot(conn2);
  return true;
}

static bool test_net_socket_loopback_send_recv() {
  using namespace attome::net;

  g_pool.init();

  asio::io_context ioc_recv{1};
  asio::io_context ioc_send{1};

  NetSocket sock_recv{ioc_recv};
  NetSocket sock_send{ioc_send};

  CHECK(sock_recv.bind(0));
  CHECK(sock_send.bind(0));

  asio::error_code ec;
  const auto recv_local = sock_recv.socket().local_endpoint(ec);
  CHECK(!ec);
  const uint16_t recv_port = recv_local.port();

  const auto send_local = sock_send.socket().local_endpoint(ec);
  CHECK(!ec);
  const uint16_t send_port = send_local.port();

  NetEndpoint dst{};
  dst.port = recv_port;
  dst.is_v6 = 0;
  dst.addr.v4 = 0x7F000001u; // 127.0.0.1

  SpscQueue<RawRecv, 8192> raw{};

  asio::co_spawn(ioc_recv, sock_recv.recv_loop(g_pool, raw), asio::detached);
  std::thread net_thread([&]() { ioc_recv.run(); });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  constexpr uint32_t kBatch = 100;
  constexpr uint32_t kSingles = 1000;
  constexpr uint32_t kTotal = kBatch + kSingles;

  std::array<uint8_t, kTotal> seen{};
  seen.fill(0);

  std::array<uint32_t, kBatch> batch_ids{};
  std::array<SendItem, kBatch> items{};
  for (uint32_t i = 0; i < kBatch; ++i) {
    batch_ids[i] = i;
    items[i] = SendItem{
        .data = reinterpret_cast<const uint8_t *>(&batch_ids[i]),
        .len = 4,
        .endpoint = &dst,
    };
  }
  sock_send.send_batch(items.data(), static_cast<int>(kBatch));

  for (uint32_t i = 0; i < kSingles; ++i) {
    const uint32_t id = kBatch + i;
    sock_send.send_one(reinterpret_cast<const uint8_t *>(&id), 4, dst);
  }

  uint32_t got = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (got < kTotal && std::chrono::steady_clock::now() < deadline) {
    RawRecv r{};
    if (!raw.pop(r)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    CHECK(r.len == 4);
    CHECK(r.endpoint.is_v6 == 0);
    CHECK(r.endpoint.addr.v4 == 0x7F000001u);
    CHECK(r.endpoint.port == send_port);

    uint32_t id = 0;
    std::memcpy(&id, g_pool.ptr(r.buf_idx), 4);
    CHECK(id < kTotal);
    CHECK(seen[id] == 0);
    seen[id] = 1;

    g_pool.release(r.buf_idx);
    got += 1;
  }

  sock_recv.close();
  sock_send.close();
  ioc_recv.stop();
  net_thread.join();

  CHECK(got == kTotal);
  for (uint32_t i = 0; i < kTotal; ++i) {
    CHECK(seen[i] == 1);
  }

  return true;
}

static bool run_server_client_multi_client_echo(uint32_t client_count,
                                                uint32_t msg_count,
                                                bool include_unreliable,
                                                std::chrono::milliseconds timeout,
                                                bool paced_send) {
  using namespace attome::net;

  constexpr uint16_t kC2S_CONNECT = 0x0001;
  constexpr uint16_t kC2S_DISCONNECT = 0x0002;
  constexpr uint16_t kC2S_CH0 = 0x0003;
  constexpr uint16_t kC2S_CH1 = 0x0004;
  constexpr uint16_t kC2S_CH2 = 0x0005;

  constexpr uint16_t kS2C_WELCOME = 0x0001;
  constexpr uint16_t kS2C_CH0 = 0x0002;
  constexpr uint16_t kS2C_CH1 = 0x0003;
  constexpr uint16_t kS2C_CH2 = 0x0004;

  std::array<OpcodeInfo, 6> c2s{};
  c2s.fill(OpcodeInfo{"", 0});
  c2s[kC2S_CONNECT] = OpcodeInfo{"CONNECT", NET_PAYLOAD_DYNAMIC};
  c2s[kC2S_DISCONNECT] = OpcodeInfo{"DISCONNECT", 0};
  c2s[kC2S_CH0] = OpcodeInfo{"CH0", 4};
  c2s[kC2S_CH1] = OpcodeInfo{"CH1", 4};
  c2s[kC2S_CH2] = OpcodeInfo{"CH2", 4};

  std::array<OpcodeInfo, 5> s2c{};
  s2c.fill(OpcodeInfo{"", 0});
  s2c[kS2C_WELCOME] = OpcodeInfo{"WELCOME", NET_PAYLOAD_DYNAMIC};
  s2c[kS2C_CH0] = OpcodeInfo{"CH0", 4};
  s2c[kS2C_CH1] = OpcodeInfo{"CH1", 4};
  s2c[kS2C_CH2] = OpcodeInfo{"CH2", 4};

  bool ok = true;

  std::vector<int32_t> conn_to_idx(NET_MAX_CONNECTIONS, -1);
  std::vector<ConnId> conn_list{};
  conn_list.reserve(client_count);

  std::vector<std::vector<uint8_t>> server_ch1_seen(
      client_count, std::vector<uint8_t>(msg_count, 0));
  std::vector<std::vector<uint8_t>> server_ch2_seen(
      client_count, std::vector<uint8_t>(msg_count, 0));
  std::vector<uint32_t> server_ch1_got(client_count, 0);
  std::vector<uint32_t> server_ch2_next(client_count, 0);

  auto server = std::make_unique<Server>(c2s, s2c);
  server->on(kC2S_CONNECT, [&](ConnId conn, MessageReader &r) {
    (void)r;
    if (conn >= conn_to_idx.size()) {
      ok = false;
      return;
    }
    if (conn_to_idx[conn] != -1) {
      return;
    }
    if (conn_list.size() >= client_count) {
      ok = false;
      return;
    }
    const int32_t idx = static_cast<int32_t>(conn_list.size());
    conn_to_idx[conn] = idx;
    conn_list.push_back(conn);
  });

  auto server_echo_u32 = [&](ConnId conn, uint8_t ch, uint16_t opcode,
                             uint32_t v) {
    uint8_t payload[4]{};
    write_u32_le(payload, v);
    server->send(conn, ch, opcode, std::span<const uint8_t>(payload, 4));
  };

  server->on(kC2S_CH0, [&](ConnId conn, MessageReader &r) {
    const uint32_t seq = r.read_u32();
    if (!r.ok()) {
      ok = false;
      return;
    }
    if (include_unreliable) {
      server_echo_u32(conn, NET_CHANNEL_UNRELIABLE, kS2C_CH0, seq);
    }
  });

  server->on(kC2S_CH1, [&](ConnId conn, MessageReader &r) {
    const uint32_t seq = r.read_u32();
    if (!r.ok()) {
      ok = false;
      return;
    }
    if (seq >= msg_count) {
      ok = false;
      return;
    }
    if (conn >= conn_to_idx.size()) {
      ok = false;
      return;
    }
    const int32_t idx = conn_to_idx[conn];
    if (idx < 0 || static_cast<uint32_t>(idx) >= client_count) {
      ok = false;
      return;
    }
    if (server_ch1_seen[idx][seq] != 0) {
      ok = false;
      return;
    }

    server_ch1_seen[idx][seq] = 1;
    server_ch1_got[idx] += 1;
    server_echo_u32(conn, NET_CHANNEL_RELIABLE_UNORD, kS2C_CH1, seq);
  });

  server->on(kC2S_CH2, [&](ConnId conn, MessageReader &r) {
    const uint32_t seq = r.read_u32();
    if (!r.ok()) {
      ok = false;
      return;
    }
    if (seq >= msg_count) {
      ok = false;
      return;
    }
    if (conn >= conn_to_idx.size()) {
      ok = false;
      return;
    }
    const int32_t idx = conn_to_idx[conn];
    if (idx < 0 || static_cast<uint32_t>(idx) >= client_count) {
      ok = false;
      return;
    }

    if (seq != server_ch2_next[idx]) {
      ok = false;
      return;
    }
    server_ch2_next[idx] += 1;

    if (server_ch2_seen[idx][seq] != 0) {
      ok = false;
      return;
    }
    server_ch2_seen[idx][seq] = 1;
    server_echo_u32(conn, NET_CHANNEL_RELIABLE_ORD, kS2C_CH2, seq);
  });

  CHECK(server->bind(0));
  const uint16_t port = server->bound_port();
  CHECK(port != 0);

#if NET_SIM
  constexpr uint32_t kConnectTimeoutMs = 8000;
#else
  constexpr uint32_t kConnectTimeoutMs = 3000;
#endif

  std::vector<std::unique_ptr<Client>> clients{};
  clients.reserve(client_count);

  for (uint32_t i = 0; i < client_count; ++i) {
    clients.emplace_back(std::make_unique<Client>(c2s, s2c));
    CHECK(clients.back()->connect("127.0.0.1", port, kConnectTimeoutMs));
  }

  std::vector<std::vector<uint8_t>> client_ch1_seen(
      client_count, std::vector<uint8_t>(msg_count, 0));
  std::vector<std::vector<uint8_t>> client_ch2_seen(
      client_count, std::vector<uint8_t>(msg_count, 0));
  std::vector<uint32_t> client_ch1_got(client_count, 0);
  std::vector<uint32_t> client_ch2_next(client_count, 0);
  std::vector<uint32_t> client_ch0_got(client_count, 0);

  for (uint32_t i = 0; i < client_count; ++i) {
    if (include_unreliable) {
      clients[i]->on(kS2C_CH0, [&, i](MessageReader &r) {
        const uint32_t seq = r.read_u32();
        if (!r.ok() || seq >= msg_count) {
          ok = false;
          return;
        }
        client_ch0_got[i] += 1;
      });
    }

    clients[i]->on(kS2C_CH1, [&, i](MessageReader &r) {
      const uint32_t seq = r.read_u32();
      if (!r.ok() || seq >= msg_count) {
        ok = false;
        return;
      }
      if (client_ch1_seen[i][seq] != 0) {
        ok = false;
        return;
      }
      client_ch1_seen[i][seq] = 1;
      client_ch1_got[i] += 1;
    });

    clients[i]->on(kS2C_CH2, [&, i](MessageReader &r) {
      const uint32_t seq = r.read_u32();
      if (!r.ok() || seq >= msg_count) {
        ok = false;
        return;
      }
      if (seq != client_ch2_next[i]) {
        ok = false;
        return;
      }
      client_ch2_next[i] += 1;
      if (client_ch2_seen[i][seq] != 0) {
        ok = false;
        return;
      }
      client_ch2_seen[i][seq] = 1;
    });
  }

  // Ensure the server has dispatched all CONNECT handlers (build conn_to_idx map).
  {
    const auto connect_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (conn_list.size() < client_count &&
           std::chrono::steady_clock::now() < connect_deadline) {
      server->tick(0);
      for (uint32_t i = 0; i < client_count; ++i) {
        clients[i]->tick(0);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(conn_list.size() == client_count);
    CHECK(ok);
  }

  std::vector<uint32_t> next_send(client_count, 0);
  if (!paced_send) {
    for (uint32_t seq = 0; seq < msg_count; ++seq) {
      uint8_t payload[4]{};
      write_u32_le(payload, seq);
      const auto span = std::span<const uint8_t>(payload, 4);
      for (uint32_t i = 0; i < client_count; ++i) {
        if (include_unreliable) {
          clients[i]->send(NET_CHANNEL_UNRELIABLE, kC2S_CH0, span);
        }
        clients[i]->send(NET_CHANNEL_RELIABLE_UNORD, kC2S_CH1, span);
        clients[i]->send(NET_CHANNEL_RELIABLE_ORD, kC2S_CH2, span);
      }
    }
    std::fill(next_send.begin(), next_send.end(), msg_count);
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (paced_send) {
#if NET_SIM
      // Under NET_SIM, send/recv packets are queued for at least
      // NET_SIM_LATENCY_MS. Keep per-client bursts small so we don't exceed the
      // simulation delay queue under many clients.
      constexpr uint32_t kBurstPerClient = 2;
#else
      constexpr uint32_t kBurstPerClient = 4;
#endif
      for (uint32_t i = 0; i < client_count; ++i) {
        for (uint32_t b = 0; b < kBurstPerClient; ++b) {
          const uint32_t seq = next_send[i];
          if (seq >= msg_count) {
            break;
          }
          uint8_t payload[4]{};
          write_u32_le(payload, seq);
          const auto span = std::span<const uint8_t>(payload, 4);
          if (include_unreliable) {
            clients[i]->send(NET_CHANNEL_UNRELIABLE, kC2S_CH0, span);
          }
          clients[i]->send(NET_CHANNEL_RELIABLE_UNORD, kC2S_CH1, span);
          clients[i]->send(NET_CHANNEL_RELIABLE_ORD, kC2S_CH2, span);
          next_send[i] = seq + 1;
        }
      }
    }

    server->tick(0);
    for (uint32_t i = 0; i < client_count; ++i) {
      clients[i]->tick(0);
    }

    bool done = ok;
    for (uint32_t i = 0; i < client_count && done; ++i) {
      if (client_ch1_got[i] != msg_count) {
        done = false;
        break;
      }
      if (client_ch2_next[i] != msg_count) {
        done = false;
        break;
      }
      if (server_ch1_got[i] != msg_count) {
        done = false;
        break;
      }
      if (server_ch2_next[i] != msg_count) {
        done = false;
        break;
      }
    }

    if (done) {
      for (uint32_t i = 0; i < client_count; ++i) {
        clients[i]->disconnect();
      }
      server->stop();
      return true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Timeout diagnostics (only printed on failure).
  if (ok) {
    uint32_t printed = 0;
    for (uint32_t i = 0; i < client_count; ++i) {
      const bool bad = (client_ch1_got[i] != msg_count) ||
                       (client_ch2_next[i] != msg_count) ||
                       (server_ch1_got[i] != msg_count) ||
                       (server_ch2_next[i] != msg_count);
      if (!bad) {
        continue;
      }

      std::fprintf(stderr,
                   "TIMEOUT client=%u next_send=%u c1=%u/%u c2=%u/%u s1=%u/%u "
                   "s2=%u/%u\n",
                   i, next_send[i], client_ch1_got[i], msg_count,
                   client_ch2_next[i], msg_count, server_ch1_got[i], msg_count,
                   server_ch2_next[i], msg_count);

      printed += 1;
      if (printed >= 8) {
        break;
      }
    }
  }

  for (uint32_t i = 0; i < client_count; ++i) {
    clients[i]->disconnect();
  }
  server->stop();

  CHECK(ok);
  for (uint32_t i = 0; i < client_count; ++i) {
    CHECK(client_ch1_got[i] == msg_count);
    CHECK(client_ch2_next[i] == msg_count);
    CHECK(server_ch1_got[i] == msg_count);
    CHECK(server_ch2_next[i] == msg_count);
  }

  return true;
}

static bool test_server_client_multi_client_channels_echo() {
#if NET_SIM
  constexpr auto kTimeout = std::chrono::milliseconds(20000);
#else
  constexpr auto kTimeout = std::chrono::milliseconds(8000);
#endif
  return run_server_client_multi_client_echo(8, 64, true, kTimeout, false);
}

static bool test_server_client_ack_only_allows_large_reliable_stream() {
  using namespace attome::net;

  constexpr uint16_t kC2S_CONNECT = 0x0001;
  constexpr uint16_t kC2S_DISCONNECT = 0x0002;

  constexpr uint16_t kS2C_WELCOME = 0x0001;
  constexpr uint16_t kS2C_BIG = 0x0002;

  constexpr uint16_t kBigPayloadSize =
      static_cast<uint16_t>(NET_MAX_PAYLOAD_ENC - NET_OPCODE_SIZE);

  std::array<OpcodeInfo, 3> c2s{};
  c2s.fill(OpcodeInfo{"", 0});
  c2s[kC2S_CONNECT] = OpcodeInfo{"CONNECT", NET_PAYLOAD_DYNAMIC};
  c2s[kC2S_DISCONNECT] = OpcodeInfo{"DISCONNECT", 0};

  std::array<OpcodeInfo, 3> s2c{};
  s2c.fill(OpcodeInfo{"", 0});
  s2c[kS2C_WELCOME] = OpcodeInfo{"WELCOME", NET_PAYLOAD_DYNAMIC};
  s2c[kS2C_BIG] = OpcodeInfo{"BIG", static_cast<int16_t>(kBigPayloadSize)};

  ConnId server_conn = kInvalidConnId;

  auto server = std::make_unique<Server>(c2s, s2c);
  server->on(kC2S_CONNECT, [&](ConnId conn, MessageReader &r) {
    (void)r;
    server_conn = conn;
  });

  CHECK(server->bind(0));
  const uint16_t port = server->bound_port();
  CHECK(port != 0);

#if NET_SIM
  constexpr uint32_t kConnectTimeoutMs = 8000;
  constexpr auto kTimeout = std::chrono::milliseconds(30000);
#else
  constexpr uint32_t kConnectTimeoutMs = 3000;
  constexpr auto kTimeout = std::chrono::milliseconds(8000);
#endif

  auto client = std::make_unique<Client>(c2s, s2c);
  CHECK(client->connect("127.0.0.1", port, kConnectTimeoutMs));

  const auto map_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (server_conn == kInvalidConnId &&
         std::chrono::steady_clock::now() < map_deadline) {
    server->tick(0);
    client->tick(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK(server_conn != kInvalidConnId);

  uint32_t kTotal = NET_RETRANSMIT_SLOTS + 32;
#if NET_SIM
  // Under NET_SIM, the ACK-only path itself can be dropped. Keep this test
  // bounded to the retransmit window to avoid requiring progress beyond
  // NET_RETRANSMIT_SLOTS in a lossy, best-effort ACK environment.
  kTotal = NET_RETRANSMIT_SLOTS;
#endif
  bool ok = true;
  bool reported = false;
  std::vector<uint8_t> seen(kTotal, 0);
  uint32_t got = 0;

  client->on(kS2C_BIG, [&](MessageReader &r) {
    const uint32_t seq = r.read_u32();
    if (!r.ok() || seq >= kTotal) {
      if (!reported) {
        std::fprintf(stderr, "BIG invalid: ok=%d seq=%u (kTotal=%u)\n",
                     r.ok() ? 1 : 0, seq, kTotal);
        reported = true;
      }
      ok = false;
      return;
    }
    if (seen[seq] != 0) {
      if (!reported) {
        std::fprintf(stderr, "BIG duplicate seq=%u\n", seq);
        reported = true;
      }
      ok = false;
      return;
    }
    seen[seq] = 1;
    got += 1;
  });

  uint32_t sent = 0;
  std::vector<uint8_t> payload(kBigPayloadSize, 0xAB);

  const uint32_t kInflightLimit = NET_ACK_BITS_COUNT;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while ((sent < kTotal || got < kTotal) &&
         std::chrono::steady_clock::now() < deadline) {
    while (sent < kTotal && (sent - got) < kInflightLimit) {
      write_u32_le(payload.data(), sent);
      server->send(server_conn, NET_CHANNEL_RELIABLE_UNORD, kS2C_BIG,
                   std::span<const uint8_t>(payload.data(), payload.size()));
      sent += 1;
    }

    server->tick(0);
    client->tick(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  client->disconnect();
  server->stop();

  if (!ok || got != kTotal) {
    std::fprintf(stderr, "ACK-only stream stats: sent=%u got=%u total=%u\n",
                 sent, got, kTotal);
    if (got != kTotal) {
      uint32_t missing_printed = 0;
      for (uint32_t i = 0; i < kTotal && missing_printed < 8; ++i) {
        if (seen[i] == 0) {
          std::fprintf(stderr, "  missing seq=%u\n", i);
          missing_printed += 1;
        }
      }
    }
  }

  CHECK(ok);
  CHECK(got == kTotal);
  for (uint32_t i = 0; i < kTotal; ++i) {
    CHECK(seen[i] == 1);
  }

  return true;
}

#if NET_ENCRYPTION

static bool test_crypto_layer_encrypt_decrypt_and_tamper() {
  using namespace attome::net;

  g_pool.init();
  g_slots.init();

  CryptoLayer crypto{};
  CHECK(crypto.init());

  NetEndpoint ep{};
  ep.port = 9011;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  const ConnId c0 = g_slots.alloc_slot(ep, 0);
  const ConnId c1 = g_slots.alloc_slot(ep, 0);
  CHECK(c0 != kInvalidConnId);
  CHECK(c1 != kInvalidConnId);
  CHECK(c0 != c1);

  g_slots.encryption_conn_id[c0] = 1234;
  g_slots.encryption_conn_id[c1] = 5678;
  crypto.set_psk(g_slots, c0);
  crypto.set_psk(g_slots, c1);

  const PoolIdx idx = g_pool.acquire();
  CHECK(idx != kInvalidPoolIdx);

  uint8_t *buf = g_pool.ptr(idx);
  std::memset(buf, 0, NET_MTU);

  auto *hdr = reinterpret_cast<PacketHeader *>(buf);
  hdr->channel = NET_CHANNEL_UNRELIABLE;
  hdr->seq = 42;
  hdr->ack = 0;
  hdr->ack_bits = 0;

  static constexpr uint8_t kPlain[8] = {0xDE, 0xAD, 0xBE, 0xEF,
                                        0x11, 0x22, 0x33, 0x44};
  std::memcpy(buf + NET_HEADER_SIZE, kPlain, sizeof(kPlain));
  const uint16_t plain_len =
      static_cast<uint16_t>(NET_HEADER_SIZE + sizeof(kPlain));

  std::array<uint8_t, NET_HEADER_SIZE + sizeof(kPlain)> before{};
  std::memcpy(before.data(), buf, plain_len);

  // Round-trip.
  {
    uint16_t len = plain_len;
    CHECK(crypto.encrypt_in_place(buf, len, c0, g_slots));
    CHECK(len == plain_len + NET_ENCRYPTION_TAG_SIZE);
    CHECK(crypto.decrypt_in_place(buf, len, c0, g_slots));
    CHECK(len == plain_len);
    CHECK(std::memcmp(buf, before.data(), plain_len) == 0);
  }

  // Tamper 1 byte of ciphertext -> decrypt fails.
  {
    std::memcpy(buf, before.data(), plain_len);
    uint16_t len = plain_len;
    CHECK(crypto.encrypt_in_place(buf, len, c0, g_slots));
    buf[NET_HEADER_SIZE] ^= 0x01;
    uint16_t tam_len = len;
    CHECK(!crypto.decrypt_in_place(buf, tam_len, c0, g_slots));
  }

  // Wrong nonce (different conn_id) -> decrypt fails.
  {
    std::memcpy(buf, before.data(), plain_len);
    uint16_t len = plain_len;
    CHECK(crypto.encrypt_in_place(buf, len, c0, g_slots));
    uint16_t wrong_len = len;
    CHECK(!crypto.decrypt_in_place(buf, wrong_len, c1, g_slots));
  }

  g_pool.release(idx);
  g_slots.free_slot(c1);
  g_slots.free_slot(c0);
  return true;
}

#endif

#if NET_SIM

static bool test_sim_layer_latency_drain() {
  using namespace attome::net;

  g_pool.init();

  SimLayer sim{};
  sim.init(1234);
  const auto p = sim.params();

  if (p.send_loss != 0.0f || p.recv_loss != 0.0f || p.jitter_ms != 0) {
    std::fprintf(stderr,
                 "SKIP SimLayer latency drain (requires 0 loss/jitter)\n");
    return true;
  }

  NetEndpoint ep{};
  ep.port = 9012;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  constexpr uint32_t now0 = 1000;
  {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    g_pool.ptr(idx)[0] = NET_CHANNEL_UNRELIABLE;

    CHECK(sim.enqueue_send(g_pool, idx, NET_HEADER_SIZE + 4, ep, now0));

    std::array<SimSend, 4> send_out{};
    std::array<RawRecv, 4> recv_out{};
    uint32_t send_count = 0;
    uint32_t recv_count = 0;

    sim.drain_ready(g_pool, now0 + (static_cast<uint32_t>(p.latency_ms) - 1),
                    send_out, send_count,
                    recv_out, recv_count);
    CHECK(send_count == 0);
    CHECK(recv_count == 0);

    sim.drain_ready(g_pool, now0 + static_cast<uint32_t>(p.latency_ms), send_out,
                    send_count,
                    recv_out, recv_count);
    CHECK(send_count == 1);
    CHECK(recv_count == 0);
    CHECK(send_out[0].buf_idx == idx);

    g_pool.sim_send_ref_dec(send_out[0].buf_idx);
    g_pool.release(send_out[0].buf_idx);
  }

  constexpr uint32_t now1 = 2000;
  {
    const PoolIdx idx = g_pool.acquire();
    CHECK(idx != kInvalidPoolIdx);
    g_pool.ptr(idx)[0] = NET_CHANNEL_UNRELIABLE;

    CHECK(sim.enqueue_recv(g_pool, idx, NET_HEADER_SIZE + 1, ep, now1));

    std::array<SimSend, 4> send_out{};
    std::array<RawRecv, 4> recv_out{};
    uint32_t send_count = 0;
    uint32_t recv_count = 0;

    sim.drain_ready(g_pool, now1 + (static_cast<uint32_t>(p.latency_ms) - 1),
                    send_out, send_count,
                    recv_out, recv_count);
    CHECK(send_count == 0);
    CHECK(recv_count == 0);

    sim.drain_ready(g_pool, now1 + static_cast<uint32_t>(p.latency_ms), send_out,
                    send_count,
                    recv_out, recv_count);
    CHECK(send_count == 0);
    CHECK(recv_count == 1);
    CHECK(recv_out[0].buf_idx == idx);

    g_pool.release(recv_out[0].buf_idx);
  }

  return true;
}

static bool test_sim_layer_overflow_drops_oldest_ack_only() {
  using namespace attome::net;

  g_pool.init();

  SimLayer sim{};
  sim.init(1);
  const auto p = sim.params();

  if (p.send_loss != 0.0f || p.jitter_ms != 0) {
    std::fprintf(stderr,
                 "SKIP SimLayer overflow (requires 0 send loss/jitter)\n");
    return true;
  }

  NetEndpoint ep{};
  ep.port = 9013;
  ep.is_v6 = 0;
  ep.addr.v4 = 0x7F000001u;

  constexpr uint32_t now_ms = 0;

  const PoolIdx ack_idx = g_pool.acquire();
  CHECK(ack_idx != kInvalidPoolIdx);
  g_pool.ptr(ack_idx)[0] = NET_CHANNEL_RELIABLE_UNORD;
  uint16_t ack_len = NET_HEADER_SIZE;
#if NET_ENCRYPTION
  ack_len = static_cast<uint16_t>(NET_HEADER_SIZE + NET_ENCRYPTION_TAG_SIZE);
#endif
  CHECK(sim.enqueue_send(g_pool, ack_idx, ack_len, ep, now_ms));

  // Fill to capacity+1 to force a drop of the oldest entry.
  std::array<PoolIdx, NET_SIM_DELAY_QUEUE_SIZE> held{};
  for (uint32_t i = 0; i < NET_SIM_DELAY_QUEUE_SIZE; ++i) {
    held[i] = g_pool.acquire();
    CHECK(held[i] != kInvalidPoolIdx);
    g_pool.ptr(held[i])[0] = NET_CHANNEL_UNRELIABLE;
    CHECK(sim.enqueue_send(g_pool, held[i], NET_HEADER_SIZE + 1, ep, now_ms));
  }

  // Oldest entry was ACK-only on a reliable channel: must have been released.
  const PoolIdx reacq = g_pool.acquire();
  CHECK(reacq == ack_idx);
  g_pool.release(reacq);

  std::array<SimSend, NET_SIM_DELAY_QUEUE_SIZE> send_out{};
  std::array<RawRecv, 1> recv_out{};
  uint32_t send_count = 0;
  uint32_t recv_count = 0;

  sim.drain_ready(g_pool, now_ms + static_cast<uint32_t>(p.latency_ms), send_out,
                  send_count,
                  recv_out, recv_count);
  CHECK(recv_count == 0);
  CHECK(send_count == NET_SIM_DELAY_QUEUE_SIZE);

  for (uint32_t i = 0; i < send_count; ++i) {
    CHECK(send_out[i].buf_idx != ack_idx);
    g_pool.sim_send_ref_dec(send_out[i].buf_idx);
    g_pool.release(send_out[i].buf_idx);
  }

  return true;
}

#endif

static bool test_action_builder_and_reader_round_trip() {
  using namespace attome::net;

  ActionBuilder b{};
  b.reset();

  constexpr int8_t kI8 = -7;
  constexpr uint8_t kU8 = 200;
  constexpr int16_t kI16 = -12345;
  constexpr uint16_t kU16 = 54321;
  constexpr int32_t kI32 = -123456789;
  constexpr uint32_t kU32 = 0xDEADBEEFu;
  constexpr float kF32 = 1.5f;
  constexpr std::string_view kStr = "hello";

  b.write_i8(kI8)
      .write_u8(kU8)
      .write_i16(kI16)
      .write_u16(kU16)
      .write_i32(kI32)
      .write_u32(kU32)
      .write_f32(kF32)
      .write_str(kStr);

  CHECK(!b.overflowed());
  const auto payload = b.build();
  CHECK(!payload.empty());

  MessageReader r{payload, 0};
  CHECK(r.read_i8() == kI8);
  CHECK(r.read_u8() == kU8);
  CHECK(r.read_i16() == kI16);
  CHECK(r.read_u16() == kU16);
  CHECK(r.read_i32() == kI32);
  CHECK(r.read_u32() == kU32);
  CHECK(r.read_f32() == kF32);

  const std::string_view sv = r.read_str();
  CHECK(sv == kStr);

  // Verify zero-copy: string bytes start after 18 bytes of scalars + 2 bytes len.
  constexpr size_t kExpectedStrOffset = 18 + 2;
  CHECK(sv.data() ==
        reinterpret_cast<const char *>(payload.data() + kExpectedStrOffset));

  CHECK(r.ok());
  CHECK(r.remaining().empty());

  return true;
}

static bool test_action_builder_overflow_and_reader_bounds() {
  using namespace attome::net;

  // Overflow: (u16 length prefix + bytes) > NET_MTU.
  ActionBuilder b{};
  b.reset();

  static std::array<char, NET_MTU> big{};
  big.fill('x');

  b.write_str(std::string_view{big.data(), big.size()});
  CHECK(b.overflowed());
  CHECK(b.build().empty());

  // After overflow, further writes should remain safe and produce empty build().
  b.write_u8(1).write_u32(2);
  CHECK(b.overflowed());
  CHECK(b.build().empty());

  // Reader bounds: reading past end forces ok()==false.
  const uint8_t one = 0xAB;
  MessageReader r{std::span<const uint8_t>{&one, 1}, 0};
  (void)r.read_u32();
  CHECK(!r.ok());

  return true;
}

struct TestCase {
  const char *name{};
  bool (*fn)(){};
};

} // namespace

namespace attome::net {

bool run_self_tests() {
  const TestCase cases[] = {
      {"PacketPool basic", &test_packet_pool_basic},
      {"ConnectionSlots alloc/free/context",
       &test_connection_slots_alloc_free_and_context},
      {"SpscQueue order/capacity", &test_spsc_queue_order_and_capacity},
      {"ReliabilityLayer ACK/retransmit",
       &test_reliability_layer_ack_and_retransmit},
      {"ReliabilityLayer ACK bits (non-contiguous)",
       &test_reliability_layer_ack_bits_non_contiguous},
      {"ReliabilityLayer bulk ACK advances head",
       &test_reliability_layer_bulk_ack_advances_head},
      {"ReorderBuffer out-of-order drain",
       &test_reorder_buffer_out_of_order_drain},
      {"ReorderBuffer in-order delivery", &test_reorder_buffer_in_order_delivery},
      {"ReorderBuffer wraparound delivery",
       &test_reorder_buffer_wraparound_delivery},
      {"ReorderBuffer window overflow + drain",
       &test_reorder_buffer_window_overflow_and_drain},
      {"ReorderBuffer duplicate drop", &test_reorder_buffer_duplicate_drop},
      {"OpcodeRegistry basic", &test_opcode_registry},
      {"MessageBatcher MTU overflow flush",
       &test_message_batcher_mtu_overflow_flushes},
      {"MessageBatcher wire layout", &test_message_batcher_unreliable_wire_layout},
      {"NetSocket loopback send/recv", &test_net_socket_loopback_send_recv},
      {"Server+Client multi-client echo (CH0/CH1/CH2)",
       &test_server_client_multi_client_channels_echo},
      {"Client ACK-only drains server retransmit window",
       &test_server_client_ack_only_allows_large_reliable_stream},
#if NET_ENCRYPTION
      {"CryptoLayer encrypt/decrypt + tamper",
       &test_crypto_layer_encrypt_decrypt_and_tamper},
#endif
#if NET_SIM
      {"SimLayer latency drain", &test_sim_layer_latency_drain},
      {"SimLayer overflow drops oldest ACK-only",
       &test_sim_layer_overflow_drops_oldest_ack_only},
#endif
      {"ActionBuilder/MessageReader round-trip",
       &test_action_builder_and_reader_round_trip},
      {"Overflow + reader bounds", &test_action_builder_overflow_and_reader_bounds},
  };

  int failed = 0;
  const size_t total = sizeof(cases) / sizeof(cases[0]);
  for (size_t idx = 0; idx < total; ++idx) {
    const TestCase &tc = cases[idx];
    const size_t n = idx + 1;

    std::fprintf(stderr, "RUN  [%zu/%zu] %s\n", n, total, tc.name);
    std::fflush(stderr);

    if (!tc.fn()) {
      std::fprintf(stderr, "FAIL [%zu/%zu] %s\n", n, total, tc.name);
      std::fprintf(stderr, "  test: %s\n", tc.name);
      ++failed;
    } else {
      std::fprintf(stderr, "PASS [%zu/%zu] %s\n", n, total, tc.name);
    }
  }

  if (failed == 0) {
    std::printf("OK (%zu tests)\n", total);
    return true;
  }

  std::fprintf(stderr, "FAILED (%d/%zu)\n", failed, total);
  return false;
}

} // namespace attome::net
