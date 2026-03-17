#pragma once

// =============================================================================
//  NetConfig.h  —  Compile-time network configuration
//  All values are #defines so the compiler can size arrays statically and
//  eliminate disabled feature paths entirely (no runtime overhead).
// =============================================================================


// -----------------------------------------------------------------------------
//  CAPACITY
//  Memory is allocated once at startup based on these limits.
//  Total pool memory ≈ NET_PACKET_POOL_SIZE × NET_MTU bytes.
// -----------------------------------------------------------------------------

// Maximum simultaneous connections (server) / 1 for client
#define NET_MAX_CONNECTIONS         5000

// Shared packet buffer pool (not per-connection).
// Must be >= NET_MAX_CONNECTIONS to avoid starvation under load.
#define NET_PACKET_POOL_SIZE        8192

// Max raw UDP datagram size. Stay under typical internet MTU of 1472.
#define NET_MTU                     1400

// Per-connection retransmit slots (reliable channels).
// Limits how many unACKed packets can be in-flight per connection.
#define NET_RETRANSMIT_SLOTS        128

// Per-connection reorder buffer size (reliable-ordered channel only).
// Packets arriving further ahead than this are dropped and retransmitted.
#define NET_REORDER_BUF_SIZE        64

// Max messages batched into one outbound datagram per flush.
#define NET_MAX_MESSAGES_PER_PACKET 32

// Delay queue slots for simulation (see NET_SIM section below).
//
// NOTE: This queue is shared for both delayed sends and delayed receives per
// SimLayer instance. Under NET_SIM with many simultaneous clients, latency can
// require thousands of in-flight entries to avoid artificial packet loss from
// queue overflow.
#define NET_SIM_DELAY_QUEUE_SIZE    4096

// Per-connection inline context buffer size (bytes).
// Game code uses emplace_context<T> / get_context<T> / clear_context to manage it.
// A static_assert fires at compile time if your context struct exceeds this.
// Increase if your largest context struct requires more space.
#define NET_CONTEXT_SIZE            256


// -----------------------------------------------------------------------------
//  CHANNELS
// -----------------------------------------------------------------------------

// Number of logical channels per connection (do not change).
#define NET_CHANNEL_COUNT           3

#define NET_CHANNEL_UNRELIABLE      0   // fire and forget, no ordering
#define NET_CHANNEL_RELIABLE_UNORD  1   // ACKed + retransmit, any order
#define NET_CHANNEL_RELIABLE_ORD    2   // ACKed + retransmit + ordered delivery


// -----------------------------------------------------------------------------
//  PROTOCOL
// -----------------------------------------------------------------------------

// Packet header layout (bytes):
//   channel(1) + sequence(2) + ack(2) + ack_bits(4) = 9 bytes
#define NET_HEADER_SIZE             9

// Opcode width in bytes (uint16 → up to 65535 opcodes).
#define NET_OPCODE_SIZE             2

// Dynamic payload length prefix width (uint16 → up to 65535 bytes).
#define NET_DYNLEN_SIZE             2

// Sentinel value in the opcode table meaning payload size is dynamic.
#define NET_PAYLOAD_DYNAMIC         -1

// Max usable payload per datagram (reduced further when encryption is on).
#define NET_MAX_PAYLOAD             (NET_MTU - NET_HEADER_SIZE)


// -----------------------------------------------------------------------------
//  RELIABILITY
// -----------------------------------------------------------------------------

// Fixed retransmit timeout in milliseconds.
// In simulation builds we default to a higher timeout to avoid spurious
// retransmits when simulated RTT exceeds 100ms.
#ifndef NET_RETRANSMIT_TIMEOUT_MS
#  if NET_SIM
#    define NET_RETRANSMIT_TIMEOUT_MS                                           \
      (((NET_SIM_LATENCY_MS + NET_SIM_JITTER_MS) * 4) + 100)
#  else
#    define NET_RETRANSMIT_TIMEOUT_MS   100
#  endif
#endif

// ACK bitfield covers this many past packets (must be 32).
#define NET_ACK_BITS_COUNT          32

// Network thread flush timer interval (milliseconds).
// Bounds outbound latency when packets are not MTU-filled.
#define NET_FLUSH_INTERVAL_MS       10


// -----------------------------------------------------------------------------
//  ENCRYPTION  (ChaCha20-Poly1305 AEAD)
//  Set to 1 to enable. When 0, all crypto code is compiled out entirely.
// -----------------------------------------------------------------------------

#ifndef NET_ENCRYPTION
#define NET_ENCRYPTION              0
#endif

// AEAD authentication tag appended after encrypted payload (bytes).
// Reduces usable payload: NET_MAX_PAYLOAD - NET_ENCRYPTION_TAG_SIZE.
#define NET_ENCRYPTION_TAG_SIZE     16

// Key exchange mode:
//   0 = pre-shared key (set NET_PSK_* below)
//   1 = X25519 ECDH on connect (requires libsodium)
#ifndef NET_ENCRYPTION_KEY_MODE
#define NET_ENCRYPTION_KEY_MODE     0
#endif

// Pre-shared key (32 bytes, hex). Only used when NET_ENCRYPTION_KEY_MODE = 0.
#define NET_PSK \
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, \
      0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, \
      0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, \
      0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }


// -----------------------------------------------------------------------------
//  SIMULATION  (dev/testing only — compile out in release)
//  Set feature flag to 1 to enable. Values are fixed at compile time.
// -----------------------------------------------------------------------------

// Master switch — set to 0 in release builds to remove all sim overhead.
#ifndef NET_SIM
#define NET_SIM                     0
#endif

// NET_SIM runtime overrides (dev/test only, read by SimLayer::init):
//   ATTOME_NET_SIM_SEND_LOSS, ATTOME_NET_SIM_RECV_LOSS, ATTOME_NET_SIM_LATENCY_MS, ATTOME_NET_SIM_JITTER_MS

// Drop outbound packets with this probability  (0.0 = none, 1.0 = all).
#ifndef NET_SIM_SEND_LOSS
#define NET_SIM_SEND_LOSS           0.05f
#endif

// Drop inbound packets with this probability.
#ifndef NET_SIM_RECV_LOSS
#define NET_SIM_RECV_LOSS           0.05f
#endif

// Artificial inbound latency (milliseconds, applied before processing).
#ifndef NET_SIM_LATENCY_MS
#define NET_SIM_LATENCY_MS          80
#endif

// ± jitter added to NET_SIM_LATENCY_MS (uniform random).
#ifndef NET_SIM_JITTER_MS
#define NET_SIM_JITTER_MS           20
#endif


// -----------------------------------------------------------------------------
//  DERIVED — do not edit below this line
// -----------------------------------------------------------------------------

#if NET_ENCRYPTION
#   define NET_MAX_PAYLOAD_ENC  (NET_MAX_PAYLOAD - NET_ENCRYPTION_TAG_SIZE)
#else
#   define NET_MAX_PAYLOAD_ENC  NET_MAX_PAYLOAD
#endif

// Sanity checks
static_assert(NET_PACKET_POOL_SIZE >= NET_MAX_CONNECTIONS,
    "Packet pool too small: starvation possible under full load.");
static_assert(NET_MTU <= 1472,
    "MTU exceeds typical internet safe limit of 1472 bytes.");
static_assert(NET_REORDER_BUF_SIZE  > 0 && (NET_REORDER_BUF_SIZE  & (NET_REORDER_BUF_SIZE  - 1)) == 0,
    "NET_REORDER_BUF_SIZE must be a power of 2 (used as ring index mask).");
static_assert(NET_RETRANSMIT_SLOTS  > 0 && (NET_RETRANSMIT_SLOTS  & (NET_RETRANSMIT_SLOTS  - 1)) == 0,
    "NET_RETRANSMIT_SLOTS must be a power of 2.");
static_assert(NET_SIM_DELAY_QUEUE_SIZE > 0 && (NET_SIM_DELAY_QUEUE_SIZE & (NET_SIM_DELAY_QUEUE_SIZE - 1)) == 0,
    "NET_SIM_DELAY_QUEUE_SIZE must be a power of 2.");
static_assert(NET_CONTEXT_SIZE >= 8,
    "NET_CONTEXT_SIZE too small to be useful (minimum 8 bytes).");
