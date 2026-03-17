# AAA C++ Game Network Library — Architecture Plan

> Target: 10,000+ concurrent players · Zero heap allocation after init · C++20 · Asio standalone · Single UDP socket · 3 logical channels · Inline connection context

---

## Table of Contents

1. [Overview](#overview)
2. [File Layout](#file-layout)
3. [Class Reference](#class-reference)
4. [Connection Context](#connection-context)
5. [Memory Layout](#memory-layout)
6. [Send Data Flow](#send-data-flow)
7. [Recv Data Flow](#recv-data-flow)
8. [Ordered Channel (CH2) Detail](#ordered-channel-ch2-detail)
9. [Opcode System](#opcode-system)
10. [Optional Layers](#optional-layers)
11. [Key Optimization Decisions](#key-optimization-decisions)
12. [Implementation Order](#implementation-order)

---

## Overview

Single UDP socket with 3 logical channels differentiated by a 1-byte channel field in every packet header. All memory is pre-allocated at `Server::bind()` / `Client::connect()` — zero heap allocation in the hot path. Feature toggles (`NET_ENCRYPTION`, `NET_SIM`) are compile-time `#define`s in `NetConfig.h`; disabled features produce zero overhead (compiler eliminates dead code entirely).

### Channel Summary

| ID | Name | Guarantee |
|----|------|-----------|
| 0  | Unreliable | Fire and forget. No ordering. Fastest path. |
| 1  | Reliable Unordered | ACK + retransmit. Deliver immediately on recv. |
| 2  | Reliable Ordered | ACK + retransmit + reorder buffer. In-order delivery. |

### Thread Model

```
[Game Thread]                           [Network Thread]
  server.tick(dt)                         asio::io_context::run()
    drain net→game SPSC                     co_await recv_from
    dispatch handlers                       co_await send_flush_loop
    flush game→net SPSC
        ↕ SpscQueue (lock-free)         ↕ SpscQueue (lock-free)
```

---

## File Layout

```
games/engine/net/
├── NET_LIBRARY_PLAN.md         — this file
├── NET_LIBRARY_TODO.md         — full implementation checklist
│
├── CMakeLists.txt              — static lib target; adds asio + optional libsodium via vcpkg
│
├── NetTypes.h                  — all POD types, enums, constants derived from NetConfig.h
│
├── PacketPool.h                — shared flat packet buffer pool declaration
├── PacketPool.cpp              — init(), acquire(), release() implementations
│
├── ConnectionSlots.h           — SoA connection state (HOT and COLD split)
├── ConnectionSlots.cpp         — init(), alloc_slot(), free_slot()
│
├── ReliabilityLayer.h          — stateless reliability utilities
├── ReliabilityLayer.cpp        — assign_seq, record_sent, process_ack, retransmit_pass
│
├── ReorderBuffer.h             — CH2 ordered delivery ring (header only: insert(), try_drain())
│
├── MessageBatcher.h            — outbound per-connection per-channel batcher
├── MessageBatcher.cpp          — write(), flush_all()
│
├── OpcodeRegistry.h            — OpcodeRegistry<Enum, N> template (header only)
│
├── ActionBuilder.h             — fluent stack-based payload writer
├── ActionBuilder.cpp           — write_i8/i16/u32/f32/str, build()
│
├── MessageReader.h             — zero-copy span-based reader (header only)
│
├── SpscQueue.h                 — lock-free SPSC ring buffer (header only)
│
├── NetSocket.h                 — UDP socket wrapper (Asio)
├── NetSocket.cpp               — bind, recv coroutine, send_batch
│
├── SimLayer.h                  — #if NET_SIM: packet loss + latency delay queue
├── SimLayer.cpp                — enqueue_send, enqueue_recv, drain_ready
│
├── CryptoLayer.h               — #if NET_ENCRYPTION: ChaCha20-Poly1305 AEAD
├── CryptoLayer.cpp             — encrypt_in_place, decrypt_in_place, key exchange
│
├── Server.h                    — public Server class
├── Server.cpp                  — bind, on, send, broadcast, tick
│
├── Client.h                    — public Client class
└── Client.cpp                  — connect, on, send, tick
```

**Total: 21 source files.** Each file has a single auditable responsibility.

---

## Class Reference

### NetTypes.h — POD types (no methods, no vtables)

```cpp
// Packet header overlay — 9 bytes, packed
struct PacketHeader {
    uint8_t  channel;    // 0=unreliable, 1=rel-unord, 2=rel-ord
    uint16_t seq;        // sender's packet sequence number
    uint16_t ack;        // last seq received from remote
    uint32_t ack_bits;   // sliding window: which of prev 32 were received
};
static_assert(sizeof(PacketHeader) == NET_HEADER_SIZE);

// Opcode table entry
struct OpcodeInfo {
    const char* name;
    int16_t     payload_size;   // -1 = NET_PAYLOAD_DYNAMIC
};

using ConnId   = uint16_t;   // index into ConnectionSlots; UINT16_MAX = invalid
using PoolIdx  = uint16_t;   // index into PacketPool;      UINT16_MAX = invalid

struct RetransmitSlot {
    PoolIdx  buf_idx;       // which pool buffer holds the packet
    uint16_t seq;           // sequence number for this slot
    uint16_t conn;          // owning connection
    uint32_t sent_at_ms;    // timestamp for retransmit timeout check
    uint16_t len;           // total datagram length in bytes
};

struct ReorderEntry {
    PoolIdx  buf_idx;
    uint16_t seq;
    uint16_t len;
    bool     occupied;
};

// Passed over net→game SPSC queue
struct InboundMsg {
    ConnId   conn;
    uint8_t  channel;
    PoolIdx  buf_idx;
    uint16_t payload_offset;  // byte offset into pool buffer where messages start
    uint16_t payload_len;
};

// Passed over game→net SPSC queue (staging buffer embedded — no pool touch on game thread)
struct OutboundMsg {
    ConnId   conn;
    uint8_t  channel;
    uint16_t opcode;
    uint16_t len;
    uint8_t  staging[NET_MTU];
};
```

---

### PacketPool

**Responsibility:** Pre-allocated flat slab of `NET_PACKET_POOL_SIZE` buffers, each `NET_MTU` bytes. Free-list is an atomic Treiber stack of `PoolIdx` values.

```
Fields:
  alignas(64) uint8_t  buffers[NET_PACKET_POOL_SIZE][NET_MTU]
  alignas(64) uint16_t free_stack[NET_PACKET_POOL_SIZE]
  std::atomic<uint64_t> free_top   // tagged head to avoid ABA: [tag:48][idx:16]

Methods:
  void    init()                 — fill free_stack, set free_top = pack(tag=0, idx=N-1)
  PoolIdx acquire()              — atomic pop; UINT16_MAX if exhausted
  void    release(PoolIdx)       — atomic push back
  uint8_t* ptr(PoolIdx)          — returns &buffers[idx][0]
```

**Threading:** Only the network thread calls `acquire`/`release` in steady state. No cross-thread contention.

**NET_SIM note:** Delayed sends keep `PoolIdx` references alive; `release()` defers while a buffer is still referenced by queued SimLayer send entries (prevents pool reuse/UAF under loss + retransmits).

---

### ConnectionSlots

**Responsibility:** SoA arrays for all per-connection state, pre-allocated for `NET_MAX_CONNECTIONS` entries. Split into HOT and COLD sections to preserve L2/L3 cache efficiency on the hot path.

```
HOT (alignas(64), accessed every packet):
  uint8_t  alive            [NET_MAX_CONNECTIONS]
  uint16_t send_seq         [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]
  uint16_t recv_ack         [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]
  uint32_t recv_ack_bits    [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]
  uint16_t reorder_next_exp [NET_MAX_CONNECTIONS]   // CH2 only
  uint8_t  retransmit_head  [NET_MAX_CONNECTIONS]
  uint8_t  retransmit_tail  [NET_MAX_CONNECTIONS]

COLD (not cache-aligned, accessed on connect/disconnect only):
  NetEndpoint endpoints             [NET_MAX_CONNECTIONS]
  uint64_t connect_time_ms           [NET_MAX_CONNECTIONS]
  uint8_t  encryption_key[32]        [NET_MAX_CONNECTIONS]  // NET_ENCRYPTION only

Connection context (inline buffer — see Connection Context section):
  alignas(16) uint8_t context_buf    [NET_MAX_CONNECTIONS][NET_CONTEXT_SIZE]
  void (*context_dtor[NET_MAX_CONNECTIONS])(void*)           // nullptr = no context active

Retransmit rings:
  RetransmitSlot retransmit_slots[NET_MAX_CONNECTIONS][NET_RETRANSMIT_SLOTS]

Reorder rings (CH2):
  ReorderEntry reorder_buf[NET_MAX_CONNECTIONS][NET_REORDER_BUF_SIZE]

Slot allocator:
  uint16_t free_slot_stack[NET_MAX_CONNECTIONS]
  int32_t  free_slot_top

Methods:
  void   init()
  ConnId alloc_slot(const NetEndpoint&)
  void   free_slot(ConnId)
  bool   is_alive(ConnId) const   — inline, reads alive[]
```

**HOT size:** ~145 KB at 5000 connections. Fits in L2/L3 cache.

---

### ReliabilityLayer

**Responsibility:** Stateless utility functions (no member state) operating on `ConnectionSlots` data.

```
static functions:
  uint16_t assign_seq(slots, conn, ch)
              — increment send_seq[conn][ch], return old value

  bool record_sent(slots, pool, conn, seq, buf_idx, len, now_ms)
              — store in retransmit ring; return false if ring full

  void process_ack(slots, pool, conn, ack, ack_bits)
              — walk retransmit ring; pool.release() confirmed slots

  void retransmit_pass(slots, pool, conn, now_ms, send_list)
              — scan ring; enqueue entries older than NET_RETRANSMIT_TIMEOUT_MS
```

---

### ReorderBuffer

**Responsibility:** Stateless utility for CH2 ordered delivery. Operates on `ConnectionSlots::reorder_buf[conn]` and `reorder_next_exp[conn]`.

```
static functions:
  bool insert(slots, pool, conn, seq, buf_idx, len)
    dist = (uint16_t)(seq - reorder_next_exp[conn])   // wrapping
    if dist >= NET_REORDER_BUF_SIZE  → drop (pool.release), return false
    if dist == 0                     → deliver directly + try_drain()
    ring_idx = seq & (NET_REORDER_BUF_SIZE - 1)       // mask, no modulo
    if slot occupied                 → duplicate, drop, return false
    write entry, return true

  void try_drain(slots, pool, conn, net_to_game_queue)
    loop while ring[next_exp & mask].occupied:
      push InboundMsg to SPSC
      clear slot
      next_exp++   // wrapping uint16 — handles 65535→0 naturally
```

---

### MessageBatcher

**Responsibility:** Per-connection, per-channel outbound write cursor into a pool buffer. Accumulates messages, seals when MTU fills or message count hits `NET_MAX_MESSAGES_PER_PACKET`.

```
State arrays (indexed [conn][ch]):
  PoolIdx  cur_buf   [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]
  uint16_t cur_len   [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]
  uint8_t  msg_count [NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]

Methods:
  bool write(pool, conn, ch, opcode, payload*, plen)
    1. Acquire pool buf if none current
    2. Check fit (MTU + msg count limit) → flush first if needed
    3. Write opcode(2B) + [dynlen(2B)] + payload into pool buffer
    4. Advance cur_len, msg_count

  void flush_all(slots, reliability, send_list, now_ms)
    For each [conn][ch] with cur_buf != INVALID:
      Fill PacketHeader: channel, assign_seq(), recv_ack, recv_ack_bits
      For reliable ch: record_sent()
      [encrypt if NET_ENCRYPTION]
      Push to send_list
      Reset cur_buf = INVALID
```

---

### OpcodeRegistry\<Enum, N\> (header only, template)

```cpp
template<typename Enum, size_t N>
struct OpcodeRegistry {
    OpcodeInfo table[N];

    const OpcodeInfo& get(Enum op) const;
    bool   is_dynamic(Enum op) const;
    int16_t fixed_size(Enum op) const;
};

// Two global instances provided by game code:
extern OpcodeRegistry<C2SOpcode, C2S_COUNT> g_c2s_registry;
extern OpcodeRegistry<S2COpcode, S2C_COUNT> g_s2c_registry;
```

---

### ActionBuilder

**Responsibility:** Fluent, bounds-checked payload writer. Lives on the stack — no pool interaction. Returns `std::span<const uint8_t>` for passing to `send()`.

```
Fields:
  uint8_t  staging[NET_MTU]   // stack buffer
  uint16_t pos
  bool     overflow

Methods (all return ActionBuilder& for chaining):
  reset()
  write_i8 / write_u8 / write_i16 / write_u16
  write_i32 / write_u32 / write_f32
  write_str(string_view)       // writes uint16 length + bytes

  std::span<const uint8_t> build()
    → empty span if overflow, else {staging, pos}
```

---

### MessageReader (header only)

```cpp
struct MessageReader {
    std::span<const uint8_t> data;
    size_t pos = 0;

    int8_t   read_i8();
    uint8_t  read_u8();
    int16_t  read_i16();   // little-endian
    uint16_t read_u16();
    int32_t  read_i32();
    uint32_t read_u32();
    float    read_f32();
    std::string_view read_str();   // zero-copy: points into pool buffer

    bool ok() const;               // pos <= data.size()
    std::span<const uint8_t> remaining() const;
};
```

---

### SpscQueue\<T, N\> (header only, N must be power-of-2)

```
Fields:
  alignas(64) std::atomic<uint32_t> head   // consumer
  alignas(64) std::atomic<uint32_t> tail   // producer
  T ring[N]

Methods:
  bool push(const T&)    — producer only; false if full
  bool pop(T&)           — consumer only; false if empty

Index mask: idx & (N - 1)   — never modulo
```

Cache-line padding on `head` and `tail` prevents false sharing between producer and consumer cores.

---

### NetSocket

**Responsibility:** Asio `udp::socket` wrapper providing a `co_await`-able recv loop and a batched send method.

```
Fields:
  asio::ip::udp::socket socket_
  asio::io_context&     ioc_

Coroutines:
  asio::awaitable<void> recv_loop(PacketPool&, SpscQueue<RawRecv>&)
    loop:
      P = pool.acquire()
      if INVALID: sleep 1ms, retry
      co_await async_receive_from into pool.ptr(P)
      push RawRecv{P, len, endpoint} to raw_recv queue

Methods:
  void send_batch(SendItem* items, int count)
    Windows: loop WSASend (or WSASendMsg)
    Linux:   sendmmsg(fd, mmsghdr[], count, 0)
```

---

### Server

```
Fields:
  PacketPool          pool
  ConnectionSlots     slots
  MessageBatcher      batcher
  NetSocket           socket_
  asio::io_context    ioc_
  std::thread         net_thread_
  SpscQueue<InboundMsg,  8192> net_to_game_
  SpscQueue<OutboundMsg, 8192> game_to_net_
  Handler tables:
    std::function<void(ConnId, MessageReader&)> c2s_handlers[65536]
  OpcodeRegistry<C2SOpcode>& c2s_reg_
  OpcodeRegistry<S2COpcode>& s2c_reg_
  #if NET_SIM       SimLayer    sim_
  #if NET_ENCRYPTION CryptoLayer crypto_

Public API:
  bool bind(uint16_t port)
  void on(C2SOpcode, Handler)
  void send(ConnId, uint8_t ch, S2COpcode, std::span<const uint8_t>)
  void broadcast(uint8_t ch, S2COpcode, std::span<const uint8_t>)
  void tick(uint32_t delta_ms)   ← call from game loop

Internal coroutines (network thread):
  asio::awaitable<void> recv_loop_()
  asio::awaitable<void> send_flush_loop_()
```

---

### Client

Mirrors Server for a single connection (ConnId = 0 internally).

```
Public API:
  bool connect(const char* host, uint16_t port)
  void on(S2COpcode, std::function<void(MessageReader&)>)
  void send(uint8_t ch, C2SOpcode, std::span<const uint8_t>)
  void tick(uint32_t delta_ms)   ← call from game loop
  void disconnect()
```

---

## Connection Context

### Purpose

A connection often needs to carry temporary per-connection state across multiple packets — e.g. a login challenge key the server sends and must remember until the client replies, or post-auth player data. This is called **Session State** or **Connection Context** in networking. The specific auth pattern (server sends challenge → client replies → server verifies) is **Challenge-Response Authentication**.

### Design: Inline Buffer (not pointer, not heap)

Each connection slot contains a fixed-size byte buffer (`NET_CONTEXT_SIZE` bytes). Game code uses placement-new to construct any struct into it. Only one context is active per connection at a time — a connection is always in exactly one state (logging in, in-game, spectating), never two simultaneously.

**Why inline buffer and not `void*` or a pool:**

| Approach | Zero-alloc | Type safe | Simple |
|---|---|---|---|
| `void*` | game's problem | no | yes |
| **Inline buffer** | yes | yes | yes |
| Pool + type ID registry | yes | yes | no — only needed for multi-context per conn |

### NetConfig.h addition

```cpp
// Per-connection inline context buffer size (bytes).
// static_assert fires at compile time if your context struct exceeds this.
#define NET_CONTEXT_SIZE  256
```

### ConnectionSlots fields (COLD section)

```cpp
alignas(16) uint8_t  context_buf [NET_MAX_CONNECTIONS][NET_CONTEXT_SIZE];
void               (*context_dtor[NET_MAX_CONNECTIONS])(void*);  // nullptr = empty
```

### Server / Client API

```cpp
// Construct a context in-place for this connection. Replaces any existing context.
template<typename T, typename... Args>
T* emplace_context(ConnId conn, Args&&... args);

// Retrieve the active context. Returns nullptr if buffer is empty.
template<typename T>
T* get_context(ConnId conn);

// Explicitly destroy the active context (calls destructor, zeros buffer).
void clear_context(ConnId conn);
```

`free_slot()` calls `clear_context()` automatically — no leaks on disconnect.

### Usage Example

```cpp
struct LoginContext {
    uint64_t challenge;      // key sent to client
    uint64_t expires_at_ms;  // drop connection if reply not received in time
};

struct GameContext {
    uint32_t account_id;
    char     username[32];
    bool     admin;
};

// On new connection — create temporary login state
server.on(C2SOpcode::Connect, [](ConnId conn, MessageReader& r) {
    auto* ctx = server.emplace_context<LoginContext>(conn);
    ctx->challenge    = generate_challenge();
    ctx->expires_at_ms = now_ms() + 5000;
    // send challenge to client...
});

// On login reply — verify, then promote to game state
server.on(C2SOpcode::LoginReply, [](ConnId conn, MessageReader& r) {
    auto* ctx = server.get_context<LoginContext>(conn);
    if (!ctx || r.read_u64() != ctx->challenge) {
        server.disconnect(conn);
        return;
    }
    server.clear_context(conn);                    // free temporary login state
    auto* game = server.emplace_context<GameContext>(conn);
    game->account_id = lookup_account(...);
    // connection is now in game state
});

// On disconnect — context freed automatically by free_slot()
```

### Memory cost

`5000 connections × 256 bytes = 1.25 MB` (COLD section, never accessed on the packet hot path).

---

## Memory Layout

All allocated once at `init()`. Zero allocation afterwards.

```
┌──────────────────────────────────────────────────────────────────┐
│  PacketPool                                          ~11.2 MB    │
│  buffers[8192][1400]  = 11,468,800 B  (alignas 64)              │
│  free_stack[8192]     =     16,384 B                            │
├──────────────────────────────────────────────────────────────────┤
│  ConnectionSlots HOT                                  ~145 KB    │
│  alive + seq + ack + ack_bits + reorder_next + ret heads        │
│  (alignas 64) — fits in L2/L3 cache at 5000 connections         │
├──────────────────────────────────────────────────────────────────┤
│  ConnectionSlots COLD                                ~1.65 MB    │
│  endpoints + connect_time + encryption_key (if on)              │
│  context_buf[5000][256]  = 1,280,000 bytes                      │
│  context_dtor[5000]      =    40,000 bytes (fn ptr 8B each)     │
├──────────────────────────────────────────────────────────────────┤
│  Retransmit rings                                     ~7.3 MB    │
│  RetransmitSlot[5000][128]  (12 B each)                         │
├──────────────────────────────────────────────────────────────────┤
│  Reorder rings (CH2)                                  ~2.4 MB    │
│  ReorderEntry[5000][64]     (8 B each)                          │
├──────────────────────────────────────────────────────────────────┤
│  MessageBatcher arrays                                 ~75 KB    │
│  cur_buf + cur_len + msg_count [5000][3]                        │
├──────────────────────────────────────────────────────────────────┤
│  SPSC net→game   InboundMsg[8192]                      ~96 KB    │
├──────────────────────────────────────────────────────────────────┤
│  SPSC game→net   OutboundMsg[8192]  (1406 B each)    ~11.5 MB   │
│  (staging buffer embedded — game thread never touches pool)      │
├──────────────────────────────────────────────────────────────────┤
│  SimLayer (NET_SIM=1 only)  SimEntry[512]              ~20 KB    │
└──────────────────────────────────────────────────────────────────┘
  TOTAL (NET_SIM=0, NET_ENC=0):  ≈ 35.3 MB
  TOTAL (NET_SIM=1, NET_ENC=1):  ≈ 35.5 MB
  (+1.25 MB for connection context buffers, NET_CONTEXT_SIZE=256)
```

---

## Send Data Flow

```
[GAME THREAD]
① server.send(conn=42, CH1, S2C_PLAYER_UPDATE, payload)
    Look up OpcodeInfo → is_dynamic? → needed wire size
    Build OutboundMsg: copy payload into msg.staging, set fields
    game_to_net_.push(msg)                ← lock-free, returns immediately
    No pool interaction on game thread.

[NETWORK THREAD — Asio io_context]
② send_flush_loop() wakes (timer every NET_FLUSH_INTERVAL_MS)
    while game_to_net_.pop(msg):
        batcher.write(pool, msg.conn, msg.ch, msg.opcode, msg.staging, msg.len)

③ batcher.write() internals:
    If no cur_buf: pool.acquire() → write PacketHeader placeholder
    If cur_len + wire_size > NET_MAX_PAYLOAD_ENC OR msg_count >= 32:
        flush current buffer (→ step ④), acquire new one
    Write opcode(2B) + [dynlen(2B)] + payload into pool buffer
    Advance cur_len, msg_count

④ batcher.flush_all():
    For each [conn][ch] with pending data:
    a. Fill PacketHeader:
         buf[0]   = channel
         buf[1:2] = assign_seq(conn, ch)     ← increments send_seq
         buf[3:4] = recv_ack[conn][ch]       ← piggybacked ACK
         buf[5:8] = recv_ack_bits[conn][ch]
    b. CH1/CH2: record_sent() → store in retransmit ring
    c. #if NET_ENCRYPTION: encrypt_in_place() → append 16B auth tag
    d. #if NET_SIM: enqueue_send() → may drop or delay
    e. Push SendItem{ptr, len, &endpoints[conn]} to send_list[]

⑤ retransmit_pass():
    Scan all retransmit rings for entries older than NET_RETRANSMIT_TIMEOUT_MS
    Re-enqueue those SendItems (same pool buf, no re-acquire)
    Reset sent_at_ms = now_ms

⑥ socket.send_batch(send_list, count)
    → WSASendMsg / sendmmsg — ONE syscall for all pending datagrams
```

---

## Recv Data Flow

```
[NETWORK THREAD]
① recv_loop() coroutine:
    P = pool.acquire()
    if INVALID: yield, retry
    co_await async_receive_from(pool.ptr(P), NET_MTU, ep)

② #if NET_SIM: SimLayer::enqueue_recv(P, len, ep, now_ms)
    Loss roll → pool.release(P), loop back if dropped
    Latency → push to delay ring with future deliver_at_ms, loop back

③ Parse PacketHeader: reinterpret_cast<PacketHeader*>(pool.ptr(P))
    Validate: len >= NET_HEADER_SIZE, channel in [0,2]
    Lookup conn = endpoint_to_conn(ep)      ← cold path hash map
    If new endpoint + CONNECT opcode → alloc_slot()

④ #if NET_ENCRYPTION: decrypt_in_place(pool.ptr(P), len, conn, slots)
    AEAD verify fails → pool.release(P), return

⑤ process_ack(slots, pool, conn, hdr.ack, hdr.ack_bits)
    → pool.release() for all confirmed retransmit slots (zero-copy)

⑥ Update recv ACK state (for outgoing piggyback):
    recv_ack[conn][ch]      = hdr.seq (if newer, wrapping)
    recv_ack_bits[conn][ch] = updated sliding window

⑦ Channel dispatch:
    CH0: push InboundMsg immediately (no dup check)
    CH1: dup check vs recv_ack window → push immediately if new
    CH2: ReorderBuffer::insert()
         ReorderBuffer::try_drain() → may push multiple InboundMsgs

[GAME THREAD — server.tick()]
⑧ while net_to_game_.pop(msg):
    cursor = pool.ptr(msg.buf_idx) + msg.payload_offset
    Parse messages until payload exhausted:
        opcode = read_u16(cursor)
        info   = c2s_reg_.get(opcode)
        plen   = info.is_dynamic ? read_u16(cursor) : info.fixed_size
        reader = MessageReader{ span{cursor, plen} }
        c2s_handlers[opcode](msg.conn, reader)   ← zero-copy into pool buf
        cursor += plen
    pool.release(msg.buf_idx)                     ← after ALL messages parsed
```

---

## Ordered Channel (CH2) Detail

### State (per connection in ConnectionSlots)

```
uint16_t reorder_next_exp[conn]       // next in-order seq expected
ReorderEntry reorder_buf[conn][64]    // ring, indexed by seq & 63
  .occupied   bool
  .seq        uint16_t
  .buf_idx    PoolIdx
  .len        uint16_t
```

### Sequence Number Arithmetic

All arithmetic uses wrapping `uint16_t`. Distance from A to B: `(uint16_t)(B - A)`. If distance >= 32768, B is behind A (late duplicate).

### Insert

```
dist = (uint16_t)(seq - reorder_next_exp[conn])

dist >= NET_REORDER_BUF_SIZE (64):
    → too far ahead OR late duplicate → pool.release(), return false

dist == 0:
    → in-order: deliver directly to SPSC + call try_drain()

0 < dist < 64:
    ring_idx = seq & 63   (mask — no modulo)
    if ring[ring_idx].occupied → duplicate → drop, return false
    ring[ring_idx] = { buf_idx, seq, len, occupied=true }
    return true  (try_drain NOT called — gap still present)
```

### try_drain

```
loop:
    ring_idx = reorder_next_exp[conn] & 63
    if NOT ring[ring_idx].occupied: break
    if ring[ring_idx].seq != reorder_next_exp[conn]: break  // stale (wraparound edge)

    push InboundMsg to net_to_game SPSC
    ring[ring_idx].occupied = false
    reorder_next_exp[conn]++   // wrapping uint16 — 65535→0 handled correctly
```

### Window Capacity

64-slot ring at 100ms retransmit timeout → supports ~640 packets/sec/connection on CH2 (1400B each ≈ 896 KB/sec). Sufficient for game state updates. If a packet is lost, retransmit fires within 100ms; subsequent buffered packets are held without blocking CH0 or CH1.

---

## Opcode System

### Two Separate Enums

```cpp
// Client → Server actions
enum class C2SOpcode : uint16_t {
    Connect     = 0x0001,
    Disconnect  = 0x0002,
    Move        = 0x0010,   // fixed payload: int16 x, int16 y, uint8 flags = 5B
    Attack      = 0x0011,   // fixed payload: uint32 target_id = 4B
    Chat        = 0x0020,   // dynamic payload: string
    C2S_COUNT
};

// Server → Client actions
enum class S2COpcode : uint16_t {
    Welcome         = 0x0001,   // dynamic
    SpawnEntity     = 0x0010,   // fixed: uint32 id, int16 x, int16 y, uint8 type = 9B
    UpdatePosition  = 0x0011,   // fixed: uint32 id, int16 x, int16 y = 8B
    DestroyEntity   = 0x0012,   // fixed: uint32 id = 4B
    ChatBroadcast   = 0x0020,   // dynamic
    S2C_COUNT
};
```

### Opcode Table

```cpp
constexpr OpcodeInfo C2S_TABLE[] = {
    // idx  name            payload_size (-1 = dynamic)
    [0x00]  { "INVALID",   0  },
    [0x01]  { "Connect",  -1  },
    [0x02]  { "Disconnect", 0  },
    // ...
    [0x10]  { "Move",      5  },
    [0x11]  { "Attack",    4  },
    [0x20]  { "Chat",     -1  },
};
```

### Wire Format

```
Fixed payload opcode:
  [opcode: 2B][payload: payload_size B]

Dynamic payload opcode:
  [opcode: 2B][length: 2B][payload: length B]
```

---

## Optional Layers

### Encryption (NET_ENCRYPTION=1)

**Cipher:** ChaCha20-Poly1305 AEAD (libsodium: `crypto_aead_chacha20poly1305_ietf`)

**Nonce derivation** (no nonce transmitted — derived from header):
```
nonce[12] = conn_id(4B) | channel(1B) | seq(2B) | zeros(5B)
```

**Encrypt:** Bytes `[NET_HEADER_SIZE .. len)` encrypted in-place. 16B Poly1305 tag appended. `len += 16`.

**Decrypt:** Tag verified. Bytes decrypted in-place. `len -= 16`. Fail = drop packet.

**MTU impact:** `NET_MAX_PAYLOAD_ENC = NET_MAX_PAYLOAD - 16` (defined in NetConfig.h derived section).

**Key exchange:**
- Mode 0 (NET_ENCRYPTION_KEY_MODE=0): pre-shared key from `NET_PSK` define
- Mode 1 (NET_ENCRYPTION_KEY_MODE=1): X25519 ECDH on connect handshake

### Simulation (NET_SIM=1)

**Send loss:** `if (rand_float() < NET_SIM_SEND_LOSS) { drop; if (ch==CH0 || ack_only) pool.release(P); }`

**Recv loss:** Same check before enqueuing received packet.

**Latency:** `deliver_at_ms = now_ms + NET_SIM_LATENCY_MS ± rand(NET_SIM_JITTER_MS)`

**Delay queue:** `SimEntry ring[NET_SIM_DELAY_QUEUE_SIZE]` — pre-allocated ring, power-of-2. Drained each tick: entries with `deliver_at_ms <= now_ms` are forwarded to their destination queues.

**Delayed send lifetime:** When a send is queued (deliver_at > now), SimLayer increments a per-buffer send refcount in PacketPool; the send flush loop decrements it after the datagram is actually sent (drop paths decref immediately).

---

## Key Optimization Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **Shared PacketPool (not per-connection)** | Burst traffic uses only what it needs. Idle connections use 0 buffers. O(1) acquire/release with atomic stack. |
| 2 | **SoA HOT/COLD split** | 145 KB HOT at 5000 conns fits in L2/L3. Cold data (endpoints, keys) never pollutes packet-processing cache lines. |
| 3 | **Power-of-2 rings everywhere** | `idx & (N-1)` = 1 instruction vs `idx % N` = division (20-80 cycles). Enforced by `static_assert` in NetConfig.h. |
| 4 | **SPSC queues between threads** | Strict producer/consumer per direction. No mutex on hot path. Cache-line-padded head/tail prevent false sharing. |
| 5 | **OutboundMsg embeds staging buffer** | Game thread never touches the pool's atomic free-list. Network thread is sole pool owner. Eliminates cross-thread contention. |
| 6 | **`sendmmsg` / `WSASendMsg` batch** | One syscall per flush tick. At 5k conns × 100 sends/sec = 500k sends/sec; per-send syscalls would cost ~2500ms/sec in kernel-crossing overhead alone. |
| 7 | **Zero-copy recv handlers** | `MessageReader` is a `std::span` into the pool buffer. No memcpy from UDP recv to handler callback. |
| 8 | **Compile-time feature elimination** | `#if NET_SIM=0` removes SimLayer entirely from binary. No branch prediction penalty, no dead code. Same for encryption. |
| 9 | **ACK piggybacking** | Every outgoing packet carries `recv_ack + recv_ack_bits` for free. Standalone ACK-only packets only sent when outbound queue is idle. |
| 10 | **Fixed retransmit timer** | Adaptive SRTT = float math + Karn's algorithm + per-conn state in hot path. Fixed 100ms is correct for 60Hz tick rate (6 frames). Add adaptive later if needed as cold-path addition. |
| 11 | **uint16 sequence numbers** | 16-bit wrap-around arithmetic handles 65535→0 naturally. Saves 2B per header vs uint32 (0.14% more payload per packet). |
| 12 | **Flush-on-timer not flush-on-write** | Bounds latency to `NET_FLUSH_INTERVAL_MS` (e.g. 10ms). Full MTU packets send immediately when filled. Avoids both per-message syscall overhead and Nagle-style unbounded wait. |
| 13 | **Inline context buffer over `void*`** | `void*` shifts lifetime responsibility to game code with no safety net. Inline buffer with placement new gives type-safe create/free with automatic cleanup on disconnect. `static_assert(sizeof(T) <= NET_CONTEXT_SIZE)` catches oversized contexts at compile time. No heap involved. |

---

## Implementation Order

Status (2026-03-17): Phase 1-7 core implementation landed (Sim/Crypto/NetSocket/Server/Client + CMake/vcpkg wiring). Self-tests now live inside the library via `attome::net::run_self_tests()` (enabled with `ATTOME_NET_SELF_TEST=ON`), covering Phases 1-3 plus reliability/reorder edge-cases, MessageBatcher MTU flush behavior, NetSocket loopback send/recv, basic Sim/Crypto checks (when enabled), and Server↔Client integration coverage (multi-client CH0/CH1/CH2 echo, plus an ACK-only one-way reliable stream test; optional stress via `SelfTestConfig{ .run_stress=true, .stress_iterations=N }`). Clean builds verified on Windows/MSVC for NET_ENCRYPTION=0/1 and NET_SIM=0/1. Remaining work is primarily game loop wiring, perf validation at scale, and cross-platform verification. See `NET_LIBRARY_TODO.md` for the live checklist.

### Phase 1 — Foundation (no Asio, unit-testable standalone)

```
1. NetTypes.h              — POD structs + static_asserts
2. PacketPool.h/cpp        — slab + free-list. Unit test: acquire N, release all, verify top.
3. SpscQueue.h             — SPSC ring. Unit test: push/pop 8192, verify order.
4. OpcodeRegistry.h        — template + test with dummy enum
5. ActionBuilder.h/cpp     — writer. Unit test: round-trip all write_* types.
6. MessageReader.h         — reader. Unit test: ActionBuilder → MessageReader values match.
```

### Phase 2 — Connection State and Reliability

```
7. ConnectionSlots.h/cpp   — SoA init + slot allocation. Test: alloc 5000, free all.
8. ReliabilityLayer.h/cpp  — retransmit ring. Test: fill ring, ACK half, verify half released.
9. ReorderBuffer.h         — CH2 ring (header-only). Test: out-of-order, wraparound, window overflow.
```

### Phase 3 — Wire Protocol

```
10. MessageBatcher.h/cpp   — batching. Test: verify exact wire byte layout against spec.
```

### Phase 4 — Optional Layers

```
11. SimLayer.h/cpp         — Test: 100% loss → 0 delivered. 0ms latency → all immediate.
12. CryptoLayer.h/cpp      — Test: encrypt→decrypt round-trip. Tampered ciphertext → false.
```

### Phase 5 — Network I/O

```
13. NetSocket.h/cpp        — Test: two loopback sockets, 1000 packets, verify all received.
```

### Phase 6 — Top-Level API

```
14. Server.h/cpp           — wire everything. Integration test: 5000-client loopback.
15. Client.h/cpp           — mirror of server.
```

### Phase 7 — Integration and Performance

```
16. CMakeLists.txt         — static lib target, vcpkg asio + libsodium deps.
17. Game loop wiring       — Server::tick() called alongside engine_update().
18. Performance validation — 5000 loopback clients, 1000 CH0 pkts/sec/conn.
                             Target: tick() < 1ms for 5000 connections.
```

---

## Dependencies to Add to vcpkg.json

```json
{
  "dependencies": [
    "asio",
    { "name": "libsodium", "features": [], "platform": "!(uwp)" }
  ]
}
```

`libsodium` only required when `NET_ENCRYPTION=1`. Consider wrapping the vcpkg dep in a CMake option so it's not fetched on `NET_ENCRYPTION=0` builds.

---

## Integration with ATMEngine

```cpp
// In ashlands-dominion/main.cpp game loop:
while (running) {
    // ... existing SDL event poll + delta time ...

    game.update(dt);
    engine_update(engine);     // existing — physics, hybrid updates
    server.tick(delta_ms);     // NEW — drain recv, flush send
    engine_render_scene(engine);
    engine_present(engine);
}
```

`ActionBuilder` lives on the stack inside handler callbacks. `MessageReader` is passed by reference — points directly into the pool buffer (zero copy). Pool buffer released automatically after `tick()` finishes processing that packet.
