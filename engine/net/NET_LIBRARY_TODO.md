# AAA Network Library — Full Implementation TODO

> See `NET_LIBRARY_PLAN.md` for architecture details, data flows, and optimization rationale.
>
> Status (2026-03-17): Phase 1-7 core implementation in place (Sim/Crypto/NetSocket/Server/Client + CMake/vcpkg wiring). Self-tests now live inside the library via `attome::net::run_self_tests()` (enabled with `ATTOME_NET_SELF_TEST=ON`), covering multi-client CH0/CH1/CH2 echo + ACK-only one-way stream + Sim/Crypto checks (when enabled). NET_SIM delayed-send buffer safety is in place (SimLayer send refs + PacketPool deferred free). Next: game loop wiring + perf validation + cross-platform testing.

---

## Phase 1 — Foundation
*No Asio dependency. All items independently unit-testable.*

### NetTypes.h
- [x] Define `PacketHeader` struct (9 bytes packed: channel u8, seq u16, ack u16, ack_bits u32)
- [x] Add `static_assert(sizeof(PacketHeader) == NET_HEADER_SIZE)`
- [x] Define `OpcodeInfo` struct (`const char* name`, `int16_t payload_size`)
- [x] Define `ConnId` typedef (`uint16_t`, UINT16_MAX = invalid)
- [x] Define `PoolIdx` typedef (`uint16_t`, UINT16_MAX = invalid)
- [x] Define `RetransmitSlot` struct (buf_idx, seq, conn, sent_at_ms, len)
- [x] Define `ReorderEntry` struct (buf_idx, seq, len, occupied)
- [x] Define `InboundMsg` struct (conn, channel, buf_idx, payload_offset, payload_len)
- [x] Define `OutboundMsg` struct (conn, channel, opcode, len, staging[NET_MTU])
- [x] Define `RawRecv` struct (buf_idx, len, endpoint) — internal network thread use
- [x] Define `SendItem` struct (data ptr, len, endpoint ptr) — internal send_batch use
- [x] Add `static_assert` for `OutboundMsg` staging buffer alignment

### PacketPool
- [x] Declare `PacketPool` class in `PacketPool.h`
- [x] Add `alignas(64) uint8_t buffers[NET_PACKET_POOL_SIZE][NET_MTU]` field
- [x] Add `alignas(64) uint16_t free_stack[NET_PACKET_POOL_SIZE]` field
- [x] Add `std::atomic<uint64_t> free_top` field (tagged head to avoid ABA)
- [x] Implement `init()` — fill free_stack 0..N-1, set free_top = pack(tag=0, idx=N-1)
- [x] Implement `acquire()` — atomic CAS pop (Treiber stack), return UINT16_MAX if empty
- [x] Implement `release(PoolIdx)` — atomic CAS push back
- [x] Implement `ptr(PoolIdx)` — inline, returns `&buffers[idx][0]`
- [x] NET_SIM: defer `release()` while delayed SimLayer sends still reference a buffer (free on last `sim_send_ref_dec`)
- [x] Unit test: acquire all N buffers, verify no duplicates returned
- [x] Unit test: release all N, verify free_top returns to N-1
- [x] Unit test: acquire beyond capacity returns UINT16_MAX

### SpscQueue
- [x] Define `SpscQueue<T, N>` template in `SpscQueue.h` (N must be power-of-2)
- [x] Add `alignas(64) std::atomic<uint32_t> head` (consumer side)
- [x] Add `alignas(64) std::atomic<uint32_t> tail` (producer side)
- [x] Add `T ring[N]` array
- [x] Implement `push(const T&)` — producer only; `idx & (N-1)` mask
- [x] Implement `pop(T&)` — consumer only; `idx & (N-1)` mask
- [x] Add `static_assert` that N is power-of-2
- [x] Unit test: push 8192 items, pop all, verify values and order
- [x] Unit test: push when full returns false
- [x] Unit test: pop when empty returns false

### OpcodeRegistry
- [x] Define `OpcodeRegistry<Enum, N>` template in `OpcodeRegistry.h`
- [x] Add `OpcodeInfo table[N]` field
- [x] Implement `get(Enum op)` — inline, `return table[static_cast<uint16_t>(op)]`
- [x] Implement `is_dynamic(Enum op)` — inline, checks `payload_size == NET_PAYLOAD_DYNAMIC`
- [x] Implement `fixed_size(Enum op)` — inline, returns `payload_size`
- [x] Unit test: instantiate with dummy enum, verify get/is_dynamic/fixed_size correct
- [ ] Define game-side `C2SOpcode` enum (Connect, Disconnect, Move, Attack, Chat, ...)
- [ ] Define game-side `S2COpcode` enum (Welcome, SpawnEntity, UpdatePos, DestroyEntity, Chat, ...)
- [ ] Define `C2S_TABLE[]` opcode info array with correct payload sizes
- [ ] Define `S2C_TABLE[]` opcode info array with correct payload sizes
- [ ] Instantiate `g_c2s_registry` and `g_s2c_registry` globals

### ActionBuilder
- [x] Declare `ActionBuilder` class in `ActionBuilder.h`
- [x] Add `uint8_t staging[NET_MTU]` field (stack buffer)
- [x] Add `uint16_t pos`, `bool overflow` fields
- [x] Implement `reset()` — pos=0, overflow=false
- [x] Implement `write_i8(int8_t)` — bounds check, write LE, advance pos
- [x] Implement `write_u8(uint8_t)`
- [x] Implement `write_i16(int16_t)` — LE encoding
- [x] Implement `write_u16(uint16_t)`
- [x] Implement `write_i32(int32_t)`
- [x] Implement `write_u32(uint32_t)`
- [x] Implement `write_f32(float)` — memcpy to avoid UB
- [x] Implement `write_str(std::string_view)` — write u16 length + bytes
- [x] Implement `build()` — return empty span if overflow, else `{staging, pos}`
- [x] All write methods return `ActionBuilder&` for chaining
- [x] Unit test: round-trip each write type (write then read back raw bytes)
- [x] Unit test: overflow detection (write past NET_MTU)
- [x] Unit test: overflow after overflow is safe (no UB)

### MessageReader
- [x] Define `MessageReader` struct in `MessageReader.h` (header only)
- [x] Add `std::span<const uint8_t> data` and `size_t pos` fields
- [x] Implement `read_i8()`, `read_u8()`, `read_i16()`, `read_u16()`, `read_i32()`, `read_u32()`, `read_f32()`
- [x] Implement `read_str()` — reads u16 length, returns `string_view` into span (zero copy)
- [x] Implement `ok()` — `return pos <= data.size()`
- [x] Implement `remaining()` — `return data.subspan(pos)`
- [x] Unit test: ActionBuilder writes → MessageReader reads → all values match
- [x] Unit test: read_str returns string_view pointing into original buffer (no copy)
- [x] Unit test: reading past end leaves `ok() == false`

---

## Phase 2 — Connection State and Reliability

### ConnectionSlots
- [x] Declare `ConnectionSlots` struct in `ConnectionSlots.h`
- [x] Add HOT arrays (all `alignas(64)`):
  - [x] `uint8_t  alive[NET_MAX_CONNECTIONS]`
  - [x] `uint16_t send_seq[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
  - [x] `uint16_t recv_ack[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
  - [x] `uint32_t recv_ack_bits[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
  - [x] `uint16_t reorder_next_exp[NET_MAX_CONNECTIONS]`
  - [x] `uint8_t  retransmit_head[NET_MAX_CONNECTIONS][2]` (reliable CH1+CH2)
  - [x] `uint8_t  retransmit_tail[NET_MAX_CONNECTIONS][2]` (reliable CH1+CH2)
- [x] Add COLD arrays (no alignment requirement):
  - [x] `NetEndpoint endpoints[NET_MAX_CONNECTIONS]`
  - [x] `uint64_t connect_time_ms[NET_MAX_CONNECTIONS]`
  - [x] `#if NET_ENCRYPTION: uint8_t encryption_key[NET_MAX_CONNECTIONS][32]`
- [x] Add retransmit rings: `RetransmitSlot retransmit_slots[NET_MAX_CONNECTIONS][2][NET_RETRANSMIT_SLOTS]`
- [x] Add reorder rings: `ReorderEntry reorder_buf[NET_MAX_CONNECTIONS][NET_REORDER_BUF_SIZE]`
- [x] Add free-list: `uint16_t free_slot_stack[NET_MAX_CONNECTIONS]`, `int32_t free_slot_top`
- [x] Implement `init()` — memset HOT to 0, populate free_slot_stack 0..N-1
- [x] Implement `alloc_slot(endpoint)` — pop free stack, write cold data, set alive=1
- [x] Implement `free_slot(ConnId)` — zero HOT arrays for slot, push back, set alive=0
- [x] Implement `is_alive(ConnId)` — inline
- [x] Unit test: alloc 5000 slots, verify all alive
- [x] Unit test: free all, verify alive=0, free_slot_top=N-1
- [x] Unit test: alloc beyond capacity returns UINT16_MAX
- [ ] Verify HOT section size <= 256 KB (fits in L2 cache at 5000 conns)

### Connection Context (inline buffer)
- [x] Add `#define NET_CONTEXT_SIZE 256` to `NetConfig.h`
- [x] Add `static_assert(NET_CONTEXT_SIZE > 0)` to `NetConfig.h`
- [x] Add `alignas(16) uint8_t context_buf[NET_MAX_CONNECTIONS][NET_CONTEXT_SIZE]` to ConnectionSlots COLD section
- [x] Add `void (*context_dtor[NET_MAX_CONNECTIONS])(void*)` to ConnectionSlots COLD section
- [x] Zero `context_dtor` array in `init()`
- [x] Implement `emplace_context<T>(ConnId, Args&&...)`:
  - [x] `static_assert(sizeof(T) <= NET_CONTEXT_SIZE)` — compile-time size guard
  - [x] Call `clear_context(conn)` first if a context is already active
  - [x] `new (context_buf[conn]) T(std::forward<Args>(args)...)` — placement new
  - [x] Store `context_dtor[conn] = [](void* p){ static_cast<T*>(p)->~T(); }`
  - [x] Return `reinterpret_cast<T*>(context_buf[conn])`
- [x] Implement `get_context<T>(ConnId)`:
  - [x] Return `nullptr` if `context_dtor[conn] == nullptr` (no active context)
  - [x] Return `reinterpret_cast<T*>(context_buf[conn])`
- [x] Implement `clear_context(ConnId)`:
  - [x] If `context_dtor[conn] != nullptr`: call it with `context_buf[conn]`
  - [x] `memset(context_buf[conn], 0, NET_CONTEXT_SIZE)`
  - [x] Set `context_dtor[conn] = nullptr`
- [x] Call `clear_context(conn)` inside `free_slot()` — auto-cleanup on disconnect
- [x] Unit test: emplace `LoginContext`, read fields back via `get_context<LoginContext>`
- [x] Unit test: `clear_context` calls struct destructor (use a destructor with a side-effect flag)
- [x] Unit test: emplace overtop existing context — old dtor called, new context valid
- [x] Unit test: `get_context<T>` after `clear_context` returns nullptr
- [ ] Unit test: `static_assert` fires for struct larger than `NET_CONTEXT_SIZE`
- [x] Unit test: `free_slot()` auto-clears active context (destructor called)
- [ ] Integration test: login flow — emplace `LoginContext` on connect, verify challenge, `clear_context` + emplace `GameContext` on auth success

### ReliabilityLayer
- [x] Declare free functions in `ReliabilityLayer.h`
- [x] Implement `assign_seq(slots, conn, ch)` — return `send_seq[conn][ch]++` (wrapping)
- [x] Implement `record_sent(slots, pool, conn, ch, seq, buf_idx, len, now_ms)`:
  - [x] Check retransmit ring has space (tail-head < NET_RETRANSMIT_SLOTS)
  - [x] Write RetransmitSlot at `slots.retransmit_slots[conn][ridx][tail & mask]` (reliable CH1/CH2)
  - [x] Advance tail
  - [x] Return false if ring full (caller must drop the packet)
- [x] Implement `process_ack(slots, pool, conn, ch, ack, ack_bits)`:
  - [x] Walk retransmit ring head..tail
  - [x] For each slot: check if seq is within ack + ack_bits window
  - [x] If ACKed: pool.release(buf_idx), clear slot, advance head
- [x] Implement `retransmit_pass(slots, pool, conn, ch, now_ms, send_list)`:
  - [x] Walk retransmit ring
  - [x] For each slot where `now_ms - sent_at_ms >= NET_RETRANSMIT_TIMEOUT_MS`:
    - [x] Push SendItem{pool.ptr(buf_idx), len, &endpoints[conn]} to send_list
    - [x] Reset `sent_at_ms = now_ms`
- [x] Unit test: send 128 reliable packets, fill ring; ACK first 64; verify 64 released from pool
- [x] Unit test: retransmit_pass fires only after NET_RETRANSMIT_TIMEOUT_MS, not before
- [x] Unit test: ring full returns false from record_sent
- [x] Unit test: ACK bitfield correctly clears non-contiguous entries

### ReorderBuffer
- [x] Declare free functions in `ReorderBuffer.h`
- [x] Implement `insert(slots, pool, conn, seq, buf_idx, len, out_queue)`:
  - [x] Compute `dist = (uint16_t)(seq - slots.reorder_next_exp[conn])`
  - [x] `dist >= NET_REORDER_BUF_SIZE` → pool.release(), return false
  - [x] `dist == 0` → deliver + try_drain(), return true
  - [x] Compute `ring_idx = seq & (NET_REORDER_BUF_SIZE - 1)`
  - [x] Occupied slot → duplicate, pool.release(), return false
  - [x] Write entry, return true
- [x] Implement `try_drain(slots, pool, conn, out_queue)`:
  - [x] Loop while `ring[next_exp & mask].occupied` AND seq matches
  - [x] Push InboundMsg to out_queue
  - [x] Clear `.occupied`, increment `reorder_next_exp[conn]`
- [x] Unit test: in-order delivery (all dist==0) — all delivered immediately
- [x] Unit test: out-of-order — seq 5 arrives before seq 3; after seq 3 arrives, drain delivers 3,4,5
- [x] Unit test: wraparound — seq 65534, 65535, 0, 1 delivered in correct order
- [x] Unit test: window overflow — 65 packets with gap at 0 → 65th dropped
- [x] Unit test: duplicate detection — same seq twice, second dropped

---

## Phase 3 — Wire Protocol

### MessageBatcher
- [x] Declare `MessageBatcher` class in `MessageBatcher.h`
- [x] Add state arrays: `PoolIdx cur_buf[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
- [x] Add: `uint16_t cur_len[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
- [x] Add: `uint8_t msg_count[NET_MAX_CONNECTIONS][NET_CHANNEL_COUNT]`
- [x] Implement `init()` — set all cur_buf to UINT16_MAX (invalid)
- [x] Implement `write(pool, slots, conn, ch, opcode, OpcodeInfo, payload*, plen, now_ms, send_list)`:
  - [x] Acquire pool buf if cur_buf[conn][ch] == INVALID
  - [x] Write PacketHeader placeholder (zeros) at buf start
  - [x] Compute wire size: `NET_OPCODE_SIZE + (dynamic ? NET_DYNLEN_SIZE + plen : plen)`
  - [x] If `cur_len + wire_size > NET_MAX_PAYLOAD_ENC` → flush then acquire new buf
  - [x] If `msg_count >= NET_MAX_MESSAGES_PER_PACKET` → flush then acquire new buf
  - [x] Write opcode (2B LE) at cur position
  - [x] If dynamic: write plen as u16 LE
  - [x] memcpy payload into buffer (only copy in entire send path)
  - [x] Advance cur_len, msg_count
- [x] Implement `flush_all(pool, slots, now_ms, send_list)`:
  - [x] For each [conn][ch] with cur_buf != INVALID:
    - [x] Fill PacketHeader: channel, seq=assign_seq(), ack=recv_ack, ack_bits=recv_ack_bits
    - [x] CH1/CH2: call record_sent(); if ring full, log warning and skip send
    - [x] `#if NET_ENCRYPTION`: CryptoLayer::encrypt_in_place()
    - [x] Push SendItem to send_list
    - [x] Reset cur_buf[conn][ch] = UINT16_MAX
- [x] Unit test: pack 3 fixed-size messages, verify exact byte layout: header + opcode+payload repeated
- [x] Unit test: MTU overflow triggers flush and starts new buffer
- [x] Unit test: dynamic message encodes length prefix correctly
- [x] Unit test: msg_count limit triggers flush before MTU hit

---

## Phase 4 — Optional Layers

### SimLayer (NET_SIM=1 only)
- [x] Wrap entire `SimLayer.h` and `SimLayer.cpp` in `#if NET_SIM`
- [x] Define `SimEntry` struct (buf_idx, len, deliver_at_ms, endpoint, kind)
- [x] Add `SimEntry delay_ring[NET_SIM_DELAY_QUEUE_SIZE]`
- [x] Add `uint32_t head`, `uint32_t tail` (ring pointers, power-of-2 mask)
- [x] Implement `enqueue_send(pool, buf_idx, len, endpoint, now_ms)`:
  - [x] Roll send loss: `if (rand_float() < NET_SIM_SEND_LOSS)` → drop; `pool.release()` only for CH0/ACK-only; return true
  - [x] Compute `deliver_at = now_ms + NET_SIM_LATENCY_MS + rand_jitter()`
  - [x] If delayed: `pool.sim_send_ref_inc(buf_idx)` before queueing
  - [x] Push to delay_ring; if full, drop oldest (overwrite)
- [x] Implement `enqueue_recv(pool, buf_idx, len, endpoint, now_ms)`:
  - [x] Roll recv loss: `if (rand_float() < NET_SIM_RECV_LOSS)` → pool.release(), return true
  - [x] Same latency logic as send
- [x] Implement `drain_ready(now_ms, send_out, recv_out)`:
  - [x] Pop entries where `deliver_at_ms <= now_ms`, push to appropriate queue
  - [x] Send entries: consumer calls `pool.sim_send_ref_dec(buf_idx)` after actual send (drop paths decref immediately)
- [x] In `SimLayer.h` when `NET_SIM=0`: define empty inline no-op stubs for all methods
- [ ] Unit test (NET_SIM=1): 100% send loss → 0 packets reach send_out
- [ ] Unit test: 0% loss, 0ms latency → all packets delivered immediately
- [x] Unit test: 80ms latency → packets delivered only after 80ms in drain_ready

### CryptoLayer (NET_ENCRYPTION=1 only)
- [x] Wrap entire `CryptoLayer.h` and `CryptoLayer.cpp` in `#if NET_ENCRYPTION`
- [x] Add libsodium include: `#include <sodium.h>`
- [x] Implement nonce derivation (12 bytes): `conn_id(4B) | channel(1B) | seq(2B) | zeros(5B)`
- [x] Implement `encrypt_in_place(buf, len, conn, slots)`:
  - [x] Derive nonce from PacketHeader fields
  - [x] `crypto_aead_chacha20poly1305_ietf_encrypt_detached()` on `buf[NET_HEADER_SIZE..]`
  - [x] Append 16B tag after ciphertext
  - [x] `len += NET_ENCRYPTION_TAG_SIZE`
- [x] Implement `decrypt_in_place(buf, len, conn, slots)`:
  - [x] Verify AEAD tag; return false on failure
  - [x] Decrypt `buf[NET_HEADER_SIZE..]` in place
  - [x] `len -= NET_ENCRYPTION_TAG_SIZE`
- [x] `#if NET_ENCRYPTION_KEY_MODE == 1`: implement X25519 ECDH key exchange
  - [x] Generate server keypair at init (`crypto_box_keypair`)
  - [x] Include server public key in CONNECT_ACK packet
  - [x] On client HELLO: compute shared secret (`crypto_scalarmult`)
  - [x] HKDF-expand to 32-byte ChaCha key (`crypto_kdf_derive_from_key`)
  - [x] Store derived key in `slots.encryption_key[conn]`
- [x] In `CryptoLayer.h` when `NET_ENCRYPTION=0`: define no-op stubs
- [x] Unit test: encrypt then decrypt → plaintext matches original
- [x] Unit test: tamper 1 byte of ciphertext → decrypt returns false
- [x] Unit test: different conn nonce → decrypt with wrong nonce returns false

---

## Phase 5 — Network I/O

### NetSocket
- [x] Add Asio includes (avoid `asio.hpp` in public headers)
- [x] Declare `NetSocket` class in `NetSocket.h`
- [x] Add `asio::ip::udp::socket socket_` field
- [x] Add `asio::io_context& ioc_` reference
- [x] Implement `bind(uint16_t port)` — opens IPv4 UDP socket, binds to port
- [x] Implement `recv_loop(PacketPool&, SpscQueue<RawRecv>&)` coroutine:
  - [x] `co_await async_receive_from(pool.ptr(P), NET_MTU, ep, asio::use_awaitable)`
  - [x] Push `RawRecv{P, len, ep}` to raw_recv queue
  - [x] Loop indefinitely
- [x] Implement `send_batch(SendItem* items, int count)`:
  - [x] Windows: loop `WSASend()` or `::sendto()` per item (WSASendMsg if scatter-gather needed)
  - [x] Linux: `sendmmsg(fd, mmsghdr[], count, 0)` — one syscall
  - [x] Log warning if any individual send fails (EAGAIN/EWOULDBLOCK)
- [x] Unit test: bind two sockets on loopback, send 1000 packets from A to B, verify all received
- [x] Unit test: send_batch with 100 items — verify all 100 received by peer
- [x] Unit test: recv_loop correctly pushes RawRecv items to queue

---

## Phase 6 — Top-Level API

### Server
- [x] Declare `Server` class in `Server.h` with public API only
- [x] Add all subsystem fields (PacketPool, ConnectionSlots, MessageBatcher, NetSocket, etc.)
- [x] Add `SpscQueue<InboundMsg, 8192> net_to_game_`
- [x] Add `SpscQueue<OutboundMsg, 8192> game_to_net_`
- [x] Add `std::function<void(ConnId, MessageReader&)> c2s_handlers_[65536]`
- [x] Add `asio::io_context ioc_`
- [x] Add `std::thread net_thread_`
- [x] `#if NET_SIM` add `SimLayer sim_`
- [x] `#if NET_ENCRYPTION` add `CryptoLayer crypto_`
- [x] Implement `bind(uint16_t port)`:
  - [ ] Call `pool_.init()`, `slots_.init()`, `batcher_.init()`
  - [ ] Call `socket_.bind(port)`
  - [ ] Launch `net_thread_` running `ioc_.run()`
  - [ ] co_spawn `recv_loop_()` and `send_flush_loop_()`
- [x] Implement `on(C2SOpcode, Handler)` — store in `c2s_handlers_[opcode]`
- [x] Implement `send(ConnId, ch, S2COpcode, span)`:
  - [ ] Build OutboundMsg (copy payload into staging)
  - [ ] `game_to_net_.push(msg)` (may fail if full — log warning)
- [x] Implement `broadcast(ch, S2COpcode, span)`:
  - [ ] Iterate alive connection slots, call `send()` for each
- [x] Implement `tick(uint32_t delta_ms)` (called by game thread):
  - [ ] Drain `net_to_game_` queue
  - [ ] For each InboundMsg: parse opcode loop, dispatch handlers, pool.release()
  - [ ] Drain `game_to_net_` → call batcher.write() for each
  - [ ] Call batcher.flush_all() → fills send_list
  - [ ] Signal network thread to send (or let flush_loop timer pick up)
- [x] Implement `recv_loop_()` internal coroutine (network thread):
  - [ ] pool.acquire() → co_await recv_from → push RawRecv
  - [ ] `#if NET_SIM`: enqueue_recv sim
  - [ ] Parse header, lookup conn, decrypt if NET_ENCRYPTION
  - [ ] process_ack(), update recv_ack state
  - [ ] Channel dispatch (CH0/CH1/CH2 paths)
  - [ ] Push InboundMsg to net_to_game_
- [x] Implement `send_flush_loop_()` internal coroutine (network thread):
  - [ ] Timer: co_await every NET_FLUSH_INTERVAL_MS
  - [ ] retransmit_pass() for all alive connections
  - [ ] socket_.send_batch(send_list)
- [x] Handle new connection (CONNECT opcode): alloc_slot, send WELCOME
- [x] Handle disconnect (DISCONNECT opcode or timeout): free_slot
- [x] Integration test: multi-client Server↔Client echo (CH0/CH1/CH2), verifies reliable delivery + ordered channel behavior
- [x] Integration test: ACK-only one-way reliable stream (exercises retransmit window without reverse payload traffic)

### Client
- [x] Declare `Client` class in `Client.h`
- [x] Mirror Server internals but single-connection (ConnId=0 internally)
- [x] Implement `connect(host, port)`:
  - [x] Resolve endpoint (Asio resolver)
  - [x] Send CONNECT handshake packet
  - [x] `#if NET_ENCRYPTION_KEY_MODE==1`: ECDH key exchange
  - [x] co_await WELCOME ACK with timeout
- [x] Implement `on(S2COpcode, Handler)` — no ConnId in callback
- [x] Implement `send(ch, C2SOpcode, span)` — push to game_to_net_
- [x] Implement `tick(uint32_t delta_ms)` — same drain/dispatch as Server
- [x] Implement `disconnect()` — send DISCONNECT packet, teardown
- [x] Integration test: Client connects to Server, bidirectional message exchange (multi-client echo)
- [ ] Integration test: Client reconnect after disconnect

---

## Phase 7 — Build System and Integration

### CMakeLists.txt
- [x] Create `games/engine/net/CMakeLists.txt`
- [x] Define static library target `AttomeNet`
- [x] Add all source files to target
- [x] `find_package(asio REQUIRED)` + `target_link_libraries(AttomeNet PUBLIC asio::asio)`
- [x] `if(WIN32): target_link_libraries(AttomeNet PRIVATE ws2_32 mswsock)`
- [x] `option(ATTOME_NET_ENCRYPTION ...)` → passes `-DNET_ENCRYPTION=1` compile definition
- [x] `option(ATTOME_NET_SIM ...)` → passes `-DNET_SIM=1` compile definition
- [x] `if(ATTOME_NET_ENCRYPTION): find_package(unofficial-sodium REQUIRED)` + link
- [x] Add AttomeNet to game `CMakeLists.txt` via `target_link_libraries`
- [x] Add `asio` to `games/vcpkg.json`
- [x] Add `libsodium` to `games/vcpkg.json` (platform-conditional later if needed)
- [x] Verify clean build with NET_ENCRYPTION=0 NET_SIM=0
- [x] Verify clean build with NET_ENCRYPTION=1 NET_SIM=1

### Game Loop Integration
- [ ] Add `Server server(c2s_reg, s2c_reg)` to game init
- [ ] Call `server.bind(PORT)` at startup
- [ ] Register all C2S opcode handlers with `server.on(...)`
- [ ] Add `server.tick(delta_ms)` to main loop (after `engine_update`, before render)
- [ ] Wire interest management: inside player update handlers, use `engine->grid.queryRect()` for broadcast filtering
- [ ] Add `client.tick(delta_ms)` to client-side game loop

### Performance Validation
- [ ] Build release configuration (no NET_SIM, no NET_ENCRYPTION)
- [ ] Spawn 1000 loopback client connections; verify pool not exhausted
- [ ] Spawn 5000 loopback client connections; profile `tick()` duration
- [ ] Target: `tick()` completes in < 1ms for 5000 connections
- [ ] Verify zero heap allocations in steady state (use _CrtSetAllocHook on MSVC or Valgrind massif on Linux)
- [ ] Verify PacketPool `free_top` stable under sustained load (no leak)
- [ ] Measure throughput: 5000 conns × 1000 CH0 pkts/sec = 5M pkts/sec target
- [ ] Profile send_batch syscall count: verify 1 call per flush interval, not per packet

---

## Ongoing / Cross-Cutting

- [ ] Add `PROFILE_FUNCTION()` macros (from `ATMProfiler.h`) to `Server::tick()`, `recv_loop_()`, `send_flush_loop_()`
- [ ] Add connection timeout: if no packet received in N seconds, call `free_slot()`
- [x] Add standalone ACK-only packet logic (sent by flush_loop when outbound queue is idle but ACK pending)
- [x] Add `NET_FLUSH_INTERVAL_MS` define to `NetConfig.h`
- [ ] Add connection stats (optional): RTT estimate, packets sent/recv, bytes sent/recv per conn
- [ ] Document all public API methods with brief doc comments
- [x] Test on Windows (primary) with MSVC
- [ ] Test on Linux (secondary) with GCC for future cross-platform support
- [ ] Verify `sendmmsg` path on Linux compiles and works correctly
- [ ] Add example: minimal echo server + client using the library
