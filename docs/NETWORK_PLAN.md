# AttomeNet 2 — network library plan for 10,000 players per server

Status: proposal for review, revision 2 (target raised to 10,000 players per
server from the start; own-hardware hosting; X-macro message schema). Companion documents:
`docs/GAME_DESIGN.md` (game, zones, economy) and `docs/GPU_3D_PLAN.md`
(engine).

## 1. Goal

Our own network library that lets one zone server hold **10,000 players**
in fast Trove-style action combat. **10,000 is the hard requirement from the
start** — every limit, pool, id width and gate in this document is sized for
it.

Three requirements rank above features:

1. **Performance** — measured, gated at every milestone (§8).
2. **Reliability** — never crash, corrupt, leak or hang, whatever arrives on
   the wire or however overloaded the server is (§6A).
3. **Memory** — every byte has a budget; nothing sized for the worst case
   when the typical case is far smaller; memory is recycled with O(1),
   lock-free (mostly thread-local) operations (§5A).

"Our own library" means **evolving AttomeNet** (`engine/net/`), not starting
over: its transport core already has the right design (no allocation while
running, cache-friendly connection data, lock-free thread handoff, batching,
reliability, encryption). What it lacks is listed in §3.

## 2. What 10,000 players means in numbers

Assumptions: 30 Hz simulation, 20 Hz snapshots to clients, clients send input
at 30 Hz, ~150 relevant entities per client on average (more in hotspots).

| Quantity | Per player | Per server (5,000) | Per server (10,000) |
|---|---|---|---|
| Packets in (client input) | 30/s | 150,000/s | 300,000/s |
| Packets out (snapshots + reliable) | ~25/s | 125,000/s | 250,000/s |
| Bytes out | 25–60 KB/s | 125–300 MB/s (1–2.4 Gb/s) | 250–600 MB/s (2–4.8 Gb/s) |
| Bytes in | 2–4 KB/s | 10–20 MB/s | 20–40 MB/s |
| Snapshot builds | 20/s | 100,000/s | 200,000/s |
| Encryption (ChaCha20-Poly1305) | — | ~300 MB/s ≈ 1 core | ~600 MB/s ≈ 1–2 cores |

Consequences (the **10,000** column is the design point):

- **Bytes out are the main cost.** The replication layer (§6) — interest,
  priority, deltas, bit-packing — matters more than the socket layer.
- **Packet rate** needs batched system calls and several network threads
  (§5).
- **Snapshot building** must be parallel (many cores) and cheap per client.
- Servers need **25 Gb/s networking** (10 Gb/s is the absolute minimum and
  leaves no headroom at 4.8 Gb/s peak); 1 Gb/s caps a zone near ~2,000
  players in crowded conditions.
- Interest management and relevancy caps (§6.2) are not optional: without
  them, egress grows with players² in hotspots.

## 3. Current state of AttomeNet

| Area | Today | Needed at 10,000 |
|---|---|---|
| Connections | `NET_MAX_CONNECTIONS 5000`, 16-bit ids reused without a generation (D7) | 10,000 (configurable up to 65,536); 32-bit handles (16-bit index + 16-bit generation) |
| Threads | 1 network thread (Asio), 1 game thread | N network threads, each with its own socket (§5) |
| Receive | One `async_receive_from` per packet | Batched receive: `recvmmsg` / io_uring (Linux), RIO (Windows) |
| Send | `sendmmsg` on Linux; one `send_to` per packet on Windows | Batched on both; Linux is the production target |
| Retransmit timer | Fixed 100 ms | Adaptive per connection (RTT + variance) |
| Congestion / bandwidth | None | Per-connection send budget + global egress budget (§6.5) |
| Large messages | None (max 1 packet) | Fragmentation + reassembly, and bulk streams for chunks (§4.4) |
| Channels | 3 fixed | Unreliable, several independent ordered streams, bulk (§4.3) |
| Connection security | Encryption optional (PSK or X25519) | Token-based connect, anti-spoofing cookies, per-session keys always on (§4.2) |
| Game data | Byte-aligned opcodes (`ActionBuilder`) | Entity replication with bit-packing, quantisation and deltas (§6) |
| Tests | Unit tests, loopback self-test | Load-test harness with 10,000+ headless bot clients (§8) |

### 3.1 Known defects (code review, verified against the source)

A read-only review of `engine/net/` found problems that must be fixed before
any scaling work. Severity: **C** = critical (security, data loss, lockup),
**H** = high, **M** = medium. Line numbers refer to the current code.

| # | Sev | Defect | Where | Effect | Fix (milestone) |
|---|---|---|---|---|---|
| D1 | C | **Encryption nonce reuse.** Nonce = conn id + channel + 16-bit seq; same key both directions, both sides start at seq 0; ack-only packets reuse the next seq; seq wraps every ~44 min; PSK mode is one all-zero key for everyone; no replay protection on the unreliable channel | `CryptoLayer.cpp:26-31, 128-175`, `NetConfig.h:130` | ChaCha20-Poly1305 security is broken when a nonce repeats (keystream reuse, forgeable packets) | Per-direction keys from the connect token, 64-bit per-direction packet counter as nonce, replay window (N2) |
| D2 | C | **Slot exhaustion.** Any CONNECT from a new address allocates a slot, no challenge; **no idle timeout** (`connect_time_ms` is written, never read); no API to kick a connection | `Server.cpp:441-490`, `ConnectionSlots.cpp:67` | 5,000 spoofed packets lock out every real player until restart; dead clients never freed | Stateless challenge cookie + tokens, last-receive timeout, `Server::disconnect()` (N0.5 timeout/kick, N2 cookies) |
| D3 | C | **Pool starvation stops all receiving.** 8,192 shared buffers; each connection can pin 256 retransmit + 63 reorder buffers; retransmit never gives up; receive loop then backs off forever | `ReliabilityLayer.cpp:83-122`, `NetSocket.cpp:62-67` | ~32 clients that stop acking freeze the whole server | Per-connection in-flight quota, retry cap → disconnect, reserved receive partition (N0.5) |
| D4 | C | **Ack window (32) smaller than send (128) and reorder (64) windows.** Packets accepted >32 behind can never be acked | `Server.cpp:84-89`, `ReliabilityLayer.h:11-21` | Endless retransmit, pinned slots, then D5 | Wider ack field (e.g. 128-bit) or in-flight cap = ack window (N0.5) |
| D5 | C | **Reliable packet dropped after its seq was assigned** when the retransmit ring is full; `write()` failures ignored | `MessageBatcher.cpp:86-93, 388-411`, `Server.cpp:708` | Ordered channel stalls forever; reliable messages silently lost | Check capacity before assigning seq; backpressure instead of drop (N0.5) |
| D6 | C | **Acked but never delivered.** CH1 marks received/acks before the push to the game queue; push can fail | `Server.cpp:564-616`, `Client.cpp:492` | One slow game tick loses reliable messages permanently | Ack only after successful enqueue; size queue ≥ 2 ticks (N0.5) |
| D7 | H | **Unsafe connection id reuse.** LIFO free stack, no generation; DISCONNECT event push result ignored | `ConnectionSlots.cpp:54-61, 150-157`, `Server.cpp:586` | Queued data and game references reach the *next* player in that slot (cross-player data leak); missed disconnects leak buffers | 32-bit handle (index + generation), FIFO free list, guaranteed disconnect event (N0.5) |
| D8 | H | **Game/network thread data race** on `alive[]`; context destroyed on the network thread; SPSC single-producer not enforced; context API unreachable from `Server` | `Server.cpp:279, 322`, `ConnectionSlots.cpp:113` | Undefined behaviour; queue corruption if `send()` is called from two threads | Connection state owned by one thread; events for lifecycle; thread asserts (N0.5) |
| D9 | H | **1,424-byte `OutboundMsg` copied through the queue** (zero-fill + 3 copies ≈ 4.3 KB per 20-byte message); a 5,000-way broadcast fills the 8,192-entry queue; reliable sends then spin the game thread; `fprintf` per dropped message | `Server.cpp:279-322, 309-314` | ~640 MB/s of pointless memory traffic at target load; stalls on broadcast | Write directly into packet blocks (§5A.3); fan-out primitive for broadcast (N1) |
| D10 | H | **One network thread, one syscall per packet** (receive always; send on Windows); no socket buffer sizing; Linux fallback may block | `NetSocket.cpp:35-190` | Cannot reach 275k packets/s plus encryption | §5 multi-socket threads + batched I/O (N1) |
| D11 | H | **Sequence sentinel 0xFFFF** used for "nothing received", but 65535 is a real sequence | `ReliabilityLayer.h:391`, `Server.cpp:61` | Window reset at wrap → duplicate delivery, wrong acks | Separate "has received" flag (N0.5) |
| D12 | H | **Spoofable control packets**: plaintext DISCONNECT/CONNECT accepted in encrypted mode; spoofed acks in unencrypted mode; unauthenticated server key (MITM) in key mode 1; WELCOME re-sends unthrottled | `Server.cpp:531-540`, `Client.cpp:418-449` | Anyone who can spoof an IP can kick players | All post-handshake packets authenticated; server key signed/pinned via token (N2) |
| D13 | M | `NET_RETRANSMIT_SLOTS=256` allowed but a `uint8_t` ring can't represent full | `ConnectionSlots.h:19` | Silent overflow if configured | Replaced by shared record pool (§5A.4) (N1) |
| D14 | M | Pool free list: non-atomic array access across threads (formally UB); 1,400-byte stride not 64-byte aligned (false sharing) | `PacketPool.cpp:61, 101` | Rare corruption risk; cache-line sharing | Per-thread slabs, 1,280-byte aligned blocks (§5A) (N1) |
| D15 | M | Four O(connections) passes every 10 ms (reorder drain, flush all 15,000 conn×channel pairs, ack-only, retransmit) | `Server.cpp:682`, `send_flush_loop_` | ~4M iterations/s even when idle | Dirty lists + timer wheel (N1) |
| D16 | M | Receive processed on a 1 ms polling timer through an extra same-thread queue | `Server.cpp:396` | Up to 1 ms added latency, 1,000 wakeups/s | Process on receive completion (N1) |
| D17 | M | SPSC queue reloads the other side's index on every operation; `tail_` shares a line with data | `SpscQueue.h` | Cross-core cache misses per message | Cached indices, padding (N1) |
| D18 | M | Stale queue entries survive stop → rebind; send size limits disagree (1,400 accepted, 1,391/1,375 enforced silently); `reuse_address` on UDP lets another process share the port on Windows | `Server.cpp`, `NetSocket.cpp:35` | Use of stale buffers; silent drops; port hijack | Clear on bind; one size limit with an error; exclusive bind (N0.5) |

### 3.2 Memory today (5,000 connections, measured from the code)

| Structure | Size | Use |
|---|---|---|
| Packet pool 8,192 × 1,400 B | 11.5 MB | One size for everything (a 30-byte input takes 1,400 B), yet too few buffers for the load |
| Retransmit slots 5,000 × 2 × 128 × 12 B | 15.4 MB | ~96% idle (typically 2–5 in flight per channel) |
| Reorder buffers 5,000 × 64 × 8 B | 2.6 MB | Almost always empty |
| Game → network queue 8,192 × 1,424 B | 11.7 MB | ~95% of staging bytes unused |
| Handler table 65,536 × `std::function` | 4.2 MB | ~99.7% unused |
| Send list 65,536 × 24 B | 1.6 MB | |
| Context buffers | 1.3 MB | Unreachable through `Server` |
| Endpoint map (32,768 capacity) | 0.9 MB | 4× over-provisioned |
| Hot/cold connection arrays, queues, misc | 0.8 MB | |
| **Total** | **≈ 50 MB** | Plan document claims 35 MB |

About 45 MB of the 50 MB is worst-case sizing that the new design replaces with
shared pools (§5A). The `Server` object itself is ~50 MB, which overflows the
stack if constructed as a local variable.

### 3.3 Test gaps

Not tested today: encrypted end-to-end sessions, nonce uniqueness and replay,
disconnect/timeout/slot reuse/reconnect, stop → rebind, pool-leak accounting,
the ack-behind-32 case, full retransmit rings, sequence wrap on the reliable
unordered channel, full queues, malformed input/fuzzing, connect floods,
more than 8 clients, multi-threaded send, the Linux `sendmmsg` path, 100%
loss. Each defect fix in N0.5 comes with a test that reproduces it first.

The design notes in `engine/net/NET_LIBRARY_PLAN.md` and
`NET_LIBRARY_TODO.md` have drifted from the code (memory figures, "one copy",
"one syscall per flush", ticked items not implemented and vice versa); they
are superseded by this document.

## 4. Layers

```
 Game (zone server / client)
        │
 ┌──────▼──────────────────────────────────────────────┐
 │ L4  Replication   entities, snapshots, interest,     │  §6
 │                   priority, deltas, RPCs             │
 ├──────────────────────────────────────────────────────┤
 │ L3  Streams       unreliable, ordered streams,        │  §4.3–4.4
 │                   fragmentation, bulk transfer        │
 ├──────────────────────────────────────────────────────┤
 │ L2  Session       handshake, tokens, encryption,      │  §4.2
 │                   reliability, RTT, timeouts          │
 ├──────────────────────────────────────────────────────┤
 │ L1  Transport     sockets, batched I/O, threads,      │  §5
 │                   packet pool                         │
 └──────────────────────────────────────────────────────┘
        │  UDP
```

Each layer is usable and testable on its own. Server-to-server links (zone
borders, §7) use L1–L3 with a trusted-network profile.

### 4.1 Wire format

- Packet header: connection id (for NAT rebinding, §4.2), sequence, ack +
  ack bits, flags. Encrypted packets carry a nonce and 16-byte tag.
- Messages inside a packet are **bit-packed** (§6.3) at L4; L3 frames are
  byte-aligned for simplicity.
- A **protocol version** is exchanged in the handshake; mismatches are
  rejected with a clear error (client must update).
- MTU: 1,200-byte payloads by default (safe across the internet including
  IPv6 and tunnels); configurable.

### 4.2 Session and security

- **Connect tokens** (netcode.io-style): the gateway (after Steam login)
  issues a short-lived signed token containing account id, zone address,
  expiry and session keys. The zone server accepts only valid tokens — it
  never trusts a raw UDP hello.
- **Anti-spoofing**: the server replies to a new connect request with a
  stateless **challenge cookie**; state is allocated only after the client
  echoes it. Stops spoofed-IP floods from filling connection slots, and the
  reply is never larger than the request (no amplification).
- **Encryption always on** for client traffic: ChaCha20-Poly1305 with
  per-session keys from the token (libsodium, already a dependency). Replay
  protection via sequence window.
- **Connection id in every packet**: survives NAT rebinding and switching
  Wi-Fi ↔ mobile data (connection migration).
- **Adaptive retransmit**: smoothed RTT and variance per connection (the
  standard SRTT/RTTVAR method), with a minimum and maximum; ack-driven fast
  resend.
- **Keepalive and timeouts**: configurable; disconnect after N seconds of
  silence; clean disconnect message.
- **Rate limits** per connection (packets/s, bytes/s, messages per type) —
  exceeding them drops input and flags the account.

### 4.3 Streams (replacing the 3 fixed channels)

| Stream kind | Guarantee | Used for |
|---|---|---|
| Unreliable | none | Input, snapshots (replication handles loss) |
| Unreliable sequenced | newest wins, older dropped | Voice/aim updates if ever needed |
| Reliable ordered ×N | ordered **within** a stream | Inventory, trades, chat, block edits — each on its own stream so a lost chat packet doesn't delay a trade (no head-of-line blocking across streams) |
| Bulk | reliable, ordered, low priority, fragmented | Chunk data, large UI data |

### 4.4 Fragmentation and bulk transfer

- Messages larger than one packet are split into fragments with
  (message id, fragment index, count); reassembled with a timeout and a
  per-connection memory cap.
- **Bulk stream** sends chunk data only with bandwidth left over after
  snapshots and reliable game messages (§6.5), so entering a new area never
  makes combat lag.
- Chunks are sent **palette-compressed + run-length** (game design §26) and
  can be resumed if a connection hiccups.

## 5. Transport and threading (L1)

### 5.1 Production target

- **Zone servers run on Linux.** It has batched UDP syscalls
  (`recvmmsg`/`sendmmsg`), io_uring, `SO_REUSEPORT` socket sharding, and it's
  the cheapest server OS to host. Windows server support stays for local
  development.
- Clients: Windows, Linux (Steam Deck), macOS.

### 5.2 Threads

```
  NIC ─▶ socket 0 ─▶ net thread 0 ─┐
  NIC ─▶ socket 1 ─▶ net thread 1 ─┼─▶ per-thread queues ─▶ simulation workers
  NIC ─▶ socket K ─▶ net thread K ─┘                         (by sector)
```

- **K network threads** (e.g. 4–8), each with its own UDP socket on the same
  port (`SO_REUSEPORT`); the kernel spreads clients across sockets by
  address hash, so each connection is always handled by the same thread —
  no locks on connection state.
- Each network thread: receive batch → decrypt → session/reliability →
  push messages to the simulation; and pull outgoing packets → encrypt →
  send batch.
- Encryption and decryption happen on network threads, spread across cores.
- Queues between threads stay single-producer/single-consumer (existing
  `SpscQueue`), one pair per (network thread, simulation worker).
- Windows development build: same design with one socket per thread and
  `WSARecvFrom`/RIO; correctness first, speed second.

### 5.3 I/O backends

| Backend | Platform | Batch | Notes |
|---|---|---|---|
| `recvmmsg`/`sendmmsg` | Linux | 32–64 packets per call | First production backend |
| io_uring | Linux 5.19+ | submission batches, zero-copy send | Added if syscalls show up in profiles |
| RIO (Registered I/O) | Windows | request queues | For Windows-hosted servers, if ever needed |
| `WSASendTo`/`WSARecvFrom` | Windows | 1 | Dev fallback |

Asio can remain for timers and the Windows path; the Linux hot path uses the
system calls directly.

## 5A. Memory design

### 5A.1 Rules

1. **Allocate once, at startup.** Every pool and arena is sized from config
   when the server starts. After that, the steady state performs **zero heap
   allocations** (enforced by a test that hooks the allocator, §8).
2. **Size for the typical case, share for the worst case.** No per-connection
   worst-case arrays (e.g. 128 retransmit slots × every connection). Per
   connection keeps only small indices; the storage comes from shared pools
   that absorb bursts.
3. **Recycle in O(1), locally.** Free lists are **per thread** (no atomics).
   Freed items go to the front (LIFO), so the next allocation reuses memory
   that is still in the CPU cache. Items that cross threads are returned in
   **batches** through the existing SPSC queues, not one atomic operation per
   item.
4. **Rings recycle for free.** Anything with a natural lifetime order
   (acked snapshots, per-tick scratch) lives in ring buffers or per-tick bump
   arenas, where "freeing" is moving an index.
5. **Pass descriptors, not payloads.** Queues carry 8–16-byte descriptors
   (buffer index, length, connection), never full MTU-sized copies.
6. **Every pool is accounted.** Each pool reports size, in use, and
   high-water mark; exhausting a pool is a handled condition (§6A.3), never a
   crash.

### 5A.2 Allocators

| Allocator | Used for | Alloc/free cost | Notes |
|---|---|---|---|
| **Slab pool** (fixed-size blocks, per thread) | Packet buffers, fragments, retransmit records | a few instructions, no atomics | LIFO free list stored inside the free blocks (no side array) |
| **Batched return queue** | Blocks freed on a thread other than the owner | one SPSC push per 64 blocks | Owner thread splices the batch back into its free list |
| **Per-tick bump arena** | Snapshot building scratch, parsing temporaries, relevancy scratch | pointer increment; free = reset at tick end | One per simulation/network worker |
| **Ring buffers** | Entity state history, per-client ack windows, logs | index move | Fixed capacity, oldest overwritten |
| **Slot map with generation** | Connections, fragments in reassembly, bulk transfers | O(1) | A reused slot gets a new generation, so stale references (late packets from an old connection) are rejected safely |

All arenas are allocated with **huge pages** where available (Linux
`madvise(MADV_HUGEPAGE)`) to reduce TLB misses, and **NUMA-local** to the
thread that uses them.

### 5A.3 Packet buffers

- Block size **1,280 bytes** (1,200-byte payload + headers + encryption tag),
  64-byte aligned.
- Each network thread owns its pool (e.g. 8,192 blocks = 10 MB per thread).
- Receive: the batch receive call fills pool blocks directly (no copy); the
  block travels to the simulation as a descriptor and comes back through the
  batched return queue.
- Send: game code writes messages **directly into packet blocks** through a
  per-connection write cursor (no intermediate `OutboundMsg` staging copy);
  full blocks are queued for the network thread by descriptor.
- A block that holds reliable data is **reference-counted** (one count for the
  send, one per retransmit record) and freed when the last reference drops —
  no second copy is kept for retransmission.

### 5A.4 Reliability state without worst-case arrays

- **Retransmit records** (16 bytes: block index, sequence, sent time,
  connection) come from a shared slab pool and are linked per connection;
  a connection with nothing in flight uses **zero** records.
- **Reorder buffers** are allocated from the pool only when a packet arrives
  out of order (rare on healthy connections) and returned when the gap
  fills.
- **Fragment reassembly** uses slot-map entries with a per-connection cap and
  a timeout; memory is bounded per connection and globally.

### 5A.5 Replication memory — the biggest item

A naive design keeps, per client, a copy of every relevant entity's last
acknowledged state: 10,000 clients × 250 entities × 64 bytes × 32 snapshots ≈
**5.1 GB**. Instead:

- **Shared entity history**: each entity keeps one ring of its last 32 ticks
  of quantised state, shared by all clients:
  40,000 entities (players + monsters + items) × 32 × 64 B ≈ **82 MB**.
- **Per client**, only the tick each relevant entity was last acknowledged
  at: 250 × 4 B = **1 KB per client** (10 MB at 10,000).
- Delta encoding reads the baseline from the shared history using that tick.
  If the baseline is older than the ring (client lagged too long), the entity
  is sent in full — correct, just larger.

### 5A.6 Memory budget (10,000 players, Proposed)

| Item | Size | Total |
|---|---|---|
| Packet blocks: 8 network threads × 16,384 × 1,280 B | — | 168 MB |
| Connection hot state (sequence, acks, RTT, cursors) | ~128 B each | 1.3 MB |
| Connection cold state (address, keys, stats, context) | ~512 B each | 5.1 MB |
| Retransmit records (shared, ~8 in flight avg, 4× headroom) | 16 B each | 5.1 MB |
| Per-client relevancy set + ack ticks + priority | ~3 KB each | 31 MB |
| Shared entity history (40,000 entities) | — | 82 MB |
| Per-tick arenas (32 workers × 4 MB) | — | 128 MB |
| **Total** | | **≈ 420 MB** |

Every number is a config value; the server prints the full budget at
startup and refuses to start if it exceeds the configured memory limit.
For comparison, today's library uses ~50 MB for 5,000 connections without
any replication data (§3.2), most of it idle.

## 6. Replication (L4) — the part that makes 10,000 possible

### 6.1 Model

- The server owns the truth. Each replicated entity has an id, a type, and a
  set of **replicated fields** declared once (position, yaw, animation,
  health, equipment, …), each with a **quantisation rule** (§6.3).
- Each client has a **relevancy set**: the entities it currently knows
  about.

### 6.2 Interest management

- Uses the engine's **spatial grid** (already used for culling): each tick,
  a client's relevant entities are the ones in cells within its view radius.
- Relevance rings:

| Ring | Distance (Proposed) | Update rate |
|---|---|---|
| Near | 0–32 blocks | every snapshot (20 Hz) |
| Mid | 32–96 blocks | every 2nd snapshot |
| Far | 96–160 blocks | every 4th snapshot, reduced fields |
| Out | > 160 blocks | removed from the relevancy set |

- **Hotspot cap** (Decided: 250): if more than N entities are in range, the
  client gets the N most important (priority, §6.4); the rest are shown as
  simplified crowd or hidden. Keeps bandwidth bounded when 2,000 players
  stand at the Grand Exchange.
- Relevancy is updated incrementally as entities change grid cells, not
  rebuilt from scratch.

### 6.3 Encoding

- **Bit-packing**: a bit writer/reader replaces byte-aligned
  `ActionBuilder` for replication data.
- **Quantisation**: positions relative to the client's reference point
  (sector origin) in 1/64-block units; yaw in 8–10 bits; velocities and
  health in the minimum bits needed.
- **Delta against the last acknowledged snapshot**: only changed fields are
  sent, with a per-entity changed-field mask. An entity standing still costs
  a few bits.
- **Baselines** per client: the server keeps the last N acknowledged states
  per relevant entity in a ring buffer (memory sized per client × relevancy
  cap).

### 6.4 Priority

- Each (client, entity) pair has a **priority accumulator**: it grows each
  tick by a weight (distance, whether it's attacking the player, party
  member, recent change) and resets when sent.
- Each snapshot packs the highest-priority entities until the packet budget
  is full. Important things (the enemy hitting you) are always current;
  distant background players update less often.

### 6.5 Bandwidth budget

- Per client: a byte budget per snapshot (e.g. 3 KB at 20 Hz = 60 KB/s,
  configurable), adapted down when loss or RTT rises (simple congestion
  control).
- Order within the budget: reliable game messages → snapshot (by priority)
  → bulk chunk data.
- Per server: a global egress cap; when near the cap, far rings drop to
  lower rates first.

### 6.6 Client side

- **Prediction** for the local player using the shared movement code;
  **reconciliation** when the server state arrives (rewind + replay inputs).
- **Interpolation** for other entities with a ~100 ms buffer; extrapolation
  for short gaps.
- **Input**: client sends inputs with tick numbers, redundantly (last 3
  inputs per packet) so one lost packet doesn't lose an input.

### 6.7 Server-side combat support

- **Lag compensation**: the server keeps a short history (e.g. 250 ms) of
  hitbox positions per entity and checks hits against what the attacker saw
  (their tick + interpolation delay), capped to limit abuse.
- Hits, damage and drops are server events sent on reliable streams.

### 6.8 RPCs

- Typed messages for actions that aren't state (use ability, open GE, trade
  offer), generated from one schema so client and server can't disagree on
  the format.

### 6.9 Schema: X-macro tables — Decided

Every message and replicated entity type is declared once as a table of
fields; macros generate everything else:

```cpp
// messages/UseAbility.def
#define ATM_MSG_FIELDS(F)                                   \
  F(uint8_t,  slot,      bits(3))                            \
  F(uint32_t, target_id, bits(20))                           \
  F(Vec3q,    aim,       quantized(1.0f / 64, 1024))
ATM_DEFINE_MESSAGE(UseAbility, /*id*/ 42, ATM_MSG_FIELDS)
#undef ATM_MSG_FIELDS
```

Generated from that table:

- the plain struct (`UseAbility { uint8_t slot; ... }`),
- bit-packed `encode()` / `decode()` with range checks on decode,
- for entity types, the per-field **changed mask** used by delta encoding,
- a **compile-time schema hash** (constexpr) over every message's name,
  fields, types and bit sizes. Client and server exchange it at handshake and
  refuse to talk on mismatch.

Why X-macros rather than an IDL + generator: no extra build tool, errors are
ordinary compiler errors at the table, and every consumer (client, zone
server, bots, services) is C++. If non-C++ tools need the definitions later,
a small generator can read the same tables.

## 6A. Reliability design

### 6A.1 Guarantees

- **No crash on any input.** Every parser validates lengths and ranges
  before reading; untrusted data never indexes memory directly. Enforced by
  fuzzing every parser (§8).
- **No leaks.** Every pool block has exactly one owner at any time; debug
  builds track owners and assert at shutdown that every pool is back to
  full.
- **No stale data.** Connection slots and other reused slots carry a
  generation; packets or references with an old generation are dropped.
- **Correct sequence arithmetic** across wraparound (tested exhaustively
  around the wrap point).
- **Bounded everything.** Queues, pools, reassembly, reliable backlog per
  connection — all have caps, and hitting a cap has a defined behaviour.

### 6A.2 Threading correctness

- Each connection is owned by exactly one network thread (socket sharding),
  so its state is never touched by two threads.
- Cross-thread communication only through SPSC queues with one documented
  producer and consumer each; debug builds assert the calling thread.
- Memory ordering reviewed per queue/pool; ThreadSanitizer runs on the test
  suite and the load test.

### 6A.3 Overload behaviour (degrade, never fail)

| Condition | Behaviour |
|---|---|
| Packet pool nearly empty | Stop reading new bulk data; drop lowest-priority snapshot content first |
| Outbound queue for a connection full | Coalesce: newer snapshot replaces older unsent one; reliable messages keep priority |
| Reliable backlog per connection over cap | Disconnect that connection with a reason (slow/stalled client), never grow memory |
| Tick overrun | Far relevancy rings drop to lower rates; metrics raise an alert |
| Connect flood | Challenge cookies; token checks before any allocation |

### 6A.4 Recovery

- Client reconnect with the same session token within a grace period
  restores the session without relogging.
- Zone server crash: players reconnect through the gateway to the restarted
  zone (game design §23).
- Watchdog per network thread: a stalled thread is reported with its last
  activity.

## 7. Server-to-server and services

- **Zone ↔ zone** (border ghosts and handoff, game design §23): L1–L3 over
  the data-centre network, encryption optional, larger MTU allowed, same
  replication encoding for ghost entities.
- **Handoff protocol**: old zone freezes the player's input at tick T, sends
  full state; new zone acknowledges and takes ownership at T+1; the client is
  told to switch zone address using a pre-issued token (no re-login). Target:
  no visible hitch.
- **Zone ↔ global services** (GE, auctions, ledger, chat): request/response
  over TCP or QUIC (gRPC-style), not the game UDP protocol — these are
  transactional, not real-time.
- **Gateway**: Steam login → account → issues connect tokens for the right
  zone.

## 8. Testing and tools

| Tool | Purpose |
|---|---|
| Unit tests (existing + new) | Each layer in isolation: fragmentation, reliability under loss, replay protection, bit-packing, delta encode/decode round-trips |
| NetSim (existing) | Loss, latency, jitter, reordering, duplication, bandwidth caps |
| **Bot client** | Headless client that logs in, moves, fights and builds with scripted behaviour; thousands per process |
| **Load test** | 10,000 bots (and 12,000 for headroom) against one zone server, bots on separate machines; hotspot scenario (2,000 bots in one area) |
| Metrics | Per tick: packets/bytes in/out, encryption time, snapshot build time, relevancy size, priority starvation, RTT/loss distribution, queue depths |
| Replay capture | Record a session's packets for debugging and regression tests |
| Fuzzing | Malformed packets against every parser (must never crash); runs continuously during development |
| Sanitizers | AddressSanitizer + UndefinedBehaviorSanitizer + ThreadSanitizer builds of the tests and load test |
| Allocation hook | Fails the test if any heap allocation happens in steady state |
| Pool accounting | Asserts all pools are full again after shutdown (no leaked blocks) |
| Soak test | 24 h with bots joining/leaving, loss and latency injection; memory must be flat |

**Gates** (on the reference server: 16–32 cores, 10 Gb/s, Linux):

| Scenario | Target |
|---|---|
| 10,000 bots spread across the zone | Network + replication < 50% of tick budget at 30 Hz; p99 tick < 33 ms |
| 2,000 bots in one hotspot | Every client within its bandwidth budget; near entities updated every snapshot |
| 12,000 bots (20% headroom) | Tick stable at 30 Hz with far-ring degradation, no crash |
| 5% loss, 150 ms RTT, 30 ms jitter | Movement smooth (no visible corrections > 0.25 block) |
| Connect flood (spoofed) | No slot exhaustion; legitimate clients still connect |
| 24 h soak | Zero crashes, zero leaked blocks, memory flat, zero steady-state allocations |
| Memory | Within the §5A.6 budget at 10,000 players |

## 9. Milestones

| # | Deliverable | Acceptance |
|---|---|---|
| N0 | Measure current AttomeNet: echo server + 1,000 / 5,000 loopback clients (its current maximum), profile, memory report | Baseline numbers recorded |
| N0.5 | **Fix the defects** D2 (timeout/kick part), D3–D8, D11, D18, each with a reproducing test first | All reproducing tests pass; sanitizer builds clean; pools full after shutdown |
| N1 | Transport + memory redesign for 10,000 connections: multi-socket `SO_REUSEPORT` threads, `recvmmsg`/`sendmmsg`, per-thread slab pools, direct-write send path, shared retransmit/reorder pools, dirty lists + timer wheel (D9, D10, D13–D17) | 600k packets/s in + out on the reference server (300k each way); memory within §5A.6 budget |
| N2 | Session: connect tokens, challenge cookie, always-on encryption with per-direction keys and 64-bit nonces, replay window, authenticated control packets, connection id, adaptive RTO, rate limits (D1, D2, D12) | Security tests + fuzzing clean; nonce-uniqueness test |
| N3 | Streams: N ordered streams, fragmentation, bulk stream | Chunk transfer while under loss doesn't delay snapshots |
| N4 | Replication core: field schema, bit-packing, quantisation, deltas vs acked baselines | Round-trip tests; bytes per idle entity < 1 byte |
| N5 | Interest + priority + bandwidth budget | Hotspot test within budget |
| N6 | Client prediction/reconciliation/interpolation + lag compensation | Smooth play at 150 ms RTT, 5% loss |
| N7 | Bot client + load-test harness | 10,000-bot gate passed |
| N8 | Zone-to-zone ghosts + handoff; gateway tokens | Walk across a border with no hitch; 12,000-bot headroom test |

These run alongside the engine milestones: N0–N3 don't depend on the
renderer; N4–N6 need the voxel prototype's simulation (engine plan M7–M9).

## 10. Risks

| Risk | Mitigation |
|---|---|
| Bandwidth cost at 10,000 players | Aggressive deltas, priority, relevancy caps; measure bytes per player per scenario from N5; 25 Gb/s NICs |
| Hotspots (towns, world bosses) | Relevancy cap + crowd simplification; zone splitting for persistent crowds |
| Simulation CPU at 10,000 | Sector-parallel simulation workers; network work fully off the simulation threads; profile from N7 |
| Encryption CPU | Spread across network threads; hardware-accelerated where available |
| DDoS on own hardware | Challenge cookies + tokens handle protocol floods; volumetric attacks need an upstream scrubbing service (§11.2) |
| Linux-only production I/O | Keep the Windows dev path working; run the test suite on both |
| Scope creep in replication | X-macro schema keeps each new message/entity type cheap |

## 11. Hosting — own hardware first, cloud later

Decided: run zone servers on **own hardware** first; move to (or add) cloud
later. The design stays provider-neutral so the move doesn't change code.

### 11.1 Reference zone server (for 10,000 players)

| Part | Spec (Proposed) | Why |
|---|---|---|
| CPU | 32 cores / 64 threads, high clock (e.g. AMD EPYC or Threadripper) | ~8 network threads, ~16–24 simulation/replication workers, headroom |
| RAM | 64 GB ECC | Network ≈ 0.4 GB (§5A.6); world chunks, simulation, OS headroom |
| NIC | 25 Gb/s, multi-queue with RSS | 4.8 Gb/s peak egress with headroom; RSS spreads packets across cores |
| Storage | NVMe SSD (RAID 1) | Chunk store write-ahead log, fast restarts |
| OS | Linux (current LTS), tuned: socket buffers, IRQ affinity, huge pages | §5 transport depends on Linux features |
| Uplink | ≥ 10 Gb/s per server at the data centre, 95th-percentile billing checked | Bandwidth is the largest running cost |

### 11.2 Around the servers

- **DDoS protection**: own hardware still needs an upstream scrubbing
  service (colocation provider or a dedicated UDP-capable DDoS provider).
  The protocol defences (§4.2) stop protocol abuse; they can't stop a link
  being flooded.
- **Gateway and global services** (login, GE, auctions, ledger, chat) on
  separate machines from zone servers, with the database on its own host
  with replication and backups.
- **Deployment**: each zone server is one process per machine, started by
  a supervisor with health checks; configuration in the same JSON config
  system as the engine.

### 11.3 Cloud later

- No cloud-specific services in the game path: plain Linux, UDP, our own
  gateway and services. Moving means running the same binaries on cloud
  instances with enough network bandwidth (check per-instance packet-rate
  and bandwidth limits, which are often lower than the headline number).
- The bot load-test harness (§8) is how each new hosting option is
  qualified before players move to it.

## 12. Decisions log

| Question | Decision |
|---|---|
| Player target | **10,000 per zone server from the start** |
| Hosting | **Own hardware first**, cloud later (§11) |
| Message schema | **X-macro tables** (§6.9) |
| Tick rates | **30 Hz simulation, 20 Hz snapshots**, 30 Hz client input (all configurable) |
| Relevancy cap | **250 entities per client** in hotspots (configurable) |

No open questions remain for the network plan; new ones are added here as
they come up.
