# Game design — voxel sandbox MMO (working title: *Attome Online*)

Status: design draft, revision 1. Companion documents:
- `docs/GPU_3D_PLAN.md` — engine and Vulkan renderer plan.
- This document — the game: concept, systems, economy, world, servers,
  roadmap.

Markers used below:
- **Decided** — confirmed by the project owner.
- **Proposed** — a recommendation with its reasoning; accept or change it.

---

## Part A — Concept

## 1. One-line pitch

A Trove-style voxel action MMO with RuneScape's freedom and economy, set in
one huge persistent world that grows every day, where players win land at
auction and build anything on it.

## 2. What we take from each game

| From | What | Status |
|---|---|---|
| Trove | Voxel art style: chunky, colourful, glowing effects, bloom | Decided |
| Trove | Movement: fast WASD, jumping, double jump, gliding, mounts | Decided |
| Trove | Action combat: aim with the mouse, abilities on hotkeys, dodge | Decided |
| Trove / Minecraft | Free block building | Decided |
| RuneScape | No classes: play any style, switch any time | Decided |
| RuneScape | No forced path: skill, fight, trade, build, or explore — your choice | Decided |
| RuneScape | Player trading and a PvP Wilderness | Decided |
| RuneScape | Grand Exchange (global buy/sell order market) | Decided |
| RuneScape, changed | Drops shared between everyone who fought, by damage dealt | Decided |

## 3. What is unique

1. **Absolute open world that grows.** One continuous map. New land is added
   at the edges every day.
2. **Land auctions.** Each day's new land is auctioned; the highest bid (in
   in-game gold) wins ownership. **Decided.**
3. **Player-owned space inside the world.** The map mixes reserved land
   (game content: towns, dungeons, biomes, bosses) and player land (plots
   anyone can win and build on).
4. **Build freely** on owned land; the wilderness is editable by everyone.
   **Decided.**
5. **One map, many servers.** The map is split into areas, each run by its own
   server holding up to **10,000 players**; players cross between areas
   seamlessly. **Decided.**

## 4. Design pillars

Every feature is checked against these:

1. **Freedom** — no class, no required path, no gated playstyle.
2. **Ownership** — what you earn, build or win is yours and persists.
3. **A real economy** — prices are set by players; items and gold must be
   earned, never duplicated.
4. **Readable action** — combat is fast but always clear: strong silhouettes,
   bright telegraphs, low visual noise.
5. **A world that changes** — growth, building, and the wilderness make the
   map different every week.

## 5. Target audience and platform

- **Platform:** Steam — Windows first, Steam Deck / Linux native, macOS
  optional. **Decided** (see engine plan).
- **Audience:** players of Trove, RuneScape, Minecraft servers, and sandbox
  MMOs; sessions from 20 minutes (skilling, trading) to hours (building,
  raids).
- **Input:** mouse + keyboard primary; full controller support (required for
  Steam Deck Verified).

---

## Part B — Player experience

## 6. Core loops

```
          ┌────────────── explore ──────────────┐
          ▼                                      │
   fight / gather ──▶ loot, resources, XP ──▶ craft, trade (GE)
          ▲                                      │
          │                                      ▼
   stronger gear ◀──── gold ◀──── sell ──── build, bid on land
```

- **Minute to minute:** move, fight, gather, loot.
- **Session:** finish a dungeon, level a skill, sell on the GE, build a room.
- **Long term:** max skills, own and develop land, rare items, reputation,
  guild territory.

## 7. Movement (Trove-style) — Decided

| Ability | Behaviour |
|---|---|
| Run | WASD relative to camera, high acceleration, snappy stop |
| Jump | Variable height (hold longer = higher); extra jumps from gear/skills |
| Glide | Hold jump in the air: slow fall, forward drift; wings/gliders are items |
| Sprint/dash | Stamina-based short dash with brief invulnerability (dodge) |
| Mounts | Faster ground travel; flying mounts only in some zones and not in the Wilderness (Proposed) |
| Swim | Free 3D movement in water, slower |
| Climb | Auto step-up of 1 block; ladders/vines for climbing |

Camera: third-person, follows behind the player, mouse controls look
direction; zoom in/out; a lock-on option for controller players.

All movement code is **shared between client and server** (same C++,
deterministic fixed step) so the client can predict and the server can
verify (§25).

## 8. Combat (Trove-style action) — Decided

- **Aim** with the mouse (or aim-assist on controller). Attacks travel as
  projectiles, beams or melee arcs; hits are resolved on the server.
- **Abilities come from the weapon, not a class** (Proposed, this is how "no
  classes" works with action combat):
  - Each weapon type (sword, bow, staff, hammer, daggers, …) gives
    - a **primary attack** (left mouse),
    - a **secondary** (right mouse),
    - **two abilities** (Q, E) and
    - an **ultimate** (R), charged by dealing damage.
  - Switching weapons changes your playstyle instantly. Two weapons can be
    held in quick slots.
- **Skill level** in the matching combat skill raises damage, unlocks weapon
  tiers and ability upgrades.
- **Armour** sets add passive effects (e.g. lifesteal, speed, glide time),
  letting players build hybrid styles.
- **Dodge** with a dash; enemy attacks are telegraphed (ground markers,
  wind-ups).
- **Crowd rule:** hundreds of players in one fight must stay readable — other
  players' effects are drawn dimmer than your own (Proposed).

## 9. No classes: skills — Decided (list Proposed)

Every skill levels by doing it, 1–99 (RuneScape curve), and anyone can train
all of them.

| Group | Skills |
|---|---|
| Combat | Melee, Ranged, Magic, Defence, Vitality (health) |
| Gathering | Mining, Woodcutting, Fishing, Foraging, Hunting |
| Production | Smithing, Crafting (armour/jewellery), Fletching, Cooking, Alchemy (potions), Enchanting |
| World | Building (blocks, structures), Farming, Agility (movement upgrades), Taming (mounts, pets) |

- **Total level** is shown on the profile and hiscores.
- Some content needs levels (e.g. a boss needs 70 combat skills), never a
  specific class.

## 10. Progression and gear

- **Gear tiers** crafted or dropped, requiring skill levels to use.
- **Item rarities**: common → uncommon → rare → epic → legendary. Rarities
  change stat ranges and unlock an extra effect.
- **Unique items**: legendary drops have a serial number and owner history
  (visible when inspected) — this helps trading trust and audits.
- **Cosmetics**: voxel skins for weapons, armour, mounts, gliders; earned in
  game or bought (see monetisation §21).

## 11. Drops shared by damage — Decided (rules Proposed)

Everyone who damaged a monster gets loot based on how much they contributed.

Rules:

1. **Eligibility**: a player must deal at least **5%** of the monster's
   health in damage (prevents one-hit tagging) and be within range when it
   dies.
2. **Common drops** (resources, coins, low-tier gear): each eligible player
   gets their **own roll**, with quantity scaled by their damage share.
3. **Rare drops**: **one roll per kill**; if it hits, the winner is chosen at
   random **weighted by damage share**.
   - Why: if every player rolled for rares, a crowd of 50 would create 50× more
     rares and crash their value. One roll per kill keeps rares rare while
     still rewarding everyone.
4. Loot is **personal**: each player sees only their own drops, so there is no
   loot stealing.
5. **Healing and tanking count as contribution** (Proposed): healing done to
   eligible players and damage absorbed count at a reduced weight, so support
   play is rewarded.

## 12. Trading

- **Direct trade**: two-sided trade window with a confirm screen that shows
  total value (GE prices) to help avoid scams.
- **Grand Exchange** (§13).
- **Player shops** on owned land (Proposed): stalls that sell at set prices
  while the owner is offline.
- All trades are **server transactions**: both sides change together or not
  at all (§27).

## 13. Grand Exchange — Decided

- **Global**: one market shared by **all servers** of the map.
- Players place **buy** and **sell** orders with a price and quantity; the
  exchange matches them automatically (highest buy meets lowest sell;
  trade happens at the older order's price).
- Available from GE booths in every major town.
- **Fees**: a small sales tax (e.g. 1–2%) removed from the game — the main
  gold sink (Proposed).
- **Price history** and graphs per item.
- **Buy limits** per item per 4 hours to stop market cornering (Proposed).
- Untradeable items: quest rewards, some cosmetics.

## 14. Wilderness and PvP — Decided (rules Proposed)

- **Where**: dedicated Wilderness zones on the map, clearly marked; the rest
  of the world is PvE only.
- **Danger levels**: deeper Wilderness = better resources/bosses = harsher
  death rules.
- **Death in the Wilderness** (Proposed):
  - Outer levels: keep 3 most valuable items, drop the rest.
  - Deep Wilderness: keep 1 item.
  - A **skull** (for attacking unprovoked) means keeping 0 items.
- **Death outside the Wilderness**: keep everything, respawn at the last
  checkpoint.
- **Wilderness is editable by everyone** (Decided): players can dig, build
  forts, walls, traps. See §17 for how this stays under control.

---

## Part C — World

## 15. World structure

- **One continuous voxel map**, divided into **sectors** of 256 × 256 blocks
  (full height).
- Each sector is one of:

| Type | Editable by | Examples |
|---|---|---|
| Reserved | Nobody (designers only) | Towns, dungeons, quest areas, boss arenas, roads |
| Plot | Owner + people they allow | Player land won at auction |
| Wilderness | Everyone | PvP zones |
| Wild (unclaimed) | Nobody until auctioned (Proposed) | Future plots, visible but locked |

- **Biomes** set block palettes, monsters and resources (plains, forest,
  desert, snow, volcanic, sky islands, ocean, …).
- **Vertical space**: 512 blocks high (Proposed): underground caves, surface,
  sky islands.
- **Units**: 1 block ≈ 1 metre; a player is about 2 blocks tall (Proposed).

## 16. A world that grows every day

- Every day at a fixed time (e.g. 00:00 UTC) a **new ring of sectors** is
  added at the map's edge.
- New sectors are **generated** from the biome rules and seed
  (procedural), then designers can add reserved content later.
- A share of new sectors becomes **plots** (put up for auction), the rest is
  reserved content or wilderness (Proposed mix: 60% plots, 30% reserved,
  10% wilderness; tunable).
- Growth rate is **tunable** so the map grows with the player count, not
  faster (a mostly empty world feels dead).

## 17. Land ownership

### 17.1 Auctions — Decided (in-game gold); rules Proposed

- Each day's new plots go to auction for **24 hours** before the growth
  event.
- Bids are **in-game gold** only.
- **Escrow**: the bid amount is held when placed; outbid players get their
  gold back immediately.
- **Losing bids** are refunded minus a **1% fee** (gold sink, discourages
  spam bids). Proposed.
- **Anti-sniping**: a bid in the last 10 minutes extends that plot's auction
  by 10 minutes.
- **Limits**: max plots per account (e.g. 3) to stop the rich from buying the
  whole map (Proposed).
- Reserve price per plot based on location (next to towns costs more).

### 17.2 Owning land

- **Upkeep tax** (Proposed): a weekly gold cost scaled by plot size and
  location. Unpaid for 4 weeks → plot is re-auctioned; buildings are saved as
  a **blueprint** the owner can place again later. This stops abandoned plots
  and removes gold from the economy.
- **Permissions**: owner, co-owners, builders, visitors; guild plots.
- **Plot features**: player shops, crafting stations, farms, spawn point,
  teleport pad (to town).
- **Selling land** to other players through a plot trade (with the same
  transaction safety as items).

### 17.3 Building

- Minecraft/Trove-style: place and break blocks, with a **build mode**
  (copy/paste, fill, mirror, blueprints) for large projects.
- Blocks are crafted or gathered (Building skill unlocks block sets).
- **Limits per plot**: block count and "special block" count (lights,
  machines) to protect server and client performance.

### 17.4 Wilderness editing — Decided (everyone can edit); control rules Proposed

Open editing needs limits, or the Wilderness becomes a griefing and storage
problem:

- **Natural regrowth**: terrain dug out in the Wilderness **slowly restores**
  to the original generated shape (e.g. over 24–72 hours) unless a player
  structure is on it.
- **Player structures decay**: player-placed blocks in the Wilderness lose
  durability over time and need maintenance (Proposed: resources).
- **Break speed** depends on block hardness and tools; forts can be sieged but
  not erased instantly.
- **Storage cap**: each Wilderness sector keeps at most N edited blocks; the
  oldest decaying ones go first.

## 18. Towns, content and quests

- **Towns**: GE, bank, crafting stations, quest givers, teleports.
- **Dungeons**: instanced or open, from solo to 50-player raids.
- **World bosses**: spawn on timers; damage-share loot (§11) makes them
  crowd events.
- **Quests** (RuneScape-style stories) unlock areas, abilities, and
  cosmetics.
- **Events**: seasonal events, wilderness wars between guilds (Proposed).

## 19. Social

- Friends list, private messages, chat channels (local, world, trade,
  guild).
- **Guilds**: shared bank, guild plots, guild wars in the Wilderness
  (Proposed).
- **Parties**: shared quest progress and dungeon entry.
- Steam friends and invites through Steamworks.

## 20. Economy

### 20.1 Sources (gold/items enter)

Monster drops, quests, selling to NPC shops (low prices), skilling
products.

### 20.2 Sinks (gold/items leave)

| Sink | Where |
|---|---|
| GE sales tax | §13 |
| Auction fee on losing bids | §17.1 |
| Land upkeep | §17.2 |
| Repair costs, teleports | Gear, travel |
| Consumables (potions, food, ammo) | Combat |
| Cosmetic crafting with gold | Optional |

The economy is tuned so sinks roughly match sources; we track gold per
player over time (§31) and adjust with tunables (the JSON tunables system
already in the engine).

### 20.3 Protection

- Items and gold only change through server transactions (§27).
- Duplication is the biggest threat to a player economy; every item
  movement is logged, and rare items have serial numbers.
- **No real-money trading** allowed (Proposed): it fuels bots, fraud and
  account theft; accounts caught doing it are banned. Check Steam's current
  rules on trading and marketplaces before launch.

## 21. Monetisation — Proposed, needs decision

Options (pick one before beta):

| Model | Pros | Cons |
|---|---|---|
| Buy-to-play + cosmetic store | Clear value, fewer bots (each costs money) | Smaller player base |
| Free-to-play + cosmetic store | Largest audience | More bots, needs strong anti-cheat |
| Free + optional membership (RuneScape model) | Proven for this genre | Designing two content tiers |

In every model: **no pay-to-win**, and land is bought only with in-game gold.

---

## Part D — Technical design (game-specific)

The engine and renderer are in `GPU_3D_PLAN.md`. This part covers what the
MMO adds on top.

## 22. Architecture overview

```
      Steam client (Windows / Deck / Linux / macOS)
        │  UDP (AttomeNet): movement, combat, world edits
        │  TLS/HTTPS: login, GE, auctions, chat (via gateway)
        ▼
   ┌──────────── Gateway (login, Steam auth, routing) ────────────┐
   │                                                              │
   ▼                                                              ▼
 Zone servers (one per map area, ≤10,000 players each)     Global services
 ┌────────┬────────┬────────┐                               ┌─────────────────┐
 │ Zone A │ Zone B │ Zone C │ ◀─── border sync ───▶         │ Grand Exchange  │
 └────────┴────────┴────────┘                               │ Land auctions   │
        │                                                   │ Chat / guilds   │
        ▼                                                   │ Economy ledger  │
   World storage (voxel chunks)                              └────────┬────────┘
                                                                     ▼
                                                        Database (accounts, items,
                                                        orders, land, logs)
```

- **Zone servers** run the simulation (movement, combat, monsters, block
  edits) for their part of the map. They reuse the engine's simulation code
  in a **headless build** (no renderer), which is possible because of the
  simulation/renderer separation in the engine plan.
- **Global services** handle things shared by the whole map: GE, auctions,
  chat, guilds, the item/gold ledger.
- **Database**: PostgreSQL (Proposed) for accounts, items, orders, land,
  logs. Voxel chunk data in a separate chunk store (files or object
  storage) with a write-ahead log.

## 23. One map, many servers (zones) — Decided

- The map's sectors are **assigned to zone servers**. Each zone is a
  rectangle of sectors; assignment can change as the map grows or when an
  area gets crowded (**zone split**).
- **Border band**: each zone also simulates a read-only copy ("ghost") of
  entities within ~64 blocks of its borders, received from the neighbour.
  Players near a border see and can fight across it.
- **Handoff**: when a player crosses a border, the old zone sends their full
  state to the new zone, which takes ownership. Target: invisible to the
  player (< 1 tick).
- **Cross-border combat**: a hit on a ghost is forwarded to the owning zone,
  which applies it. Adds one server-to-server hop of latency near borders
  (Proposed: acceptable; place borders away from towns and boss arenas when
  possible).
- **Zone failure**: if a zone server crashes, its area is restarted from the
  last persisted state; players in it reconnect to the new instance.

## 24. 10,000 players per zone — the hard numbers

This is achievable but is the single most demanding part of the project.
Estimates for planning:

| Resource | Estimate | Plan |
|---|---|---|
| Simulation CPU | 10k players + monsters at 20–30 Hz | Data-oriented (the engine's SoA containers), multi-threaded by sector; 16–32 core server |
| Downstream per player | 200 visible entities × 20 Hz × ~12 bytes ≈ 50 KB/s | Interest management + delta compression + lower rates for distant entities |
| Server egress | 10k × ~50 KB/s ≈ 500 MB/s (4 Gb/s) in a crowded zone | 10 Gb/s network per zone machine; crowd caps |
| Crowd hotspots (GE, bosses) | 2,000+ players in one spot | Visible-entity cap per client (nearest N at full rate, the rest simplified or hidden), cosmetic effects client-side only |

Interest management uses the engine's existing **spatial grid**: each client
receives updates only for entities in nearby cells, at a rate based on
distance (near = every tick, far = every 3rd tick).

## 25. Networking

Built on **AttomeNet** (already in the engine: UDP, reliability layer,
batching, encryption).

- **Own player**: client-side prediction with the shared movement code;
  server reconciliation (server sends authoritative state, client rewinds
  and replays inputs).
- **Other entities**: interpolation between server snapshots (≈100 ms
  buffer).
- **Hit detection**: server-side with **lag compensation** (the server
  rewinds positions to what the shooter saw, within a cap like 200 ms).
- **Channels**: unreliable for movement snapshots, reliable-ordered for
  inventory, trades, chat, block edits.
- **Snapshots**: delta-compressed against the last acknowledged snapshot;
  quantised positions and angles.
- **World data**: chunks sent compressed (palette + run-length) when entering
  view; block edits sent as small deltas.
- **Tick rates** (Decided): simulation 30 Hz, snapshots 20 Hz, client input
  30 Hz; all configurable (engine plan §10.4).
- **Network library**: AttomeNet 2, designed for 10,000 players per zone
  server — full plan in `docs/NETWORK_PLAN.md`.

## 26. Voxel world technology

- **Chunks**: 32 × 32 × 32 blocks; a sector is 8 × 8 chunks wide × 16 high.
- **Block storage**: palette-compressed per chunk (most chunks use few block
  types), so memory and network size stay small.
- **Block types**: data-driven (JSON tunables), with properties: hardness,
  transparency, light emission, collision, sound, drop.
- **Client meshing**: greedy meshing on worker threads (merges faces into
  large quads), per-vertex **ambient occlusion** (the soft corner shading
  typical of voxel games), and per-block light levels.
- **Lighting**: sunlight + block lights propagated per chunk (flood fill),
  plus a small number of dynamic lights for effects.
- **Draw distance**: full detail near the player, lower-detail meshes
  (downsampled chunks) far away.
- **Server**: authoritative block edits; validates reach, tool, permissions
  (plot owner / wilderness / reserved) and rate limits.
- **Persistence**: edited chunks are saved as deltas against the generated
  base, so untouched terrain costs no storage.

## 27. Economy safety (transactions)

- Every change to items or gold goes through the **ledger service** in one
  database transaction: trade, GE match, auction escrow, shop sale, drop,
  death drop.
- Items have unique IDs; stackables are tracked as quantities with
  transaction history.
- **Idempotent requests**: every transaction has a request ID so a retry
  after a timeout can't apply twice.
- **Audit logs** for all transfers; tools to trace and reverse fraud.
- Zone servers never write gold/items directly; they request the ledger.

## 28. Anti-cheat and moderation

- **Server authority** over movement (speed, jump height, glide), combat
  (cooldowns, range, line of sight) and building (reach, permissions).
- Rate limits on actions, chat and trades.
- Bot detection by behaviour patterns (later).
- Report system, chat filter, mute/ban tools.
- Kernel-level anti-cheat not planned (Proposed): server authority covers
  most MMO cheats, and it keeps Steam Deck / Linux support simple.

## 29. Content and art pipeline

| Asset | Tool | Engine format |
|---|---|---|
| Characters, monsters, props, weapons | **MagicaVoxel** (`.vox`) (Proposed) | `.atmmodel` voxel model: converted to meshes at build time |
| Animations | Voxel parts animated as rigid pieces (Trove-style: body parts move, no bending) | Keyframes in `.atmanim` |
| Blocks | Block definitions in JSON + small voxel/colour data | Block registry |
| Terrain/biomes | Procedural rules in JSON + designer-placed reserved content | Generator config + prefab structures |
| Sounds, music | Recorded, licensed, or AI-generated (e.g. ElevenLabs) — check licences | `.wav`/`.ogg` |
| UI | draw2d + fonts | — |

- **Rigid-part animation** matches Trove's look (limbs are separate voxel
  pieces) and is cheaper than skinning. The engine's model format still
  supports 2-bone skinning (engine plan §11) if needed later.
- Placeholder art can be generated by scripts (simple blocks, trees, mobs)
  until real art exists.

## 30. Steam integration

Login with Steam ID, overlay, friends and invites, achievements, Cloud saves
for client settings, Steam Input (controller), Steam Deck Verified
requirements (controller-complete UI, readable at 1280×800). Details in the
engine plan §14.1.

## 31. Telemetry and live operations

- Server metrics: players per zone, tick time, bandwidth, crashes.
- Economy metrics: gold created/destroyed per day, GE volumes, prices of key
  items, plot auction prices.
- Balance changes through JSON tunables, hot-reloaded on servers.
- Scheduled maintenance for the daily growth event and updates.

---

## Part E — Plan

## 32. Development roadmap

Each phase ends with a playable build and the performance gates from the
engine plan.

| Phase | Content | Exit criteria |
|---|---|---|
| **0. Engine** | Engine plan M1–M6: benchmarks, Vulkan renderer, 3D core | 3D scene on Vulkan, gates met |
| **1. Voxel prototype** (offline) | Chunks, meshing, lighting, building, Trove movement, 1 weapon, 1 monster | Fun to move, jump, build and fight alone |
| **2. Networked prototype** | One zone server, prediction/interpolation, 50 players, block edits over the network | 50 bots + testers smooth at 30 Hz |
| **3. Vertical slice** | 1 town + GE (basic) + 3 skills + 3 weapons + damage-share loot + inventory/trading + persistence | Complete loop in a small world |
| **4. Scale** | Multiple zones, handoff, border ghosts, interest management, load tests to 10,000 bots per zone | 10k bots at target tick time and bandwidth |
| **5. Land & growth** | Plots, auctions, upkeep, permissions, daily growth, wilderness rules | A week of simulated days without manual fixes |
| **6. Content alpha** | Skills to 99, weapons, dungeons, bosses, quests, guilds | Closed alpha on Steam (Playtest) |
| **7. Beta** | Economy tuning, anti-cheat, Steam features, performance on Deck | Open beta / Early Access |

## 33. Team and tools (to decide)

- Engine + renderer, server/networking, gameplay, tools — possibly one person
  with Claude Code at first; the scale phase (4) benefits from a dedicated
  server engineer.
- Art: voxel artist(s) with MagicaVoxel; sound through licensed libraries or
  generation tools.

## 34. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| 10k players per zone | Highest technical risk | Load-test with bots from phase 2; zone splitting; crowd caps |
| Item/gold duplication | Destroys the economy | Ledger service, transactions, idempotency, audits (§27) |
| Wilderness griefing / storage growth | Bad experience, cost | Regrowth, decay, caps (§17.4) |
| Empty world early | Players feel alone in a huge map | Growth rate tied to population; start the map small |
| Land hoarding by rich players | Blocks newcomers | Plot limits, upkeep tax, re-auction (§17) |
| Scope | Very large project | Phased roadmap; each phase is playable |
| Legal (auctions, trading) | Store/legal issues | Gold-only land, no real-money trading, Steam rules review |

## 35. Open questions

1. **Monetisation model** (§21).
2. **Death rules** in the Wilderness (§14) — accept the proposal?
3. **Skill list** (§9) — add/remove skills?
4. **Plot size and per-account limit** (§17) — plot = 1 sector (256×256)?
5. **Flying mounts** — allowed where?
6. **Game name**.
