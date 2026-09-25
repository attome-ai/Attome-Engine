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
  tiers and ability upgrades (details in §9.3).
- **Armour** sets add passive effects (e.g. lifesteal, speed, glide time),
  letting players build hybrid styles.
- **Dodge** with a dash; enemy attacks are telegraphed (ground markers,
  wind-ups).
- **Crowd rule:** hundreds of players in one fight must stay readable — other
  players' effects are drawn dimmer than your own (Proposed).

## 9. Skills (RuneScape-style) — Decided; list Proposed

No classes. Everyone can train every skill, by doing it, and what you are is
simply what you've trained. **There is no level or XP cap (Decided)**: 99 is
a milestone, not the end (§9.4).

### 9.1 Which RuneScape skills we keep, and why

A skill is kept only if it passes all three checks:

1. **Fun to do** in a Trove-style action/voxel world (not a single repeated
   click).
2. **Makes something other players need** (feeds the economy or group
   content).
3. **Has a reason to reach 99** beyond the number: unlocks, gear, areas.

| RuneScape skill | Here | Why |
|---|---|---|
| Attack, Strength, Defence, Ranged, Magic, Hitpoints | **Keep** (renamed only where it helps) | Core of combat progression |
| Prayer | **Keep** | Combat buffs; gives bones/remains value |
| Mining, Woodcutting, Fishing | **Keep** | Breaking ore/tree blocks and fishing fit the voxel world |
| Hunter | **Keep** | Trapping and tracking creatures in the open world |
| Farming | **Keep** | Works on owned plots — gives land a use |
| Smithing, Crafting, Fletching, Cooking, Herblore | **Keep** | Every combat style needs their products |
| Construction | **Keep, becomes central** | Voxel building on your land (§17.3) |
| Agility | **Keep, reworked** | Unlocks movement upgrades (extra jump, glide time, dash) |
| Slayer | **Keep** | Task-based combat with unique monsters and drops |
| Firemaking | **Drop** | No product; fires become part of Cooking |
| Runecrafting | **Drop** | Long repetitive loop; Magic uses crafted catalysts instead (§9.3) |
| Thieving | **Drop** | Doesn't fit action combat; pickpocket loot moves to Hunter/Slayer |
| — | **New: Taming** | Mounts and pets, very Trove |

**Total: 21 skills.** The list is a proposal; any skill can be added or
removed before content production starts.

### 9.2 The skills

| Skill | Train by | Makes / unlocks | Level 99 reward |
|---|---|---|---|
| **Attack** | Melee combat | Melee weapon tiers; weapon abilities per tier; crit chance | Top-tier melee weapons |
| **Strength** | Melee combat | Melee damage; heavy weapons (hammers, greatswords) | Max melee damage |
| **Defence** | Taking hits in any style | Armour tiers; damage reduction | Best armour sets |
| **Ranged** | Bows, crossbows, thrown | Ranged weapon tiers and abilities; damage | Top ranged weapons |
| **Magic** | Staves, wands, spells | Spell tiers, teleports, utility spells | Top spells and staves |
| **Hitpoints** | Any combat | Max health | — |
| **Prayer** | Offering bones/remains at altars | Temporary combat buffs (protection, damage, healing); prayer points drain in use | Strongest prayers |
| **Mining** | Breaking ore and gem blocks | Ores, gems, stone blocks | Rarest ores, fastest pickaxes |
| **Woodcutting** | Chopping tree blocks | Logs, building wood | Magic trees, best axes |
| **Fishing** | Fishing spots, nets, spears | Fish for Cooking; rare catches | Deep-sea fishing |
| **Hunter** | Traps, tracking, catching creatures | Hides, feathers, rare creature parts, pet eggs | Rarest creatures |
| **Farming** | Planting on owned plots (and farming patches in towns) | Herbs, crops, trees, seeds | Rare seeds, fastest growth |
| **Smithing** | Anvils: ore → bars → weapons/armour | Metal weapons, armour, tools, construction parts | Best metal gear |
| **Crafting** | Leather, gems, cloth, jewellery, magic catalysts | Light armour, jewellery, Magic catalysts, glass blocks | Best jewellery, top catalysts |
| **Fletching** | Logs + materials → bows, arrows, staves | Ranged weapons and ammunition; staff bases | Best bows and ammo |
| **Cooking** | Fish/meat/crops on fires or ranges | Food (healing), buff meals | Best food |
| **Herblore** | Herbs + ingredients → potions | Potions (boosts, antidotes, stamina) | Best potions |
| **Construction** | Building on your plot | Block sets, furniture, crafting stations, portals, shop stalls, guild halls | Best blocks and plot features |
| **Agility** | Obstacle courses, parkour routes in the world | Movement upgrades: extra jumps, longer glide, faster dash, stamina | Best movement kit |
| **Slayer** | Tasks from Slayer masters ("kill 120 frost wolves") | Access to Slayer-only monsters and their drops | Hardest Slayer monsters |
| **Taming** | Befriending creatures, raising pets, training mounts | Mounts (speed, flying in allowed zones), combat pets | Rarest mounts |

### 9.3 How skills fit Trove-style action combat

In RuneScape, levels decide almost everything in combat. Here the player's
own aim and dodging matter too, so levels are designed to **unlock and
scale**, not to make fights automatic:

- **Attack / Ranged / Magic** level unlocks **weapon tiers**; each tier gives
  better abilities (the Q/E/R abilities from §8), not just bigger numbers.
- **Strength** and the style's level scale **damage**; **Defence** scales
  **damage reduction**; **Hitpoints** scales **max health**.
- **Hit chance comes from aim**, not from the Attack level. Attack instead
  adds **critical-hit chance** and unlocks weapons.
- **Magic** uses **catalysts** (made with Crafting from gems and ores)
  instead of runes, which is why Runecrafting isn't needed.
- **Prayer** gives timed buffs with a cooldown, so it's an active choice in
  fights rather than a toggle.

### 9.4 Experience and levels

- **The RuneScape experience curve**: the XP needed for level *L* is

  `XP(L) = floor( (1/4) × Σ from x=1 to L−1 of floor(x + 300 × 2^(x/7)) )`

  giving **13,034,431 XP for level 99** — half of it is earned between
  levels 92 and 99, the classic RuneScape feel.
- **No XP or level cap** (Decided). XP is stored as a 64-bit integer.
- **After level 99 the curve becomes linear** (Proposed): every level past 99
  costs the same XP as the 98 → 99 step, **1,228,825 XP**.

  | Level | RuneScape curve continued | This game (linear after 99) |
  |---|---|---|
  | 99 | 13,034,431 | 13,034,431 |
  | 120 | 104,273,167 | 38,839,756 |
  | 150 | 2,033,749,558 | 75,704,506 |
  | 200 | 287,416,243,706 | 137,145,756 |

  Continuing the RuneScape formula would make level 150 cost 2 billion XP
  and level 200 cost 287 billion, so progress would silently stop around
  120–130. A fixed cost per level keeps every level reachable and every
  hour of training visible, forever.
- **XP rates** per activity live in the JSON tunables, so they can be
  balanced without a client update.
- **XP drops**: numbers float up on screen as XP is earned (toggleable).
- **Level-up**: fireworks effect visible to nearby players, message with
  what the new level unlocked.

### 9.4.1 What levels above 99 give (Proposed)

Unlimited levels must not mean unlimited power, or veterans would make
combat content and the Wilderness unwinnable for everyone else:

- **Unlocks stop at 99** for content access (areas, weapon tiers, recipes),
  so 99 is always enough to use everything.
- **Stat bonuses continue with diminishing returns and a ceiling**: each
  level past 99 adds a small bonus to the skill's effect (damage, gather
  speed, crafting success) that shrinks per level and approaches a
  **hard ceiling of +10%** over level 99. A level-300 player is noticeably
  better than a level-99 one, never overwhelmingly.
- **Prestige keeps growing without limit**: mastery ranks every 10 levels
  past 99 (cape colours and trims, titles, auras, emotes), hiscores by total
  XP, and rare cosmetic unlocks at milestone levels (150, 200, 250, …).
- Values for bonus size and ceiling live in the JSON tunables.

### 9.5 Combat level

A single number showing overall combat strength, used for matching and for
the Wilderness (§14). Same shape as Old School RuneScape's formula:

```
base   = 0.25 × (Defence + Hitpoints + floor(Prayer / 2))
melee  = 0.325 × (Attack + Strength)
ranged = 0.325 × floor(1.5 × Ranged)
magic  = 0.325 × floor(1.5 × Magic)
combat = floor(base + max(melee, ranged, magic))
```

Hitpoints starts at level 10 (as in RuneScape). The formula uses levels
**capped at 99**, so combat level ranges from 3 to 126 and stays a fair
matching number; levels above 99 show separately as **mastery** on the
profile. In the Wilderness, players can attack each other only
within a combat-level range that widens the deeper they go.

### 9.6 Rewards for mastery

| Reward | How |
|---|---|
| **Skill cape** | Reaching 99 in a skill; gives a small perk (e.g. Cooking cape: never burn food) and an emote |
| **Trimmed cape** | Any cape once the player has two or more 99s |
| **Max cape** | 99 in every skill; combines all cape perks |
| **Mastery ranks** | Every 10 levels past 99 in a skill: new cape colour/trim, title; no end |
| **Skill pets** | Very rare drop while training a skill (e.g. a rock golem while mining) |
| **Hiscores** | Rank per skill (by XP, unlimited), total level, total XP; per world and global |
| **Guilds** | Areas unlocked by level (Mining guild at 60, Cooking guild at 32, …) with better resources |

### 9.7 Temporary boosts

Potions (Herblore), meals (Cooking) and some items give **temporary level
boosts** (e.g. +5 Mining for 5 minutes), letting players reach a content
requirement early — the same trick RuneScape uses to reward
cross-skilling.

### 9.8 How skills feed each other and the economy

```
 Mining ─▶ Smithing ─▶ metal weapons/armour ──┐
 Woodcutting ─▶ Fletching ─▶ bows, arrows ────┤
 Hunter ─▶ Crafting ─▶ leather, jewellery, ───┼──▶ combat (all players)
            catalysts (for Magic)             │
 Fishing/Farming ─▶ Cooking ─▶ food ──────────┤
 Farming ─▶ Herblore ─▶ potions ──────────────┘
 Mining/Woodcutting/Smithing ─▶ Construction ─▶ builds on plots
 Slayer/combat ─▶ rare drops, bones (Prayer), hides (Crafting)
```

Every combat player consumes what skillers make (food, potions, ammo,
catalysts, repairs), so pure skillers have a real place in the economy —
the RuneScape feel where "no gameplay is forced" still works.

### 9.9 Skills in the voxel world

- **Gathering is breaking blocks**: ore blocks, tree blocks, and fishing
  spots in water. In **reserved** areas they regrow on a timer so they're
  never used up.
- **On owned plots**, players can place **resource nodes** (trees, farming
  patches, an ore vein) through Construction/Farming, limited per plot so
  land doesn't become an infinite resource farm.
- **In the Wilderness**, resource nodes are richer (rare ores, magic trees)
  but you can be attacked while gathering — risk vs reward, as in
  RuneScape.

## 10. Progression and gear

- **Gear tiers** crafted or dropped, requiring skill levels to use.
- **Item rarities**: common → uncommon → rare → epic → legendary. Rarities
  change stat ranges and unlock an extra effect.
- **Unique items**: legendary drops have a serial number and owner history
  (visible when inspected) — this helps trading trust and audits.
- **Cosmetics**: voxel skins for weapons, armour, mounts, gliders; earned in
  game or bought (see monetisation §21).

### 10.1 Player character model — modular, one shared rig (Decided)

Every player character is built from the **same skeleton ("rig")**, so every
animation works for every player, and clothes, armour and weapons are pieces
that snap onto that rig. Changing equipment is swapping a piece, never
building a new model.

**The rig** (one for all players):

```
root
 └─ pelvis
     ├─ torso
     │   ├─ head
     │   ├─ arm_upper_L ─ arm_lower_L ─ hand_L ─ [off-hand socket]
     │   ├─ arm_upper_R ─ arm_lower_R ─ hand_R ─ [main-hand socket]
     │   └─ [back socket: cape / glider / quiver]
     ├─ leg_upper_L ─ leg_lower_L ─ foot_L
     └─ leg_upper_R ─ leg_lower_R ─ foot_R
```

- Each bone is a **rigid voxel part** (Trove-style: limbs move as whole
  pieces, no bending), so animation is just one transform per part.
- Fixed proportions and pivot points, published as a **MagicaVoxel template**
  that every artist models against.

**Equipment slots** and what they replace or attach to:

| Slot | Replaces / attaches | Notes |
|---|---|---|
| Head | Replaces `head` covering (helmet, hat, hood) | Can hide hair; face shows through where the design allows |
| Torso | Replaces `torso` and `arm_upper` parts | Chest armour, shirts, robes |
| Hands | Replaces `arm_lower` + `hand` parts | Gloves, gauntlets |
| Legs | Replaces `leg_upper` + `leg_lower` | Trousers, leg armour, skirts |
| Feet | Replaces `foot` parts | Boots |
| Back | Attaches to back socket | Capes (incl. skill capes), gliders, quivers |
| Main hand / off hand | Attaches to hand sockets | Weapons, shields, tools (pickaxe, axe, rod) |
| Cosmetic overrides | Same slots, drawn instead of the real item | Wear any look over your actual gear |

- **Body customisation**: skin colour, hair, face, and body shape variants
  (e.g. slimmer/bulkier torso pieces) — all built on the same rig and bone
  lengths, so animations never need changing.
- **Dyes**: each item uses a small colour palette; dye slots swap palette
  entries in the shader, so recolours cost nothing to store or draw.

**Shared animations**:

- One animation set for all players: idle, run, jump, double jump, glide,
  swim, climb, dodge, hit, death, emotes, and skilling actions (mine, chop,
  fish, cook, smith, build).
- **Weapon-type layers**: each weapon type (sword, bow, staff, hammer,
  daggers, …) has its own upper-body attack/ability animations, played on top
  of the lower-body movement, so you can run and attack at the same time.
- A new weapon or armour **needs no new animation** — only a new item of an
  existing weapon type needs art.

**Why this matters technically**:

| Concern | Result |
|---|---|
| Swapping gear | Change one mesh id per slot; instant, no rebuild, no loading hitch |
| Art cost | One rig, one animation set; each item is a small voxel piece |
| Network | A player's appearance is ~24 bytes (item id per slot + dyes + body options), sent only when it changes |
| Rendering 250 visible players | Every body part and item piece is drawn **instanced**: all players wearing the same helmet share one draw; per-player data is just part transforms + palette |
| Animation cost | One transform per part (~16 parts) per visible player per frame — tiny |

Monsters and NPCs use the same system with their own rigs (e.g. a
quadruped rig), so humanoid NPCs can wear any player equipment too.

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
| **3. Vertical slice** | 1 town + GE (basic) + 5 skills (Attack, Strength, Mining, Smithing, Construction) + 3 weapons + damage-share loot + inventory/trading + persistence | Complete loop in a small world |
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
3. **Skill list** (§9.1) — 21 skills proposed (Firemaking, Runecrafting and
   Thieving dropped; Taming added). Add or remove any?
4. **Plot size and per-account limit** (§17) — plot = 1 sector (256×256)?
5. **Flying mounts** — allowed where?
6. **Game name**.
