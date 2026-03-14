# Tower Swarm — Game Design Document

> **Version:** 1.0 | **Date:** 2026-03-15
> **Engine:** ATMEngine (C++20 + SDL3 → WebAssembly) | **Platform:** Web (WASM), Mobile (Capacitor)

---

## Table of Contents

1. [Vision & Elevator Pitch](#1-vision--elevator-pitch)
2. [Core Gameplay Loop](#2-core-gameplay-loop)
3. [Game Flow & Screens](#3-game-flow--screens)
4. [Level System](#4-level-system)
5. [Characters (The Roster)](#5-characters-the-roster)
6. [Evolution System](#6-evolution-system)
7. [Merge Mechanic](#7-merge-mechanic)
8. [Enemy Bestiary](#8-enemy-bestiary)
9. [Shop Systems (3 Shops)](#9-shop-systems-3-shops)
10. [Upgrades & Relics](#10-upgrades--relics)
11. [Economy & Currency](#11-economy--currency)
12. [Meta Progression](#12-meta-progression)
13. [Biomes & Maps](#13-biomes--maps)
14. [Viral & Social Features](#14-viral--social-features)
15. [Technical Architecture](#15-technical-architecture)
16. [Build Phases (Implementation Order)](#16-build-phases-implementation-order)

---

## 1. Vision & Elevator Pitch

### Elevator Pitch
> "Tower defense — but your towers are living creatures that fight, evolve, and move on their own.
> Collect characters. Merge them. Watch a Tier-1 pebble-thrower grow into a galaxy-eating god.
> Clear infinite levels. Never the same run twice."

### Design Pillars

| Pillar | What it means |
|---|---|
| **Living Defense** | Creatures aren't placed and forgotten — they think, reposition, and grow |
| **Infinite Depth** | Every level has a next level. Every creature has a next form. No ceiling |
| **Discrete Wins** | Each level is completable — players feel progress, not just endurance |
| **Roster Identity** | Players build attachment to specific characters and care about their army |
| **Emergent Strategy** | Merge + evolve + position = depth without complexity gates |

### What Makes It Different

| vs. Standard Tower Defense | Tower Swarm |
|---|---|
| Towers are fixed in place | Creatures reposition based on threat AI |
| Towers never change form | Creatures evolve visually and statistically |
| No tower identity | Named characters with lore and unique abilities |
| Waves go forever or end | Infinite discrete levels — always a next challenge |
| One currency, simple shop | Three distinct shop layers for different timescales |

---

## 2. Core Gameplay Loop

### The Micro Loop (within a level)

```
Place creatures on the map
         ↓
Wave of enemies advances toward your base
         ↓
Creatures auto-attack, reposition, and evolve
         ↓
Player: place new seeds, drag creatures, use wave buffs
         ↓
All enemies dead → Wave Clear
         ↓
Wave Buff Shop: pick 1 of 3 cards (temporary power spike)
         ↓
Next wave begins
         ↓
[repeat N waves]
         ↓
Final boss wave → Level Complete or Level Failed
```

### The Macro Loop (across levels)

```
Complete Level N
         ↓
Inter-Level Screen: see results, stars, spend essence in shop
         ↓
Creatures carry over (roster grows stronger across levels)
         ↓
Start Level N+1 (new map layout, harder enemies)
         ↓
[repeat to Level ∞]
```

### The Meta Loop (between sessions)

```
Earn Shards from level completions + achievements
         ↓
Visit the Armory: unlock new characters, buy passive masteries
         ↓
Return to level select with a permanently stronger roster
```

### The Three Tensions

These are the decisions that make the game interesting:

1. **Place now vs. save essence** — spend to defend this wave or hoard for a better seed next level
2. **Evolve naturally vs. merge for speed** — let creatures grind kills or merge two to skip tiers
3. **Hold position vs. reposition** — moving = not attacking; timing the repositioning matters

---

## 3. Game Flow & Screens

```
┌──────────────────────────────────────────────────────────┐
│  MAIN MENU                                               │
│  [Play]  [Armory]  [Leaderboard]  [Daily Level]          │
│  [Settings]  [Profile]                                   │
└──────────────────────────────────────────────────────────┘
         ↓ Play
┌──────────────────────────────────────────────────────────┐
│  LEVEL SELECT                                            │
│  Visual grid of levels 1 → highest_unlocked             │
│  Each tile: number, star rating, biome color, lock state │
│  [Continue] button → jumps to highest unlocked           │
└──────────────────────────────────────────────────────────┘
         ↓ Select Level N
┌──────────────────────────────────────────────────────────┐
│  PRE-LEVEL SCREEN                                        │
│  Level N — Biome name — "X waves — Boss: [type]"        │
│  Your Roster: grid of all owned creatures + tier badges  │
│  Essence balance shown                                   │
│  [Deploy] → starts the level                            │
└──────────────────────────────────────────────────────────┘
         ↓ Deploy
┌──────────────────────────────────────────────────────────┐
│  GAMEPLAY                                                │
│  HUD: Level N | Wave X/Y | Essence | Base HP | ★★★ bars │
│                                                          │
│  [WAVE N starts]                                         │
│    Enemies spawn → creatures defend → player manages     │
│  [WAVE N clear]                                          │
│    WAVE BUFF SHOP: pick 1 of 3 cards (5 sec timer)      │
│  [repeat]                                                │
│  [BOSS WAVE — last wave of level]                        │
└──────────────────────────────────────────────────────────┘
         ↓ Level ends
┌──────────────────────────────────────────────────────────┐
│  INTER-LEVEL SCREEN                                      │
│  Results: ★ rating, enemies killed, essence earned,      │
│           base HP %, time, evolutions this level         │
│                                                          │
│  [Bazaar tab]  [Forge tab]  [Relics tab]  [Repair tab]  │
│                                                          │
│  Roster strip at bottom: all creatures + their progress  │
│                                                          │
│  [Next Level →]  [Replay]  [Level Select]               │
└──────────────────────────────────────────────────────────┘
         ↓ Next Level
         (loop)
```

### Screen Inventory

| Screen | Purpose |
|---|---|
| Main Menu | Navigation hub |
| Level Select | Browse + replay levels, see star progress |
| Pre-Level | Review roster before deploying |
| Gameplay | Core game |
| Wave Buff Shop | Mid-wave temporary power pick |
| Inter-Level Screen | Results + full shop |
| Armory | Meta hub: unlock characters, buy masteries, cosmetics |
| Leaderboard | Daily + all-time ranking |
| Profile | Stats, achievements, season progress |
| Settings | Audio, graphics, controls |

---

## 4. Level System

### Level Structure

```
Level N
 ├─ wave_count     = 5 + floor(N × 0.5)          // L1=5  L10=10  L20=15  L50=30
 ├─ difficulty     = 1.18 ^ N                     // exponential base multiplier
 ├─ map_variant    = N mod 5                      // cycles 5 map templates
 ├─ biome          = floor(N / 10) mod 5          // biome changes every 10 levels
 ├─ is_elite       = (N mod 5 == 0)               // every 5th level is elite
 └─ boss_type      = biome-specific boss character
```

### Per-Wave Scaling (within a level)

```
wave_enemy_count(N, W) = floor((3 + N×1.8 + 0.05×N²) × (1 + W×0.15))
wave_enemy_hp(N, W)    = difficulty(N) × (1 + W×0.10)
wave_enemy_speed(N, W) = min(difficulty(N)^0.4 × (1 + W×0.04),  3.0)
```

### Win / Fail / Stars

| Condition | Result |
|---|---|
| Survive all waves | Level Complete |
| Base HP = 0 | Level Failed — Retry or Level Select |
| Base HP > 70% at end | ⭐⭐⭐ 3 Stars |
| Base HP 30–70% at end | ⭐⭐ 2 Stars |
| Base HP 1–29% at end | ⭐ 1 Star |

Star threshold lines shown on the base HP bar during play.

### Creature Persistence Rules

| Item | Persists? | Notes |
|---|---|---|
| Creatures (roster) | ✅ Yes | Carry over every level |
| Creature kills / tier | ✅ Yes | Cumulative across all levels |
| Essence | ✅ Yes | Fully carried over |
| Base HP | ❌ No | Resets to 100 each level start |
| Enemies | ❌ No | Cleared on level transition |
| Wave buff cards | ❌ No | Expire at level end |
| Creature map positions | ❌ No | Player re-places each level |

### Level Milestone Events

| Level | Event |
|---|---|
| 1 | Tutorial overlay active |
| 5 | First elite level — enemies glow red, +50% HP/speed |
| 10 | Biome 2 unlocks. Splasher character available in shop |
| 15 | First time Charger character available |
| 25 | Mid-game boss: The Siege Lord (variant 2) |
| 50 | Biome 5 (Void) preview moment — screen glitches briefly |
| 100 | Endgame badge unlocked. Leaderboard Hall of Fame entry |
| 200+ | Mastery tier — each level tagged "Mastery N" |

---

## 5. Characters (The Roster)

Characters are the creatures in your army. Each has:
- **Name** and lore blurb
- **Role** (combat archetype)
- **Rarity** (Common / Rare / Epic / Legendary)
- **3 Evolution stages** (name changes, visual changes, new/upgraded ability)
- **Unique Signature Ability** (unlocked at Tier 5)
- **5 Upgrade nodes** (buyable in the Forge shop)

Characters are unlocked permanently via the Armory (using Shards), then placed as seeds during play.

---

### Common Characters

#### BRIX — The Stone Golem
> *"Formed from the rubble of fallen walls, Brix remembers every battle."*

- **Role:** Shooter (single-target ranged DPS)
- **Rarity:** Common
- **Unlock:** Available from Level 1

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Brix | Small round stone, throws pebbles | — |
| 4–6 | Rockshot | Larger, angular jaw, shoots stones | Shots pierce 1 enemy |
| 7–9 | Stoneclaw | Towering golem, arm-cannon | Shots pierce 3, +20% range |
| 10+ | **Titan Lord** | Massive, glowing stone colossus | Shots detonate on impact (mini-splash) |

**Signature (Tier 5+):** *Avalanche* — every 15 seconds, fires a massive boulder that knocks back all enemies in a 150px line

---

#### FLARA — The Fire Sprite
> *"She doesn't aim. She just burns everything in the way."*

- **Role:** Splasher (area-of-effect damage)
- **Rarity:** Common
- **Unlock:** Available from Level 1

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Flara | Tiny flame wisp, lobs fireballs | — |
| 4–6 | Emberkin | Floating fire orb, wider blasts | Burning ground: 2s lingering AoE |
| 7–9 | Blazeling | Swirling inferno body | Burning ground 4s + slows enemies |
| 10+ | **Inferno God** | Volcanic pillar with erupting crown | 3 simultaneous blast targets |

**Signature (Tier 5+):** *Conflagration* — every 20 seconds, erupts into a 300px firestorm dealing 5× normal damage

---

#### MOSSLING — The Living Root
> *"It doesn't attack. It makes sure everything around it does — better."*

- **Role:** Support (aura buffs)
- **Rarity:** Common
- **Unlock:** Available from Level 1

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Mossling | Small green bulb on legs | +5% attack speed aura (96px radius) |
| 4–6 | Verdant | Leafy shrub with glowing core | +10% attack speed + +8% damage aura |
| 7–9 | Grovekeeper | Ancient tree trunk, golden leaves | Aura now heals nearby creatures 2 HP/s |
| 10+ | **World Root** | Massive glowing tree, roots spread | Aura radius 200px, also slows nearby enemies |

**Signature (Tier 5+):** *Overgrowth* — every 25 seconds, pulses a wave that instantly resets all nearby creatures' attack cooldowns

---

### Rare Characters

#### GLITCH — The Fractured Signal
> *"It wasn't created. It escaped."*

- **Role:** Trapper (crowd control, area denial)
- **Rarity:** Rare
- **Unlock:** Level 6 (Armory — costs 80 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Glitch | Flickering pixel cube | Drops slow-field orbs (50% speed for 3s) |
| 4–6 | Nether Pulse | Corrupted data stream | Orbs also reduce enemy damage output |
| 7–9 | Signal Rend | Warped code entity | Orbs detonate after 4s, dealing burst damage |
| 10+ | **Void Matrix** | Pure digital void fragment | Orbs chain to adjacent enemies on detonate |

**Signature (Tier 5+):** *System Crash* — every 18 seconds, freezes all enemies in 250px range for 2.5 seconds

---

#### IRONJAW — The Metal Beast
> *"It doesn't wait for enemies to come to it."*

- **Role:** Charger (melee rush, interceptor)
- **Rarity:** Rare
- **Unlock:** Level 10 (Armory — costs 120 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Ironjaw | Mechanical dog-like creature | Charges toward nearest enemy, knockback |
| 4–6 | Ruststorm | Spinning blade form | Charge hits up to 3 enemies in a line |
| 7–9 | Iron Colossus | Giant bipedal mech | Charge leaves shockwave trail |
| 10+ | **Steel Leviathan** | Massive serpentine war machine | Charge becomes rampage — hits all enemies in path |

**Signature (Tier 5+):** *Override* — every 22 seconds, enters a 4-second frenzy: 3× attack speed, unlimited movement

---

#### WRAITH — The Hollow Archer
> *"By the time they see the shot, it's already over."*

- **Role:** Sniper (extreme range, high single-target)
- **Rarity:** Rare
- **Unlock:** Level 15 (Armory — costs 150 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Wraith | Dark hooded figure, bow | Very long range, slow fire rate |
| 4–6 | Darkshot | Shadow archer, spectral arrows | Arrows ignore 30% armor |
| 7–9 | Phantom Arbiter | Cloaked specter, twin bows | Instakill enemies below 15% HP |
| 10+ | **Reaper** | Death entity with void scythe | Each kill chains a bolt to nearest enemy |

**Signature (Tier 5+):** *Death Mark* — every 30 seconds, marks one enemy — it dies after 4 seconds regardless of HP

---

### Epic Characters

#### CRYSTALIS — The Resonance Core
> *"It doesn't know if it's a weapon or a temple. Maybe both."*

- **Role:** Hybrid (shoots + amplifies nearby creatures)
- **Rarity:** Epic
- **Unlock:** Level 22 (Armory — costs 250 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Crystalis | Rotating crystal formation | Shoots piercing beams + aura boosts range |
| 4–6 | Prism Guard | Multi-faceted crystal tower | Beams refract to hit 2 targets |
| 7–9 | Resonance Core | Pulsing crystal matrix | Beams refract to hit 4 targets, aura boosts damage |
| 10+ | **Cosmic Array** | Orbital crystal ring | Beams bounce infinitely between enemies until they die |

**Signature (Tier 5+):** *Prismatic Nova* — every 20 seconds, fires a full-360° beam burst that hits all enemies on screen

---

#### VEX — The Chaos Shard
> *"Unpredictable. Unreliable. Unstoppable."*

- **Role:** Chaos (random powerful abilities, high variance)
- **Rarity:** Epic
- **Unlock:** Level 30 (Armory — costs 300 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–3 | Vex | Tiny crackling demon | Random ability fires every 5s: mini-explosion / teleport / slow pulse |
| 4–6 | Malice | Warped shadow creature | Random ability pool grows: + lightning strike / + clone for 3s |
| 7–9 | Dread Aura | Dark entity radiating chaos | Each random ability 2× stronger |
| 10+ | **Void Sovereign** | Reality-tearing elder chaos being | Abilities now chain — each one triggers the next |

**Signature (Tier 5+):** *Entropy Storm* — every 25 seconds, releases an uncontrolled wave of ALL random abilities simultaneously

---

### Legendary Characters

#### ORIN — The First Warden
> *"It predates the levels. It predates the swarm. It simply endures."*

- **Role:** Titan (ultimate all-rounder, passive reality warps)
- **Rarity:** Legendary
- **Unlock:** Complete Level 50 with 3 stars (Armory — costs 500 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–5 | Orin (Dormant) | Ancient stone column, faint glow | Passive: 5% chance to ignore damage taken by base |
| 6–9 | Orin (Awakened) | Radiant armored giant | Passive upgrades to 15%, also emits damage aura |
| 10+ | **Orin (Ascendant)** | Celestial being of pure light | Passive 25% base shield + resets 1 creature HP when it would die |

**Signature (Tier 5+):** *Temporal Ward* — every 60 seconds, all enemies on screen are frozen for 5 seconds while creatures continue attacking

---

#### NULL — The Void Seed
> *"It doesn't belong here. That's why it's perfect."*

- **Role:** Nullifier (suppresses enemies, drains their stats)
- **Rarity:** Legendary
- **Unlock:** Reach Level 100 (Armory — costs 800 Shards)

| Tier | Name | Visual | New Ability |
|---|---|---|---|
| 1–5 | Null (Seed) | Black pulsing void orb | Drains 10% damage from all enemies in range |
| 6–9 | Null (Expanding) | Growing void sphere | Drains 25% damage + 15% speed from enemies |
| 10+ | **Null (Complete)** | Reality-consuming black hole | Enemies in range deal 0 damage. Their kills credit Null. |

**Signature (Tier 5+):** *Consumption* — every 45 seconds, absorbs the nearest enemy entirely, gaining HP equal to its max HP

---

### Character Roster Summary

| Name | Role | Rarity | Unlock |
|---|---|---|---|
| Brix | Shooter | Common | Level 1 |
| Flara | Splasher | Common | Level 1 |
| Mossling | Support | Common | Level 1 |
| Glitch | Trapper | Rare | Level 6 |
| Ironjaw | Charger | Rare | Level 10 |
| Wraith | Sniper | Rare | Level 15 |
| Crystalis | Hybrid | Epic | Level 22 |
| Vex | Chaos | Epic | Level 30 |
| Orin | Titan | Legendary | Level 50 (3★) |
| Null | Nullifier | Legendary | Level 100 |

---

## 6. Evolution System

### How It Works

Each character evolves when it accumulates enough kills. Evolution is:
- **Permanent** (carries across levels)
- **Visual** (new sprite, size, color)
- **Mechanical** (stats scale + new/upgraded ability at thresholds)
- **Infinite** (no hard tier cap — Tier 30+ possible at extreme play)

### Kill Thresholds (Universal)

```
Tier 1 → 2:    10 kills
Tier 2 → 3:    30 kills
Tier 3 → 4:    80 kills
Tier 4 → 5:    200 kills
Tier N → N+1:  floor(10 × 2.5^(N-1))    // infinite
```

At Tier 10: requires 95,367 cumulative kills to reach. Achievable across many levels of play.

### Stat Scaling Per Tier

```
HP          = base_hp    × tier^1.4
Damage      = base_dmg   × tier^1.3
Range       = min(base_range × tier^0.5,  600px)
Attack rate = min(base_rate  × tier^0.4,  8/sec)
Move speed  = base_speed × tier^0.2        // diminishing — creatures don't become too fast
```

### Visual Evolution

| Tier Range | Size | Color Theme | Effect |
|---|---|---|---|
| 1–3 | 1.0× | Base color (white) | None |
| 4–6 | 1.3× | Green tint | Faint pulse |
| 7–9 | 1.6× | Blue glow | Slow orbit particles |
| 10–12 | 2.0× | Purple aura | Bright glow ring |
| 13–15 | 2.4× | Gold shimmer | Continuous particle trail |
| 16–19 | 2.8× | Red corona | Distortion ripple |
| 20+ | 3.0× (cap) | Void-black + white halo | Reality-tear VFX |

### Evolution Notification

When a creature evolves:
1. Creature flashes white → pulses to 1.5× size → settles at new tier size
2. Floating banner: **"[Name] → [New Stage Name] TIER N"** at creature position
3. Screen-edge glow for 1 second
4. Sound: tier-appropriate fanfare (Tier 1–3: soft chime | Tier 7+: epic sting | Tier 10+: cinematic)
5. Kills carry over (no reset) — creature is already partway to next tier

---

## 7. Merge Mechanic

Two creatures of the **same type AND same tier** that are adjacent on the grid can merge into a **Tier+1 creature**.

### Merge Rules

| Rule | Value |
|---|---|
| Requirements | Same type, same tier, adjacent cells (including diagonals) |
| Result | 1 creature at Tier+1 |
| Kill inheritance | sum of both creatures' kills ÷ 2 |
| Cooldown | 3 seconds after merge completes |
| Blocked during | Active attack, movement, evolution, already merging |

### Merge Flow

1. Two eligible creatures: pulsing amber link appears between them
2. Player can **drag one onto the other** OR wait 6 seconds for **auto-merge** (if both idle)
3. Merge animation: both creatures slide toward midpoint over 0.8 seconds → flash → new creature at midpoint
4. New creature inherits: `(kills_a + kills_b) / 2` — so it's already partway to the next threshold
5. Essence bonus: +10 essence per merge

### Why Merge vs. Grind

Merge is always faster to reach the next tier than grinding kills, but:
- Costs you 2 creatures → 1 creature (you lose field coverage temporarily)
- Requires two of the same type to be present (planning ahead)
- Best done during a between-wave grace period, not mid-combat

Strategic tension: do you merge now and fight with one fewer creature for a bit, or do you keep both and let them evolve naturally?

---

## 8. Enemy Bestiary

### Enemy Type Table

| # | Name | Role | HP | Speed | Special | Intro Level |
|---|---|---|---|---|---|---|
| 1 | **Grub** | Runner | Low | Fast | None — beelines for base | 1 |
| 2 | **Hulk** | Tank | Very High | Slow | Takes 50% reduced damage from front | 2 |
| 3 | **Scuttle** | Swarm | Very Low | Fast | Spawns in packs of 15–30 | 4 |
| 4 | **Driftwing** | Flyer | Medium | Medium | Ignores terrain, flies direct path | 7 |
| 5 | **Divide** | Splitter | Medium | Medium | On death: splits into 2 × 40% HP children | 11 |
| 6 | **Vanguard** | Shielded | High | Medium | Front shield: 80% damage resist (weak from sides/rear) | 16 |
| 7 | **Mender** | Healer | Low | Slow | Heals 8 HP/s to all allies within 120px | 22 |
| 8 | **Siege Lord** | Boss | 50× base | Slow | Multi-phase, spawns minions at 66% and 33% HP | Every level (last wave) |

### Enemy Scaling Per Level

All enemies scale with level number N:
```
HP     = base_hp    × (1.18^N) × wave_modifier
Speed  = base_speed × min((1.18^N)^0.4, 3.0)
Damage = base_dmg   × (1 + N × 0.05)
Reward = base_essence × (1 + N × 0.08)
```

### Elite Enemy Modifier (every 5th level)

On elite levels (L5, L10, L15...):
- All enemies: +50% HP, +20% speed
- Red glow tint on sprites
- HUD badge: **"ELITE LEVEL"** shown at start

### Boss: The Siege Lord

Each biome has a unique visual variant of the Siege Lord, but mechanics are shared:

| Phase | HP Threshold | Behavior |
|---|---|---|
| Phase 1 | 100% → 66% | Normal advance, stomps nearby creatures |
| Phase 2 | 66% → 33% | Spawns 20 Grubs, +30% speed |
| Phase 3 | 33% → 0% | Spawns 10 Hulks, charges toward base |

Boss death: large screen shake, confetti burst, essence reward = `150 + level × 8`

### New Enemy Introduction

First time a new enemy type appears in a level:
- Pause wave spawn for 2 seconds
- Zoom-in preview of the new enemy
- Banner: **"NEW ENEMY: [Name]"** + 1-sentence description
- Enemy enters from off-screen with dramatic animation

---

## 9. Shop Systems (3 Shops)

There are three distinct shops operating at different timescales.

---

### Shop 1 — The Wave Buff Shop (during gameplay)

**When:** Appears between every wave, during the grace period timer
**How:** Shows 3 random cards. Player picks 1. Free. Timer counts down.
**Duration:** Buffs last until end of current level.

**Card Pool (pick 1 of 3, randomly drawn):**

| Card | Effect |
|---|---|
| **Surge** | All creatures +25% attack speed for 4 waves |
| **Fortify** | Base HP +15 (one-time, this level only) |
| **Frenzied Blood** | Each kill grants +1 essence for 3 waves |
| **Slow Tide** | Next wave enemies move 35% slower |
| **Foresight** | Skip the next wave's boss/elite modifier |
| **Mend** | All creatures restore 50% of their max HP |
| **Wild Seed** | Instantly place a random Tier-2 creature on the map |
| **Echo Strike** | 20% of all projectile damage is repeated 0.3s later |
| **Essence Cache** | Immediately gain essence equal to 30% of current balance |
| **Iron Skin** | Creatures take 20% less damage for next 2 waves |
| **Apex Hunter** | The creature with most kills deals +50% damage this wave |
| **Void Pulse** | Every 10th enemy killed this wave explodes in a 80px blast |

---

### Shop 2 — The Inter-Level Shop (between levels)

**When:** After every level completes (win or loss after retry)
**How:** Tabbed interface, spend carried essence
**Reroll:** Costs 15 essence to refresh a tab's offerings (once per tab per visit)

#### Tab 1: The Bazaar (Buy New Characters)

Shows 4 random character seeds available for purchase. Rotation refreshes each level.

| Rarity | Seed Cost |
|---|---|
| Common seed | `20 + level × 2` essence |
| Rare seed | `60 + level × 4` essence |
| Epic seed | `150 + level × 6` essence |
| Legendary seed | `400 + level × 10` essence |

Buying a seed adds the character to your roster for use next level. If you already own that character, buying another gives you a duplicate seed (can be placed as a second unit, or discarded for 50% refund).

#### Tab 2: The Forge (Upgrade Characters)

Shows upgrade options for your currently owned characters.

Each character has 5 upgrade nodes. Each node has ranks I–V (except where noted):

| Node | Effect per Rank | Max Ranks |
|---|---|---|
| Strike | +15% damage | V |
| Vitality | +20% max HP | V |
| Reach | +10% attack range | III |
| Tempo | +8% attack speed | III |
| Signature | Enhances unique ability (see character sheets) | III |

Upgrade cost: `(current_rank + 1) × 15 × level_modifier` essence

#### Tab 3: The Relic Vault (Equip Passives)

You have **3 relic slots**. Relics provide global passive bonuses.

The vault shows all relics you've unlocked. You can swap relics between levels (not mid-level).

New relics are unlocked via the Armory (bought with Shards). See Section 10 for full relic list.

#### Tab 4: Repair

| Option | Cost |
|---|---|
| Restore 20 base HP for next level | 40 essence |
| Restore 50 base HP for next level | 90 essence |
| Full restore (100 HP) | 160 essence |

Note: base HP still resets to 100 each level regardless. Repair does not affect star rating of the level just completed — it helps you start the next level with full HP.

---

### Shop 3 — The Armory (Meta Hub)

**When:** Accessed from the main menu, between sessions
**Currency:** Shards (earned from first-time 3-stars, daily challenges, achievements)
**Persistent:** Purchases here are permanent across all runs

#### Armory Section 1: Character Collection

Shows all 10 characters in a gallery. Locked ones show their unlock requirement.

| Character | Shard Cost | Alternate Unlock |
|---|---|---|
| Brix | Free | — |
| Flara | Free | — |
| Mossling | Free | — |
| Glitch | 80 | — |
| Ironjaw | 120 | — |
| Wraith | 150 | — |
| Crystalis | 250 | — |
| Vex | 300 | — |
| Orin | 500 | Also: 3-star Level 50 |
| Null | 800 | Also: Reach Level 100 |

#### Armory Section 2: Passive Masteries

Global permanent upgrades bought with Shards. Each has 3 ranks.

| Mastery | Effect per Rank | Total Ranks | Cost (each rank) |
|---|---|---|---|
| **Echo Foundation** | +20 starting essence per level | III | 50 / 75 / 100 |
| **Nexus Vault** | Base starts with +10 HP per level | III | 60 / 90 / 120 |
| **Rapid Growth** | Evolution kill thresholds -5% | III | 80 / 110 / 150 |
| **Kinetic Swarm** | Creature move speed +5% | III | 40 / 60 / 80 |
| **Synthesis Mastery** | Merge cooldown -1 second | III | 70 / 100 / 140 |
| **Iron Resolve** | All creatures +5% HP per level above 20 | III | 100 / 150 / 200 |
| **Void Appetite** | Essence drops +8% | III | 45 / 65 / 90 |
| **Shard Eye** | +1 bonus Shard per first-time level completion | II | 120 / 200 |

#### Armory Section 3: Cosmetics

Visual only. No gameplay effect.

| Item | Cost |
|---|---|
| Alternate skin per character | 100–200 Shards each |
| Base / Nexus skin variants | 80 Shards each |
| Particle effect themes (fire / ice / void) | 60 Shards each |
| HUD color themes | 40 Shards each |

---

## 10. Upgrades & Relics

### Relic List (20 Relics)

Equipped in The Relic Vault (inter-level shop). 3 slots. Swap between levels.

| Relic | Effect | Unlocks |
|---|---|---|
| **Iron Core** | All creatures +10% max HP | Start unlocked |
| **Bloodshard** | +3% damage per tier level of the attacking creature | Start unlocked |
| **Essence Magnet** | +15% essence from all drops | Armory — 60 Shards |
| **Merger's Gift** | Merged creatures inherit 40% progress toward next evolution threshold | Armory — 80 Shards |
| **Warped Time** | Between-wave grace timer +3 seconds | Armory — 70 Shards |
| **Pack Instinct** | +8% attack speed per 3 same-type creatures on field | Armory — 90 Shards |
| **Eruption Core** | Splasher explosions leave burning ground for 3 seconds | Armory — 100 Shards |
| **Chain Strike** | Charger kills release a 60px shockwave | Armory — 100 Shards |
| **Void Lens** | Sniper creatures reveal enemy HP bars at 2× range | Armory — 80 Shards |
| **Living Wall** | Walls gain +20 HP per completed wave | Armory — 60 Shards |
| **Apex Hunger** | Creature with most kills deals +20% bonus damage | Armory — 110 Shards |
| **Twin Pulse** | Support aura radius +40px | Armory — 90 Shards |
| **Cold Bloom** | Trapper slow fields now also reduce enemy HP regen | Armory — 110 Shards |
| **Resonant Growth** | Crystalis aura boosts creature evolution rate +10% | Armory — 130 Shards |
| **Chaos Spark** | Vex's random ability pool has 1 additional option per level above 30 | Armory — 150 Shards |
| **Eternal Echo** | Once per level, the first time your base would reach 0 HP, it stays at 1 HP | Armory — 200 Shards |
| **Recursive Merge** | After a merge, 10% chance to trigger a free second merge on the new creature | Armory — 180 Shards |
| **Shard Hunger** | Earn +1 bonus Shard per 100 kills in a single level | Armory — 160 Shards |
| **Death Bloom** | When a creature would die in combat, it explodes for 150px damage burst | Armory — 200 Shards |
| **The Quiet** | If no enemies reach the base in a level, earn +3 bonus stars (cosmetic) | Armory — 250 Shards |

---

## 11. Economy & Currency

### Currency Types

| Currency | Name | Earned By | Spent On |
|---|---|---|---|
| 🔶 Essence | In-game | Kills, wave clear, level complete bonuses | Inter-level shop: seeds, upgrades, repair, relics |
| 💠 Shards | Meta | First-time 3-star, daily challenge, achievements | Armory: characters, masteries, cosmetics |

No premium (real money) currency in the base design. Shards replace it entirely — earnable through play.

### Essence Economy

**Income sources:**

| Source | Essence Gained |
|---|---|
| Kill: Grub | 5 |
| Kill: Hulk | 20 |
| Kill: Scuttle (per individual) | 2 |
| Kill: Driftwing | 12 |
| Kill: Divide (parent) | 15 |
| Kill: Vanguard | 18 |
| Kill: Mender | 10 |
| Kill: Siege Lord (Boss) | `150 + level × 8` |
| Wave clear bonus | `10 + level × 1.5` |
| Level complete (1★) | 50 |
| Level complete (2★) | 100 |
| Level complete (3★) | 175 |
| Merge bonus | 10 |
| Sell creature | 50% of seed purchase cost |
| Interest (hold 100+ essence) | +5% of held balance per level completed |

**Spend targets (feel guide):**

| Level Range | Typical essence per level | Spend priority |
|---|---|---|
| 1–10 | 80–150 | Buy Rare seeds when they appear |
| 11–25 | 200–400 | Forge upgrades for your best creatures |
| 26–50 | 400–900 | Relics, repair between hard levels |
| 51+ | 900+ | Rerolls, Epic seeds, stacking upgrades |

### Shard Economy

**Income sources:**

| Source | Shards Gained |
|---|---|
| First time 3-star any level | 5 |
| First time complete any level | 2 |
| Daily challenge (any stars) | 3 |
| Daily challenge (3 stars) | +5 bonus |
| Achievement unlock | 2–20 (varies) |
| Shard Hunger relic (100 kills/level) | 1 per 100 kills |

**Approximate time to unlock all characters (casual player):**
- Brix, Flara, Mossling: Free
- Glitch (80), Ironjaw (120), Wraith (150): ~15 hours
- Crystalis (250), Vex (300): ~25 hours
- Orin (500) or Null (800): ~50+ hours (or via level achievement)

---

## 12. Meta Progression

### Player Account Level

Separate from game levels. Tracks total lifetime progress.

```
XP = levels_completed × 10 + total_stars × 5 + bosses_killed × 8 + merges × 2
```

Player level unlocks:

| Player Level | Unlock |
|---|---|
| 3 | Relic slot 2 unlocked |
| 7 | Reroll available in inter-level shop |
| 12 | Relic slot 3 unlocked |
| 20 | Brutal difficulty mode |
| 30 | Custom creature skin slot |
| 50 | Hall of Fame profile badge |

### Achievement System (40 achievements)

**Progression:**

| Achievement | Condition | Shards |
|---|---|---|
| First Blood | Kill your first enemy | 2 |
| Evolver | Reach Tier 3 with any creature | 3 |
| Merger | Complete your first merge | 3 |
| Ten Forward | Complete Level 10 | 5 |
| Starbound | 3-star any level | 5 |
| Boss Slayer | Defeat 10 Siege Lords | 8 |
| Level 25 | Complete Level 25 | 10 |
| Perfect Defense | Complete a level with 100% base HP | 10 |
| Collector | Unlock 5 different characters | 8 |
| Army of One | Win a level with only 1 creature type | 15 |
| Merge Chain | Perform 5 merges in a single level | 10 |
| Evolution God | Reach Tier 10 with any creature | 20 |
| Century | Complete Level 100 | 50 |
| Legendary | Unlock Null | 20 |
| No Damage Run | Complete any level without base taking damage | 25 |

*(25 more achievements in full list, ranging from speed runs to specific build challenges)*

### Season System

Every 30 days, a new season begins:
- Season leaderboard resets (lifetime stats preserved)
- Top 100 players earn a unique seasonal cosmetic (HUD border, particle theme)
- New seasonal achievement set (10 unique seasonal goals)
- Featured character rotation in the daily Bazaar changes per season

---

## 13. Biomes & Maps

### Biome Progression

| Biome | Levels | Visual Theme | Enemies Emphasized | Boss Variant |
|---|---|---|---|---|
| **Verdant Fields** | 1–10 | Forest, grass, day | Grubs, Hulks | Moss Siege Lord |
| **Ashlands** | 11–20 | Desert, volcanic rock | Scuttles, Driftwings | Ember Siege Lord |
| **Frostmarsh** | 21–35 | Ice, snow, night | Divides, Vanguards | Frost Siege Lord |
| **Deepcore** | 36–60 | Underground, lava rivers | Menders, all prior types | Infernal Siege Lord |
| **The Void** | 61+ | Dark dimension, starfield | All types + Void variants | Void Siege Lord |

Void variants (Level 61+): every enemy type has a Void version with +30% stats and a corruption visual filter.

### Map Templates (5 rotating)

| Template | Description | Strategic implication |
|---|---|---|
| **Open Field** | Wide open, 2 spawn edges | Many placement options, hard to funnel |
| **Chokepoint** | Narrow central corridor | Powerful bottleneck but no room to spread |
| **Split Path** | Two parallel enemy routes | Must split creatures or funnel with walls |
| **Island** | Base on raised platform, paths around it | Flyers especially dangerous |
| **Maze** | Winding paths through obstacles | Long enemy travel time = more shot opportunities |

Map variant per level: `template = level_number mod 5`

### Per-Level Map Variation

Even within templates, each level gets minor variation:
- Spawn points rotate through 4 edges (N/S/E/W)
- Obstacle density ±20% randomized per level
- Biome decorations change (trees → cacti → ice spires → lava vents → void cracks)

---

## 14. Viral & Social Features

### Share Card

Triggered automatically on level complete. Generates a PNG image:
- Background: biome art thumbnail
- Level number (large)
- Star rating (3 animated stars)
- Top 3 creatures: character art + tier badge + kill count
- Player name + level completion time

Auto-shared via native share API on mobile. "Copy image" on web.

### Daily Level

One level per day, same for all players worldwide:
- Fixed `srand(date_seed)` — identical map, enemy composition, wave order
- Player's best star rating shown on main menu banner
- Leaderboard for daily level: ranked by stars first, then time
- 3-star daily = +5 bonus Shards (one-time per day)

### Leaderboard

| Board | Metric | Reset |
|---|---|---|
| Daily | Stars earned today + time | Midnight UTC |
| Weekly | Total levels completed | Monday midnight |
| All-Time | Highest level reached | Never |
| Friends | (future) cross-player | Per session |

### Shareable Run Link

On game over or level complete:
- URL: `/games/tower-swarm?run=<base64(level,stars,creatures,time)>`
- Opening the link shows a read-only replay of the run
- "Try this level" button below the replay

### Achievement Notifications

Pushed to the frontend when earned. Shown as toast notifications.
First-time milestone levels (10, 25, 50, 100) get full-screen animated overlays.

### Streaks

- Daily login streak: +1 bonus Shard per day for streaks of 3+, +3 per day for streaks of 7+
- "Daily challenge streak": consecutive days completing the daily level

---

## 15. Technical Architecture

### Engine Mapping

| Game System | ATMEngine Entity Type | Container Class |
|---|---|---|
| Creatures | Hybrid | `CreatureContainer` |
| Enemies | Dynamic | `EnemyContainer` |
| Projectiles | Dynamic (pooled) | `ProjectileContainer` |
| Particles / VFX | Dynamic (pooled) | `ParticleContainer` |
| Essence orb pickups | Dynamic (pooled) | `PickupContainer` |
| Terrain / Tilemap | Static | `TileContainer` |
| Player Base | Static | `BaseEntity` |
| Walls | Static | `WallContainer` |
| HUD / UI | Direct SDL draw (no entity system) | — |

### Data Flow Per Frame

```
SDL_PollEvent → InputManager
                ↓
engine_update(engine)
engine_update_entity_types(engine, dt)
    → CreatureContainer::update(dt)    [Hybrid: only when visible]
        → EvolutionSystem::tick(dt)
        → CreatureAI::tick(dt)
        → MergeSystem::tick(dt)
    → EnemyContainer::update(dt)       [Dynamic: always]
        → EnemyAI::tick(dt)
    → ProjectileContainer::update(dt)  [Dynamic: always]
        → collision via queryCircle
    → ParticleContainer::update(dt)    [Dynamic: always]
    → PickupContainer::update(dt)      [Dynamic: always]

LevelManager::tick(dt)                 [State machine: PLAYING/WAVE_CLEAR/LEVEL_CLEAR/etc.]
WaveSpawner::tick(dt)

SDL_RenderClear
engine_render_scene(engine)            [Batch render all visible entities]
HUD::render(engine)                    [Direct SDL draw over everything]
engine_present(engine)
```

### Key Subsystem Files

```
games/tower_swarm/src/
├── tower_swarm_main.cpp         — main loop
├── TowerSwarmGame.h/.cpp        — screen manager, root state
│
├── entities/
│   ├── CreatureContainer.h/.cpp
│   ├── EnemyContainer.h/.cpp
│   ├── ProjectileContainer.h/.cpp
│   ├── ParticleContainer.h/.cpp
│   ├── PickupContainer.h/.cpp
│   ├── TileContainer.h/.cpp
│   └── WallContainer.h/.cpp
│
├── systems/
│   ├── EvolutionSystem.h/.cpp
│   ├── MergeSystem.h/.cpp
│   ├── CreatureAI.h/.cpp
│   ├── EnemyAI.h/.cpp
│   ├── EssenceSystem.h/.cpp
│   ├── PathGrid.h/.cpp
│   └── VFX.h/.cpp
│
├── levels/
│   ├── LevelManager.h/.cpp
│   ├── LevelScaler.h            — all formulas
│   ├── LevelDefinition.h
│   ├── WaveSpawner.h/.cpp
│   ├── MapVariant.h/.cpp
│   └── SaveState.h/.cpp
│
├── screens/
│   ├── GameplayScreen.h/.cpp
│   ├── InterLevelScreen.h/.cpp
│   ├── LevelSelectScreen.h/.cpp
│   ├── ArmoryScreen.h/.cpp
│   ├── MainMenuScreen.h/.cpp
│   └── HUD.h/.cpp
│
├── shop/
│   ├── WaveBuffShop.h/.cpp
│   ├── InterLevelShop.h/.cpp    — Bazaar + Forge + Relics + Repair tabs
│   ├── RelicSystem.h/.cpp
│   └── UpgradeSystem.h/.cpp
│
├── characters/
│   ├── CharacterDefinitions.h   — all 10 character stat tables
│   ├── CharacterUnlocks.h/.cpp
│   └── CharacterRoster.h/.cpp
│
└── Constants.h                  — ALL magic numbers here, nowhere else
```

### Performance Targets

| Scenario | Target FPS |
|---|---|
| Level 1–50 (native) | 60 FPS |
| Level 50–150 (native) | 60 FPS |
| Level 150–300 (native) | 30+ FPS |
| Level 1–60 (WASM/Chrome) | 60 FPS |
| Level 60–100 (WASM/Chrome) | 30+ FPS |

### Save State

Persisted to `localStorage` (WASM) / file (native):
```json
{
  "max_level_reached": 47,
  "stars_per_level": [3, 3, 2, 3, ...],
  "essence": 340,
  "shards": 215,
  "player_level": 12,
  "roster": [
    { "character": "Brix", "tier": 7, "kills": 1840, "upgrades": [3,2,0,1,0] },
    { "character": "Flara", "tier": 5, "kills": 620, "upgrades": [2,1,0,0,0] }
  ],
  "equipped_relics": ["Iron Core", "Essence Magnet", "Merger's Gift"],
  "unlocked_characters": ["Brix", "Flara", "Mossling", "Glitch"],
  "masteries": { "Echo Foundation": 2, "Rapid Growth": 1 },
  "achievements_unlocked": ["First Blood", "Evolver", "Merger", "Ten Forward"]
}
```

### Backend API (NoobyGame Server)

| Endpoint | Purpose |
|---|---|
| `POST /api/games/tower-swarm/level-complete` | Submit level result (level, stars, time, creature snapshot) |
| `GET /api/games/tower-swarm/leaderboard?mode=daily\|weekly\|alltime` | Fetch ranked boards |
| `GET /api/games/tower-swarm/daily-level` | Today's level number + seed |
| `GET /api/games/tower-swarm/player/:id/progress` | Full player progress object |
| `POST /api/games/tower-swarm/sync-save` | Push save state to server (backup) |
| `GET /api/games/tower-swarm/replay/:run_id` | Fetch a replay for spectator view |

Database tables:
```sql
tower_swarm_runs       (id, user_id, level, stars, time_sec, creature_json, created_at)
tower_swarm_progress   (user_id, max_level, total_stars, player_level, shards, essence, roster_json, created_at)
tower_swarm_level_best (user_id, level_number, best_stars, best_time_sec)
tower_swarm_daily      (user_id, date, level_number, stars, time_sec)
```

---

## 16. Build Phases (Implementation Order)

### Milestone Map

```
PHASE 0–7   → ★ PLAYABLE PROTOTYPE
PHASE 8–11  → ★ CORE LOOP COMPLETE
PHASE 12–16 → ★ FULL GAME
PHASE 17–20 → ★ SOFT LAUNCH
PHASE 21–24 → ★ HARD LAUNCH
```

---

### Phase 0 — Project Bootstrap
- [ ] Create `/games/tower_swarm/` from `_template`
- [ ] CMakeLists.txt, register in root build
- [ ] Stub `main.cpp` compiles clean
- [ ] `tower-swarm-wasm.constants.ts` + `game-catalog.constants.ts` entry
- [ ] Verify WASM build pipeline (`.wasm` + `.js` loader)

### Phase 1 — World & Camera
- [ ] 1280×720 window, 5120×2880 world
- [ ] `CameraController`: pan + clamp + smooth lerp
- [ ] Background tilemap (`Static` entities, biome-skinnable)
- [ ] Debug grid overlay (toggle)
- [ ] Basic HUD layer placeholder

### Phase 2 — Player Base
- [ ] `BaseEntity` (Static), 100 HP
- [ ] HP bar + star threshold markers at 30% and 70%
- [ ] `LEVEL_FAILED` state trigger at HP = 0
- [ ] Fail screen: retry / level select

### Phase 3 — Creature Container
- [ ] Full SoA `CreatureContainer` (all arrays, swapSlots, resizeArrays)
- [ ] `createCreature(x, y, type, tier)` factory
- [ ] Grid-snap placement + ghost preview
- [ ] Sell mechanic (right-click → 50% refund)
- [ ] `SaveState` serialize / restore roster

### Phase 4 — Enemy Container
- [ ] Full SoA `EnemyContainer`
- [ ] Direct-vector pathfinding toward base
- [ ] HP bars above enemies
- [ ] Death → essence pickup entity spawn

### Phase 5 — Level & Wave System
- [ ] `LevelManager` state machine (PLAYING / WAVE_CLEAR / LEVEL_CLEAR / FAILED)
- [ ] `LevelScaler` module (all formulas)
- [ ] `LevelDefinition` auto-generated for any N
- [ ] `WaveSpawner`: spawn queue, inter-spawn delay, grace timer
- [ ] Wave/level banners + boss-wave detection

### Phase 6 — Projectile System
- [ ] `ProjectileContainer` pooled (10,000 slots)
- [ ] Velocity movement, lifetime, hit detection via `queryCircle`
- [ ] Per-type visual variants

### Phase 7 — Creature Combat AI
- [ ] Target acquisition via `queryCircle`
- [ ] Attack fire + cooldown
- [ ] Re-target on death/range-exit
- [ ] Range indicator on hover

> **★ PLAYABLE PROTOTYPE** — Level 1: Brix vs Grubs, 5 waves, win/fail/restart

### Phase 8 — Wave Buff Shop
- [ ] Card pool data structure (12 cards)
- [ ] Random 3-of-12 draw each wave clear
- [ ] Timer-driven selection
- [ ] Temporary buff application to game state

### Phase 9 — Evolution System
- [ ] Kill tracking per creature slot
- [ ] Threshold checking + tier-up
- [ ] Evolution animation (pulse, color shift, size)
- [ ] Floating text + screen edge glow + sound
- [ ] Progress bar on selected creature

### Phase 10 — Merge Mechanic
- [ ] Adjacency detection (same type + tier)
- [ ] Pulsing link indicator
- [ ] Manual merge (drag) + auto-merge (6s idle)
- [ ] Merge animation + essence bonus
- [ ] Kill inheritance

### Phase 11 — Inter-Level Screen (Core)
- [ ] Results panel: stars animate in, stats displayed
- [ ] Bazaar tab: 4 random seeds, buy mechanic
- [ ] Forge tab: upgrade nodes per character
- [ ] Repair tab: buy HP for next level
- [ ] Roster strip at bottom
- [ ] "Next Level" → preserve state and load Level N+1

> **★ CORE LOOP COMPLETE** — Play → Win/Fail → Shop → Next Level

### Phase 12 — All 10 Characters
- [ ] `CharacterDefinitions.h` with stat tables for all 10
- [ ] All 3 evolution stage names + ability text
- [ ] Signature ability implemented per character
- [ ] Type-specific combat branching in `CreatureAI`
- [ ] Visual shapes/colors per type + tier

### Phase 13 — All 8 Enemy Types
- [ ] Each enemy type: HP, speed, behavior logic
- [ ] New-enemy introduction banner
- [ ] Boss multi-phase logic (minion spawns at 66% / 33%)
- [ ] Level-based enemy composition table

### Phase 14 — Creature Movement AI
- [ ] Threat density query (3 radii)
- [ ] Desired-position calculation
- [ ] A* pathfinding on 64px grid
- [ ] Staggered scheduler (spread recalculation across frames)
- [ ] Smooth position interpolation
- [ ] Player drag-override

### Phase 15 — Relic System
- [ ] `RelicSystem`: 3 equipped slots, global effect application
- [ ] All 20 relics implemented
- [ ] Relic Vault tab in inter-level shop
- [ ] Relic unlock tracking in save state

### Phase 16 — Armory + Meta Progression
- [ ] Armory screen: character gallery, mastery tree, cosmetics
- [ ] Shard currency + earning hooks
- [ ] Character unlock flow (Shards → unlock → available in Bazaar)
- [ ] Passive mastery application to game formulas
- [ ] Player account level + XP system

> **★ FULL GAME** — All systems connected end-to-end

### Phase 17 — Map Variants + Biomes
- [ ] 5 map templates with spawn/obstacle/base layouts
- [ ] Biome tile skins per theme (5 biomes)
- [ ] Level select tile previews
- [ ] Biome transition banner at level 10/20/35/61

### Phase 18 — VFX System
- [ ] `ParticleContainer` pooled (50,000 slots)
- [ ] All particle types: explosion, evolve burst, merge flash, death dissolve, essence trail
- [ ] Screen shake (boss death = max)
- [ ] Damage numbers (floating text pool)
- [ ] Creature glow (additive blend, tier 5+)
- [ ] Level complete confetti + star reveal animation

### Phase 19 — Full HUD & UI Polish
- [ ] Top bar: Level N | Wave X/Y | Essence | (star thresholds)
- [ ] Base HP bar with 30% / 70% markers
- [ ] Selected creature panel (full stats + progress bar)
- [ ] Creature type selector wheel with lock states
- [ ] Pause menu (Resume / Retry / Level Select / Settings)
- [ ] Kill feed, evolution banner, new-enemy banner
- [ ] Level select screen with full tile grid

### Phase 20 — Audio
- [ ] SDL3 audio mixer + SFX pool
- [ ] Per-biome music layers (calm/tense/intense/boss)
- [ ] Evolution fanfare per tier range
- [ ] Level complete jingle
- [ ] Volume controls (master / SFX / music)

### Phase 21 — Viral Features
- [ ] Share card PNG generator
- [ ] 3-star auto-screenshot
- [ ] Shareable run URL encoder/decoder
- [ ] Daily level (date-seeded)
- [ ] Achievement push notifications
- [ ] Daily streak counter

### Phase 22 — Backend Integration
- [ ] All 6 API endpoints implemented
- [ ] Leaderboard component in Angular NoobyGame
- [ ] Save state server sync
- [ ] Daily level badge on game page
- [ ] Rate limiting + score validation

> **★ SOFT LAUNCH** — deploy to staging, 2-week feedback loop

### Phase 23 — Performance Hardening
- [ ] Entity pooling: 100k enemies, 50k projectiles, 50k particles
- [ ] LOD: distant enemies skip AI
- [ ] Batch spawn (50/frame max)
- [ ] Profiler hooks on wave peaks
- [ ] Stress test: Level 200+ headless run

### Phase 24 — QA & Balance
- [ ] Balance spreadsheet (all stat tables, formulas, economy)
- [ ] Playtest targets: Level 3 (~10 min), Level 10 (~30 min), Level 25 (~60 min total)
- [ ] Unit tests: evolution thresholds, level scaler, merge eligibility, save/restore
- [ ] Retry fairness audit (essence snapshot exact restore)
- [ ] 0-crash requirement: 50 consecutive runs to Level 10

> **★ HARD LAUNCH**

---

*Tower Swarm — GDD v1.0 | 2026-03-15 | ATMEngine C++20 SDL3 → WASM → Angular NoobyGame*
