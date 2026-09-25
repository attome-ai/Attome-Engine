// Zone server gameplay: players, monsters, combat, damage-share loot,
// inventory and skills (GAME_DESIGN §8, §9, §11).

#include "ServerState.h"

#include <algorithm>
#include <cmath>

namespace ao::server {

using namespace ao::proto;
namespace vx = atm::voxel;
using atm::model::EquipSlot;

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

Entity *ServerState::findEntity(EntityId id) {
  if (id == kNoEntity) return nullptr;
  auto it = entities.find(id);
  return it == entities.end() ? nullptr : &it->second;
}

Player *ServerState::playerOf(const Entity &e) {
  if (e.kind != EntityKind::Player) return nullptr;
  auto it = players.find(e.playerKey);
  return it == players.end() ? nullptr : &it->second;
}

Entity &ServerState::queueSpawn(EntityKind kind) {
  Entity &e = spawnQueue.emplace_back();
  e.id = nextEntityId++;
  e.kind = kind;
  return e; // valid until the next queueSpawn
}

void ServerState::flushSpawns() {
  for (Entity &e : spawnQueue) {
    const EntityId id = e.id;
    entities.emplace(id, std::move(e));
  }
  spawnQueue.clear();
}

void ServerState::removeDead() {
  uint32_t monsters = 0;
  for (auto it = entities.begin(); it != entities.end();) {
    Entity &e = it->second;
    const bool expired = e.dead && e.kind != EntityKind::Player && tick >= e.despawnTick;
    if (e.remove || expired) {
      it = entities.erase(it);
      continue;
    }
    if (e.kind == EntityKind::Monster) ++monsters;
    ++it;
  }
  monsterCount = monsters;
}

void ServerState::startAction(Entity &e, uint8_t anim) {
  e.actionAnim = anim;
  e.actionSeq = uint8_t(e.actionSeq + 1);
}

float ServerState::randf() { return float(double(rng() >> 11) * (1.0 / 9007199254740992.0)); }

int ServerState::randi(int lo, int hi) {
  if (hi <= lo) return lo;
  return lo + int(rng() % uint64_t(hi - lo + 1));
}

// ---------------------------------------------------------------------------
// Skills and items
// ---------------------------------------------------------------------------

uint32_t ServerState::skillLevel(const Player &pl, Skill s) const { return levelForXp(pl.xp[size_t(s)]); }

void ServerState::addXp(Player &pl, Skill s, double amount) {
  if (!(amount > 0.0)) return;
  const size_t i = size_t(s);
  const double total = amount + pl.xpFraction[i];
  const double whole = std::floor(total);
  pl.xpFraction[i] = total - whole;
  if (whole < 1.0) return;
  const uint32_t before = levelForXp(pl.xp[i]);
  pl.xp[i] += uint64_t(whole);
  pl.xpGained[i] += uint64_t(whole);
  if (s == Skill::Hitpoints && levelForXp(pl.xp[i]) != before) {
    if (Entity *e = findEntity(pl.entity)) {
      const uint16_t newMax = maxHpFor(pl);
      e->hp = uint16_t(std::min<uint32_t>(uint32_t(e->hp) + (newMax - std::min(newMax, e->maxHp)), newMax));
      e->maxHp = newMax;
    }
  }
}

void ServerState::flushXp(Player &pl) {
  for (int s = 0; s < kSkillCount; ++s) {
    uint64_t &g = pl.xpGained[size_t(s)];
    if (g == 0) continue;
    XpGain x;
    x.skill = uint8_t(s);
    x.amount = uint32_t(std::min<uint64_t>(g, 0xFFFFFFFFull));
    x.totalXp = pl.xp[size_t(s)];
    sendTo(pl, x, Channel::ReliableOrdered);
    g = 0;
  }
}

uint16_t ServerState::maxHpFor(const Player &pl) const {
  const uint32_t lvl = std::max<uint32_t>(10, skillLevel(pl, Skill::Hitpoints));
  return uint16_t(std::min<uint32_t>(lvl * 10u, 60000u));
}

uint16_t ServerState::armourOf(const Player &pl) const {
  uint32_t a = 0;
  for (ItemId it : pl.equipped)
    if (it) a += itemDef(it).armour;
  return uint16_t(std::min<uint32_t>(a, 60000u));
}

uint16_t ServerState::addItem(Player &pl, ItemId item, uint16_t count) {
  if (item == 0 || count == 0 || item >= itemCount()) return 0;
  const uint16_t maxStack = std::max<uint16_t>(1, itemDef(item).maxStack);
  uint32_t left = count;
  for (ItemStack &s : pl.inv) { // top up existing stacks first
    if (left == 0) break;
    if (s.item == item && s.count < maxStack) {
      const uint32_t add = std::min<uint32_t>(left, uint32_t(maxStack - s.count));
      s.count = uint16_t(s.count + add);
      left -= add;
    }
  }
  for (ItemStack &s : pl.inv) {
    if (left == 0) break;
    if (s.item == 0 || s.count == 0) {
      const uint32_t add = std::min<uint32_t>(left, maxStack);
      s.item = item;
      s.count = uint16_t(add);
      left -= add;
    }
  }
  if (left != count) pl.inventoryDirty = true;
  return uint16_t(left);
}

bool ServerState::removeFromSlot(Player &pl, int slot, uint16_t count) {
  if (slot < 0 || slot >= kInventorySlots) return false;
  ItemStack &s = pl.inv[size_t(slot)];
  if (s.item == 0 || s.count < count) return false;
  s.count = uint16_t(s.count - count);
  if (s.count == 0) s.item = 0;
  pl.inventoryDirty = true;
  return true;
}

void ServerState::giveOrDrop(Player &pl, ItemId item, uint16_t count, const glm::dvec3 &at, EntityId from,
                             bool rare) {
  if (item == 0 || count == 0) return;
  const uint16_t left = addItem(pl, item, count);
  if (left > 0) { // inventory full: a personal dropped item
    Entity &d = queueSpawn(EntityKind::DroppedItem);
    d.item = item;
    d.type = uint8_t(item & 0xFF);
    d.stack = ItemStack{item, left};
    d.owner = pl.entity;
    d.publicTick = tick + Tick(kSimHz * 60);
    d.expireTick = tick + Tick(kSimHz * 300);
    d.move.pos = at;
    d.hp = d.maxHp = 1;
  }
  if (from != kNoEntity) {
    LootMsg lm;
    lm.fromEntity = from;
    lm.item = item;
    lm.count = count;
    lm.rare = rare;
    sendTo(pl, lm, Channel::ReliableOrdered);
  }
}

void ServerState::sendInventory(Player &pl) {
  InventoryMsg m;
  m.slots.assign(pl.inv.begin(), pl.inv.end());
  m.equipped.assign(pl.equipped.begin(), pl.equipped.end());
  sendTo(pl, m, Channel::ReliableOrdered);
  pl.inventoryDirty = false;
}

void ServerState::refreshAppearance(Player &pl) {
  for (int i = 0; i < atm::model::kEquipSlotCount; ++i) {
    const ItemId it = pl.equipped[size_t(i)];
    pl.appearance.pieces[size_t(i)] = it < itemPiece.size() ? itemPiece[it] : atm::model::kNoPiece;
  }
  ++pl.appearanceVersion;
  if (pl.welcomed && pl.entity != kNoEntity) { // own client (others: replication)
    AppearanceMsg am;
    am.entity = pl.entity;
    am.name = pl.name;
    am.appearance = pl.appearance;
    sendTo(pl, am, Channel::ReliableOrdered);
  }
}

void ServerState::giveStartingKit(Player &pl) {
  pl.inv = {};
  pl.equipped = {};
  pl.xp = {};
  pl.xp[size_t(Skill::Hitpoints)] = xpForLevel(10); // RS start: 10 Hitpoints
  pl.equipped[size_t(EquipSlot::MainHand)] = items::WoodenSword;
  pl.equipped[size_t(EquipSlot::Torso)] = items::LeatherTunic;
  const ItemStack kit[] = {
      {items::Dirt, 64},        {items::Planks, 64},     {items::Bow, 1},
      {items::Pickaxe, 1},      {items::Arrows, 200},    {items::LeatherCap, 1},
      {items::LeatherPants, 1}, {items::Boots, 1},       {items::LeatherGloves, 1},
      {items::CookedMeat, 10},  {items::RedCape, 1},     {items::IronSword, 1},
  };
  size_t i = 0;
  for (const ItemStack &s : kit)
    if (s.item < itemCount() && i < pl.inv.size()) pl.inv[i++] = s;
  pl.inventoryDirty = true;
}

// ---------------------------------------------------------------------------
// Combat
// ---------------------------------------------------------------------------

uint16_t ServerState::rollDamage(const Player &pl, WeaponType weapon, Skill style, bool &critical) {
  (void)style;
  const ItemId wi = pl.equipped[size_t(EquipSlot::MainHand)];
  const double base = wi ? std::max<double>(1.0, itemDef(wi).damage) : 1.0;
  Skill dmgSkill = Skill::Strength, accSkill = Skill::Attack;
  if (weapon == WeaponType::Bow) dmgSkill = accSkill = Skill::Ranged;
  if (weapon == WeaponType::Staff) dmgSkill = accSkill = Skill::Magic;
  const double maxHit = base * (1.0 + double(skillLevel(pl, dmgSkill)) / 40.0);
  const int hi = std::max(1, int(std::lround(maxHit)));
  const int lo = std::max(1, hi / 2);
  int dmg = randi(lo, hi);
  const double critChance = std::min(0.3, 0.05 + double(skillLevel(pl, accSkill)) * 0.002);
  critical = randf() < critChance;
  if (critical) dmg = int(std::lround(dmg * 1.5));
  return uint16_t(std::clamp(dmg, 1, 60000));
}

void ServerState::damageMonster(Entity &m, Player &attacker, uint16_t amount, bool critical, Skill style) {
  if (m.dead || m.kind != EntityKind::Monster || amount == 0) return;
  const uint16_t dealt = std::min(amount, m.hp);
  m.hp = uint16_t(m.hp - dealt);
  bool found = false;
  for (auto &[who, dmg] : m.damageBy)
    if (who == attacker.entity) {
      dmg += dealt;
      found = true;
    }
  if (!found && m.damageBy.size() < 64) m.damageBy.emplace_back(attacker.entity, uint32_t(dealt));

  // RS-style combat XP per damage: 4 to the style skill + 1.33 to Hitpoints.
  addXp(attacker, style, 4.0 * dealt);
  addXp(attacker, Skill::Hitpoints, 1.33 * dealt);
  attacker.lastStyle = style;

  if (m.target == kNoEntity) m.target = attacker.entity;
  startAction(m, action::Hit);
  DamageEvent ev;
  ev.source = attacker.entity;
  ev.target = m.id;
  ev.amount = dealt;
  ev.critical = critical;
  ev.killed = m.hp == 0;
  sendNear(m.move.pos, kEventRange, ev, Channel::ReliableOrdered);
  if (m.hp == 0) killMonster(m);
}

// Damage-share loot (GAME_DESIGN §11).
void ServerState::killMonster(Entity &m) {
  m.dead = true;
  m.despawnTick = tick + Tick(kSimHz); // death animation, then gone
  m.move.vel = glm::vec3(0.0f);
  startAction(m, action::Death);
  const MonsterDef &def = monsterDef(m.type);

  struct Contributor { Player *pl; double share; };
  Contributor elig[64];
  int nElig = 0;
  double total = 0.0;
  for (const auto &[who, dmg] : m.damageBy) total += double(dmg);
  if (total <= 0.0) return;
  for (const auto &[who, dmg] : m.damageBy) {
    if (double(dmg) * 20.0 < double(def.maxHp)) continue; // rule 1: >= 5% of max HP
    Entity *pe = findEntity(who);
    if (!pe || pe->dead) continue;
    const double dx = pe->move.pos.x - m.move.pos.x, dz = pe->move.pos.z - m.move.pos.z;
    if (dx * dx + dz * dz > kEventRange * kEventRange) continue; // rule 1: in range
    Player *pl = playerOf(*pe);
    if (!pl || nElig >= 64) continue;
    elig[nElig++] = Contributor{pl, double(dmg) / total};
  }
  if (nElig == 0) return;

  const glm::dvec3 at = m.move.pos + glm::dvec3(0.0, 0.5, 0.0);
  for (int i = 0; i < nElig; ++i) {
    Player &pl = *elig[i].pl;
    // Kill bonus XP, shared by damage, to the style last used.
    addXp(pl, pl.lastStyle, double(def.xp) * elig[i].share);
    // Rule 2: own common rolls, quantity scaled by damage share.
    for (const MonsterDef::Drop &d : def.common) {
      if (d.item == 0 || randf() >= d.chance) continue;
      const int q = randi(d.min, std::max(d.min, d.max));
      const uint16_t count = uint16_t(std::max(1L, std::lround(double(q) * elig[i].share)));
      giveOrDrop(pl, d.item, count, at, m.id, false);
    }
  }
  // Rule 3: one rare roll per kill, winner weighted by damage share.
  if (def.rare.item != 0 && randf() < def.rare.chance) {
    double pick = double(randf());
    double sumShare = 0.0;
    for (int i = 0; i < nElig; ++i) sumShare += elig[i].share;
    pick *= sumShare;
    int winner = nElig - 1;
    for (int i = 0; i < nElig; ++i) {
      if (pick < elig[i].share) {
        winner = i;
        break;
      }
      pick -= elig[i].share;
    }
    giveOrDrop(*elig[winner].pl, def.rare.item, std::max<uint16_t>(1, def.rare.min), at, m.id, true);
  }
}

void ServerState::damagePlayer(Entity &victim, uint16_t amount, EntityId source) {
  Player *pl = playerOf(victim);
  if (!pl || victim.dead || amount == 0 || victim.hp == 0) return;
  const double reduce = 100.0 / (100.0 + armourOf(*pl) * 4.0 + skillLevel(*pl, Skill::Defence));
  const uint16_t dmg = uint16_t(std::clamp<long>(std::lround(amount * reduce), 1L, long(victim.hp)));
  victim.hp = uint16_t(victim.hp - dmg);
  pl->lastHurtTick = tick;
  addXp(*pl, Skill::Defence, 1.0 * dmg); // Defence trains by taking hits
  startAction(victim, action::Hit);
  DamageEvent ev;
  ev.source = source;
  ev.target = victim.id;
  ev.amount = dmg;
  ev.critical = false;
  ev.killed = victim.hp == 0;
  sendNear(victim.move.pos, kEventRange, ev, Channel::ReliableOrdered);
  if (victim.hp == 0) {
    victim.dead = true; // outside the Wilderness: keep items, respawn at spawn (§14)
    victim.move.vel = glm::vec3(0.0f);
    startAction(victim, action::Death);
    pl->respawnTick = tick + Tick(kSimHz * 3);
  }
}

void ServerState::meleeAttack(Player &pl, Entity &pe, const glm::vec3 &dir, Skill style) {
  startAction(pe, action::Swing);
  const ItemId wi = pl.equipped[size_t(EquipSlot::MainHand)];
  const WeaponType weapon = wi ? itemDef(wi).weapon : WeaponType::None;
  const glm::dvec3 eye = eyePosition(pe.move);
  const glm::dvec3 aim(dir);
  constexpr double kRange = 3.0, kMonsterRadius = 0.6, kCosHalfCone = 0.70710678; // 90 degree cone
  candidates.clear();
  for (auto &[id, m] : entities) {
    if (m.kind != EntityKind::Monster || m.dead) continue;
    const glm::dvec3 d = (m.move.pos + glm::dvec3(0.0, 0.8, 0.0)) - eye;
    const double dist = glm::length(d);
    if (dist > kRange + kMonsterRadius) continue;
    if (dist > 0.5 && glm::dot(d / dist, aim) < kCosHalfCone) continue;
    candidates.emplace_back(float(dist), id);
  }
  std::sort(candidates.begin(), candidates.end());
  const size_t n = std::min<size_t>(candidates.size(), 3); // cleave up to 3 targets
  for (size_t i = 0; i < n; ++i) {
    Entity *m = findEntity(candidates[i].second);
    if (!m) continue;
    bool crit = false;
    const uint16_t dmg = rollDamage(pl, weapon, style, crit);
    damageMonster(*m, pl, dmg, crit, style);
  }
  candidates.clear();
}

void ServerState::rangedAttack(Player &pl, Entity &pe, const glm::vec3 &dir, WeaponType weapon) {
  const bool bow = weapon == WeaponType::Bow;
  if (bow) {
    int arrowSlot = -1;
    for (int i = 0; i < kInventorySlots; ++i)
      if (pl.inv[size_t(i)].item == items::Arrows && pl.inv[size_t(i)].count > 0) {
        arrowSlot = i;
        break;
      }
    if (arrowSlot < 0) {
      ChatMsg cm;
      cm.from = "Server";
      cm.text = "You have no arrows.";
      sendTo(pl, cm, Channel::ReliableOrdered);
      return;
    }
    removeFromSlot(pl, arrowSlot, 1);
  }
  startAction(pe, bow ? action::BowShoot : action::Cast);
  const Skill style = bow ? Skill::Ranged : Skill::Magic;
  bool crit = false;
  const uint16_t dmg = rollDamage(pl, weapon, style, crit);
  const glm::dvec3 eye = eyePosition(pe.move);
  const EntityId owner = pe.id;
  Entity &p = queueSpawn(EntityKind::Projectile);
  p.type = uint8_t(weapon);
  p.owner = owner;
  p.move.pos = eye + glm::dvec3(dir) * 0.6;
  p.move.vel = dir * (bow ? 40.0f : 28.0f);
  p.move.yaw = std::atan2(-dir.x, -dir.z);
  p.gravity = bow ? 18.0f : 0.0f;
  p.expireTick = tick + Tick(kSimHz * 3);
  p.damage = dmg;
  p.critical = crit;
  p.style = style;
  p.hp = p.maxHp = 1;
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

void ServerState::simulatePlayers() {
  for (auto &[key, pl] : players) {
    if (!pl.welcomed) continue;
    Entity *e = findEntity(pl.entity);
    if (!e) continue;

    if (e->dead) {
      // Inputs still count as applied so prediction history drains.
      while (pl.inputCount > 0) {
        pl.lastAppliedSeq = pl.inputs[size_t(pl.inputHead)].seq;
        pl.inputHead = (pl.inputHead + 1) % kInputQueue;
        --pl.inputCount;
      }
      if (tick >= pl.respawnTick) {
        const int32_t sx = int32_t(std::floor(spawnPoint.x)) + randi(-4, 4);
        const int32_t sz = int32_t(std::floor(spawnPoint.z)) + randi(-4, 4);
        const int h = world->generator().surfaceHeight(sx, sz);
        e->move = MoveState{};
        e->move.pos = glm::dvec3(sx + 0.5, h + 1.05, sz + 0.5);
        e->dead = false;
        e->maxHp = maxHpFor(pl);
        e->hp = e->maxHp;
        startAction(*e, action::None);
      }
      continue;
    }

    // Input budget: on average one input per tick (speed hacks gain nothing),
    // with a small catch-up when inputs arrive in bursts.
    pl.inputCredit = std::min(pl.inputCredit + 1.0f, 4.0f);
    int steps = 0;
    while (pl.inputCount > 0 && pl.inputCredit >= 1.0f && steps < 3) {
      if (steps > 0 && pl.inputCount <= 2) break; // catch up only when behind
      const MoveInput in = pl.inputs[size_t(pl.inputHead)];
      pl.inputHead = (pl.inputHead + 1) % kInputQueue;
      --pl.inputCount;
      stepMovement(e->move, in, *world, blocks, kSimDt);
      pl.lastAppliedSeq = in.seq;
      pl.lastButtons = in.buttons;
      pl.inputCredit -= 1.0f;
      ++steps;
    }

    if (e->move.pos.y < -32.0) { // fell out of the world
      e->hp = 0;
      e->dead = true;
      startAction(*e, action::Death);
      pl.respawnTick = tick + Tick(kSimHz * 3);
      continue;
    }
    // Regeneration out of combat.
    if (e->hp < e->maxHp && tick - pl.lastHurtTick > Tick(kSimHz * 5) && tick >= pl.regenTick) {
      ++e->hp;
      pl.regenTick = tick + Tick(kSimHz * 2);
    }
  }
}

bool ServerState::findGround(int32_t x, int32_t z, int32_t &groundY) const {
  const int h = world->generator().surfaceHeight(x, z);
  for (int y = std::min(h + 4, vx::kWorldHeight - 3); y >= std::max(1, h - 8); --y) {
    const vx::BlockPos p{x, y, z};
    if (!world->isLoaded(vx::chunkOf(p)) || !world->isLoaded(vx::chunkOf(vx::BlockPos{x, y + 2, z})))
      return false;
    if (blocks.solid(world->blockAt(p)) && !blocks.solid(world->blockAt({x, y + 1, z})) &&
        !blocks.solid(world->blockAt({x, y + 2, z}))) {
      groundY = y;
      return true;
    }
  }
  return false;
}

void ServerState::spawnMonsters() {
  idScratch.clear();
  for (auto &[key, pl] : players)
    if (pl.welcomed) idScratch.push_back(pl.entity);
  if (idScratch.empty()) return;
  const int target = std::min(cfg.maxMonsters, int(idScratch.size()) * cfg.monstersPerPlayer);
  const int toSpawn = std::min(target - int(monsterCount) - int(spawnQueue.size()), 4);
  for (int i = 0; i < toSpawn; ++i) {
    const Entity *pe = findEntity(idScratch[size_t(randi(0, int(idScratch.size()) - 1))]);
    if (!pe || pe->dead) continue;
    const double angle = double(randf()) * 6.283185307179586;
    const double dist = 20.0 + double(randf()) * 20.0;
    const int32_t x = int32_t(std::floor(pe->move.pos.x + std::cos(angle) * dist));
    const int32_t z = int32_t(std::floor(pe->move.pos.z + std::sin(angle) * dist));
    int32_t gy = 0;
    if (!findGround(x, z, gy) || world->blockAt({x, gy, z}) != vx::blocks::Grass) continue;
    const float r = randf();
    const uint8_t type = r < 0.5f ? monsters::Slime : (r < 0.85f ? monsters::Wolf : monsters::Golem);
    const MonsterDef &def = monsterDef(type);
    Entity &m = queueSpawn(EntityKind::Monster);
    m.type = type;
    m.move.pos = glm::dvec3(x + 0.5, gy + 1.02, z + 0.5);
    m.move.yaw = float(randf() * 6.2831853f - 3.1415927f);
    m.hp = m.maxHp = def.maxHp;
    m.wanderTarget = m.move.pos;
    m.wanderUntil = tick + Tick(randi(30, 120));
  }
}

void ServerState::simulateMonsters() {
  const bool despawnCheck = tick % Tick(kSimHz) == 0;
  for (auto &[id, m] : entities) {
    if (m.kind != EntityKind::Monster || m.dead || m.remove) continue;
    const MonsterDef &def = monsterDef(m.type);

    // Target upkeep / acquisition.
    Entity *t = findEntity(m.target);
    if (t) {
      const double dx = t->move.pos.x - m.move.pos.x, dz = t->move.pos.z - m.move.pos.z;
      if (t->dead || dx * dx + dz * dz > double(def.aggroRange) * def.aggroRange * 6.25) {
        m.target = kNoEntity;
        t = nullptr;
      }
    } else {
      m.target = kNoEntity;
    }
    double nearest2 = 1e30;
    bool anyPlayerNear = false;
    for (auto &[key, pl] : players) {
      Entity *pe = findEntity(pl.entity);
      if (!pe) continue;
      const double dx = pe->move.pos.x - m.move.pos.x, dz = pe->move.pos.z - m.move.pos.z;
      const double d2 = dx * dx + dz * dz;
      if (d2 < 100.0 * 100.0) anyPlayerNear = true;
      if (!t && !pe->dead && d2 < double(def.aggroRange) * def.aggroRange && d2 < nearest2) {
        nearest2 = d2;
        m.target = pe->id;
      }
    }
    if (!t) t = findEntity(m.target);
    if (despawnCheck && !anyPlayerNear) {
      m.remove = true;
      continue;
    }

    MoveInput in;
    in.tick = tick;
    float want = 0.0f;
    glm::dvec3 goal = m.wanderTarget;
    if (t) {
      goal = t->move.pos;
      const glm::dvec3 d = t->move.pos - m.move.pos;
      const double dist = std::sqrt(d.x * d.x + d.z * d.z);
      if (dist > def.attackRange) {
        want = std::min(1.0f, def.speed / moveTuning().walkSpeed);
      } else if (tick >= m.attackReadyTick && std::fabs(d.y) < 2.5) {
        startAction(m, action::Swing);
        damagePlayer(*t, def.damage, m.id);
        m.attackReadyTick = tick + Tick(std::lround(def.attackCooldown * kSimHz));
      }
    } else {
      if (tick >= m.wanderUntil) {
        m.wanderTarget = m.move.pos + glm::dvec3(randi(-8, 8), 0, randi(-8, 8));
        m.wanderUntil = tick + Tick(randi(90, 180));
      }
      const glm::dvec3 d = m.wanderTarget - m.move.pos;
      if (d.x * d.x + d.z * d.z > 1.0) want = 0.5f * std::min(1.0f, def.speed / moveTuning().walkSpeed);
    }
    const glm::dvec3 d = goal - m.move.pos;
    if (d.x * d.x + d.z * d.z > 1e-4) in.yaw = float(std::atan2(-d.x, -d.z));
    else in.yaw = m.move.yaw;
    in.moveZ = want;

    // Stuck against a block: pulse jump (edge-triggered in stepMovement).
    const float hs = std::sqrt(m.move.vel.x * m.move.vel.x + m.move.vel.z * m.move.vel.z);
    if (want > 0.0f && hs < 0.3f * want * moveTuning().walkSpeed) m.stuckTime += kSimDt;
    else m.stuckTime = 0.0f;
    if (m.stuckTime > 0.25f && (tick & 1u)) in.buttons = button::Jump;

    stepMovement(m.move, in, *world, blocks, kSimDt);
    if (m.move.pos.y < -32.0) m.remove = true;
  }
}

void ServerState::simulateProjectiles() {
  constexpr int kSub = 4;
  const float dt = kSimDt / float(kSub);
  for (auto &[id, p] : entities) {
    if (p.kind != EntityKind::Projectile || p.remove) continue;
    if (tick >= p.expireTick) {
      p.remove = true;
      continue;
    }
    for (int s = 0; s < kSub && !p.remove; ++s) {
      p.move.vel.y -= p.gravity * dt;
      const glm::dvec3 next = p.move.pos + glm::dvec3(p.move.vel) * double(dt);
      const vx::BlockPos bp{int32_t(std::floor(next.x)), int32_t(std::floor(next.y)), int32_t(std::floor(next.z))};
      if (bp.y < 0 || bp.y >= vx::kWorldHeight || !world->isLoaded(vx::chunkOf(bp)) ||
          blocks.solid(world->blockAt(bp))) {
        p.remove = true;
        break;
      }
      p.move.pos = next;
      for (auto &[mid, m] : entities) {
        if (m.kind != EntityKind::Monster || m.dead) continue;
        const glm::dvec3 c = m.move.pos + glm::dvec3(0.0, 0.8, 0.0);
        const glm::dvec3 diff = c - p.move.pos;
        if (glm::dot(diff, diff) > 0.9 * 0.9) continue;
        Entity *oe = findEntity(p.owner);
        if (Player *op = oe ? playerOf(*oe) : nullptr) damageMonster(m, *op, p.damage, p.critical, p.style);
        p.remove = true; // PvP is off in the demo: players are not hit
        break;
      }
    }
    const float hv = std::sqrt(p.move.vel.x * p.move.vel.x + p.move.vel.z * p.move.vel.z);
    if (hv > 0.01f) p.move.yaw = std::atan2(-p.move.vel.x, -p.move.vel.z);
  }
}

void ServerState::simulateItems() {
  for (auto &[id, d] : entities) {
    if (d.kind != EntityKind::DroppedItem || d.remove) continue;
    if (tick >= d.expireTick) {
      d.remove = true;
      continue;
    }
    // Simple fall until resting on a solid block.
    const vx::BlockPos below{int32_t(std::floor(d.move.pos.x)), int32_t(std::floor(d.move.pos.y - 0.05)),
                             int32_t(std::floor(d.move.pos.z))};
    if (below.y >= 0 && world->isLoaded(vx::chunkOf(below)) && !blocks.solid(world->blockAt(below))) {
      d.move.vel.y = std::max(d.move.vel.y - 20.0f * kSimDt, -20.0f);
      d.move.pos.y += double(d.move.vel.y) * kSimDt;
    } else {
      d.move.vel.y = 0.0f;
    }
    // Pickup (personal until publicTick).
    for (auto &[key, pl] : players) {
      if (!pl.welcomed) continue;
      Entity *pe = findEntity(pl.entity);
      if (!pe || pe->dead) continue;
      if (d.owner != kNoEntity && d.owner != pe->id && tick < d.publicTick) continue;
      const glm::dvec3 diff = pe->move.pos - d.move.pos;
      if (glm::dot(diff, diff) > 1.6 * 1.6) continue;
      const uint16_t left = addItem(pl, d.stack.item, d.stack.count);
      if (left == 0) {
        d.remove = true;
        break;
      }
      d.stack.count = left;
    }
  }
}

} // namespace ao::server
