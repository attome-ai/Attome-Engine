// Zone server world rules (RuneScape-style overworld): the terrain can't be
// edited; resource nodes (data/resources.json) are harvested, deplete and
// regrow; NPCs come from fixed spawners (data/maps/overworld.json).

#include "ServerState.h"

#include <algorithm>
#include <cmath>

namespace ao::server {

using namespace ao::proto;
namespace vx = atm::voxel;

namespace {

void tell(ServerState &s, Player &pl, std::string text) {
  ChatMsg cm;
  cm.from = "Server";
  cm.text = std::move(text);
  s.sendTo(pl, cm, Channel::ReliableOrdered);
}

} // namespace

// ---------------------------------------------------------------------------
// Resource nodes
// ---------------------------------------------------------------------------

bool ServerState::harvestNode(Player &pl, Entity &pe, vx::BlockPos p, vx::BlockId block) {
  const ResourceDef *r = resourceForBlock(block);
  if (!r) {
    // Not a resource node: the overworld itself is not editable.
    tell(*this, pl, "You can't break that here.");
    return false;
  }
  const uint32_t level = skillLevel(pl, r->skill);
  if (level < r->level) {
    tell(*this, pl, "You need a " + std::string(skillName(r->skill)) + " level of " + std::to_string(r->level) +
                        " to harvest this " + std::string(r->display) + ".");
    return false;
  }
  if (r->tool != WeaponType::None && itemDef(heldWeapon(pl)).weapon != r->tool) {
    tell(*this, pl, std::string("You need ") + (r->tool == WeaponType::Axe ? "an axe" : "a pickaxe") +
                        " in your hand to harvest this " + std::string(r->display) + ".");
    return false;
  }
  // Gathering speed: the client's progress bar uses the same gatherTime();
  // allow some network jitter.
  pl.blockReadyTick = tick + Tick(std::lround(gatherTime(*r, level) * 0.75f * kSimHz));
  startAction(pe, action::Mine);
  giveOrDrop(pl, r->item, 1, glm::dvec3(p.x + 0.5, p.y + 0.5, p.z + 0.5), kNoEntity, false);
  addXp(pl, r->skill, double(r->xp));
  if (randf() < r->depleteChance) depleteNode(*r, p);
  return true;
}

void ServerState::depleteNode(const ResourceDef &r, vx::BlockPos p) {
  Regrowth g;
  g.at = tick + Tick(std::lround(r.respawnSeconds * kSimHz));
  auto change = [&](vx::BlockPos q, vx::BlockId to) {
    const vx::BlockId was = world->blockAt(q);
    if (!world->setBlock(q, to)) return;
    g.blocks.emplace_back(q, was);
    onBlockEdited(q, to);
  };

  if (!r.fells) { // rocks: the ore block turns into a depleted rock
    change(p, r.depleted);
    regrowths.push_back(std::move(g));
    return;
  }

  // Trees: the whole tree falls. Its logs are the connected log blocks (26
  // neighbours, near the chopped block); its leaves are the leaves within a
  // few steps of those logs. Logs standing on the ground become stumps.
  constexpr int kMaxLogs = 96, kMaxLeaves = 600, kLeafDepth = 4;
  std::vector<vx::BlockPos> logs{p}, open{p};
  auto isLoadedAt = [&](vx::BlockPos q) { return q.y >= 0 && q.y < vx::kWorldHeight && world->isLoaded(vx::chunkOf(q)); };
  auto seen = [](const std::vector<vx::BlockPos> &v, vx::BlockPos q) {
    return std::find_if(v.begin(), v.end(), [&](const vx::BlockPos &o) { return o.x == q.x && o.y == q.y && o.z == q.z; }) !=
           v.end();
  };
  while (!open.empty() && int(logs.size()) < kMaxLogs) {
    const vx::BlockPos c = open.back();
    open.pop_back();
    for (int dy = -1; dy <= 1; ++dy)
      for (int dz = -1; dz <= 1; ++dz)
        for (int dx = -1; dx <= 1; ++dx) {
          const vx::BlockPos q{c.x + dx, c.y + dy, c.z + dz};
          if (std::abs(q.x - p.x) > 6 || std::abs(q.z - p.z) > 6 || std::abs(q.y - p.y) > 24) continue;
          if (!isLoadedAt(q) || world->blockAt(q) != r.block || seen(logs, q)) continue;
          logs.push_back(q);
          open.push_back(q);
        }
  }
  std::vector<vx::BlockPos> leaves;
  std::vector<std::pair<vx::BlockPos, int>> frontier;
  for (const vx::BlockPos &l : logs) frontier.emplace_back(l, 0);
  static constexpr int kN[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (size_t i = 0; i < frontier.size() && int(leaves.size()) < kMaxLeaves; ++i) {
    const auto [c, depth] = frontier[i];
    if (depth >= kLeafDepth) continue;
    for (const auto &n : kN) {
      const vx::BlockPos q{c.x + n[0], c.y + n[1], c.z + n[2]};
      if (!isLoadedAt(q) || world->blockAt(q) != vx::blocks::OakLeaves || seen(leaves, q)) continue;
      leaves.push_back(q);
      frontier.emplace_back(q, depth + 1);
    }
  }
  for (const vx::BlockPos &q : leaves) change(q, vx::kAir);
  for (const vx::BlockPos &q : logs) {
    const vx::BlockPos below{q.x, q.y - 1, q.z};
    const vx::BlockId under = isLoadedAt(below) ? world->blockAt(below) : vx::kAir;
    const bool grounded = under != r.block && blocks.solid(under);
    change(q, grounded ? r.depleted : vx::kAir);
  }
  regrowths.push_back(std::move(g));
}

void ServerState::regrowNodes() {
  for (size_t i = 0; i < regrowths.size();) {
    Regrowth &g = regrowths[i];
    if (tick < g.at) {
      ++i;
      continue;
    }
    // Put everything back that is still missing (air or a depleted marker).
    // Chunks that aren't loaded right now are retried later.
    bool pending = false;
    for (auto it = g.blocks.begin(); it != g.blocks.end();) {
      const auto [q, orig] = *it;
      if (!world->isLoaded(vx::chunkOf(q))) {
        pending = true;
        ++it;
        continue;
      }
      const vx::BlockId now = world->blockAt(q);
      if (now != orig && (now == vx::kAir || now == vx::blocks::Stump || now == vx::blocks::DepletedRock) &&
          world->setBlock(q, orig))
        onBlockEdited(q, orig);
      it = g.blocks.erase(it);
    }
    if (pending) {
      g.at = tick + Tick(kSimHz * 5);
      ++i;
    } else {
      regrowths[i] = std::move(regrowths.back());
      regrowths.pop_back();
    }
  }
}

// ---------------------------------------------------------------------------
// Fixed NPC spawners
// ---------------------------------------------------------------------------

void ServerState::updateSpawners() {
  const auto &defs = mapSpawners();
  if (spawnerSlots.size() != defs.size()) {
    spawnerSlots.assign(defs.size(), {});
    for (size_t i = 0; i < defs.size(); ++i) spawnerSlots[i].resize(defs[i].count);
  }
  constexpr double kActiveRange = 90.0; // players this close wake a spawner up
  for (size_t si = 0; si < defs.size(); ++si) {
    const MapSpawner &sp = defs[si];
    // Slots whose NPC died wait for the respawn timer; despawned ones (no
    // players around) come back as soon as someone returns.
    for (SpawnerSlot &slot : spawnerSlots[si]) {
      if (slot.id == kNoEntity) continue;
      const Entity *m = findEntity(slot.id);
      bool queued = false;
      if (!m)
        for (const Entity &q : spawnQueue)
          if (q.id == slot.id) queued = true;
      if (queued) continue;
      if (!m || m->remove) {
        slot.id = kNoEntity; // despawned: ready now (readyTick untouched)
      } else if (m->dead) {
        slot.id = kNoEntity;
        slot.readyTick = tick + Tick(std::lround(sp.respawnSeconds * kSimHz));
      }
    }
    if (tick % Tick(kSimHz / 2) != Tick(si % (kSimHz / 2))) continue; // spread the checks
    bool active = false;
    for (auto &[key, pl] : players) {
      const Entity *pe = pl.welcomed ? findEntity(pl.entity) : nullptr;
      if (!pe) continue;
      const double dx = pe->move.pos.x - sp.x, dz = pe->move.pos.z - sp.z;
      if (dx * dx + dz * dz < kActiveRange * kActiveRange) {
        active = true;
        break;
      }
    }
    if (!active) continue;
    for (SpawnerSlot &slot : spawnerSlots[si]) {
      if (slot.id != kNoEntity || tick < slot.readyTick) continue;
      // A random spot in the spawn area with open ground.
      const double a = double(randf()) * 6.283185307179586, d = std::sqrt(double(randf())) * sp.radius;
      const int32_t x = int32_t(std::floor(sp.x + std::cos(a) * d));
      const int32_t z = int32_t(std::floor(sp.z + std::sin(a) * d));
      int32_t gy = 0;
      if (!findGround(x, z, gy)) continue;
      const vx::BlockId ground = world->blockAt({x, gy, z});
      if (blocks.get(ground).liquid || ground == vx::blocks::OakLeaves) continue;
      const MonsterDef &def = monsterDef(sp.npc);
      Entity &m = queueSpawn(EntityKind::Monster);
      m.type = sp.npc;
      m.move.pos = glm::dvec3(x + 0.5, gy + 1.02, z + 0.5);
      m.move.yaw = float(randf() * 6.2831853f - 3.1415927f);
      m.hp = m.maxHp = def.maxHp;
      m.wanderTarget = m.move.pos;
      m.wanderUntil = tick + Tick(randi(30, 120));
      m.spawner = int16_t(si);
      m.leash = sp.leash;
      m.home = glm::dvec3(sp.x + 0.5, m.move.pos.y, sp.z + 0.5);
      slot.id = m.id;
    }
  }
}

} // namespace ao::server
