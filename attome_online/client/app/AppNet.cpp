// App: server message handlers (world data, snapshots, combat, loot, chat).

#include "app/AppInternal.h"

namespace ao::client {

// ---------------------------------------------------------------------------
// Network handlers
// ---------------------------------------------------------------------------

void App::onConnected() {
  status_ = "Connected, joining...";
  proto::Hello h;
  h.schemaHash = proto::kSchemaHash;
  h.name = cfg_.name;
  h.appearance = selfAppearance_;
  net_.send(h, atm::net2::Channel::ReliableOrdered);
}

void App::onDisconnected(atm::net2::DisconnectReason reason) {
  static const char *names[] = {"none", "timeout", "requested", "rejected",
                                "protocol mismatch", "server full", "too many retries", "backlog"};
  const unsigned r = unsigned(reason);
  status_ = std::string("Disconnected: ") + (r < 8 ? names[r] : "unknown");
  addChatLine(status_);
  welcomed_ = false;
}

void App::onWelcome(const proto::Welcome &m) {
  if (m.schemaHash != proto::kSchemaHash) {
    status_ = "Server runs a different game version";
    net_.disconnect();
    return;
  }
  selfId_ = m.playerEntity;
  worldSeed_ = m.worldSeed;
  clientTick_ = m.serverTick;
  newestSnapshotTick_ = m.serverTick;
  serverTickEstimate_ = float(m.serverTick);

  atm::voxel::VoxelWorldConfig wc;
  wc.seed = m.worldSeed;
  wc.meshing = true;
  wc.viewRadiusChunks = cfg_.viewRadiusChunks;
  world_ = std::make_unique<atm::voxel::VoxelWorld>(wc, blocks_);

  MoveState s;
  s.pos = m.spawn;
  prediction_.reset(s);
  camYaw_ = 0.0f;
  welcomed_ = true;
  status_.clear();
  addChatLine("Welcome to Attome Online! WASD move, Space jump/glide, Q dash, LMB attack/mine, RMB place/use, E pick up loot, Tab inventory, K skills.");
}

void App::onChunkData(const proto::ChunkData &m) {
  if (!world_)
    return;
  auto chunk = std::make_shared<atm::voxel::Chunk>();
  if (!chunk->deserialize(m.data)) {
    SDL_Log("[client] bad chunk data for %d,%d,%d", m.cx, m.cy, m.cz);
    return;
  }
  world_->setChunkData(ChunkCoord{m.cx, m.cy, m.cz}, std::move(chunk));
}

void App::onEditedChunks(const proto::EditedChunks &m) {
  if (world_)
    world_->setExpectedEdited(m.keys);
}

void App::onBlockChanged(const proto::BlockChanged &m) {
  if (!world_)
    return;
  const BlockPos p{m.x, m.y, m.z};
  const auto old = world_->blockAt(p);
  world_->setBlock(p, m.block);
  if (m.block == atm::voxel::kAir && old != atm::voxel::kAir) {
    // The block shatters into chunks of itself.
    Particles::Burst deb;
    deb.count = 18;
    deb.block = old;
    deb.speed = 3.5f;
    deb.up = 3.5f;
    deb.size = 0.17f;
    deb.life = 0.9f;
    deb.gravity = 20.0f;
    deb.spread = 0.45f;
    particles_.burst(glm::dvec3(p.x + 0.5, p.y + 0.5, p.z + 0.5), deb);
  }
  if (sfx_) {
    const glm::dvec3 d = glm::dvec3(p.x + 0.5, p.y + 0.5, p.z + 0.5) - prediction_.current().pos;
    const float dist = float(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z));
    if (dist < 32.0f) {
      if (m.block == atm::voxel::kAir)
        sfx_->playBlockBreak(old, dist);
      else
        sfx_->playBlockPlace(m.block, dist);
    }
  }
}

void App::onSnapshot(const proto::SnapshotMsg &m) {
  const Snapshot &s = m.snapshot;
  if (s.tick < newestSnapshotTick_)
    return; // out of order
  newestSnapshotTick_ = s.tick;

  if (world_)
    prediction_.reconcile(s.self.move, s.ackInputSeq, *world_, blocks_);
  if (s.self.hp < hp_ && hp_ > 0 && sfx_)
    sfx_->play(GameSound::PlayerHurt);
  hp_ = s.self.hp;
  maxHp_ = std::max<uint16_t>(1, s.self.maxHp);

  for (const EntityState &in : s.entities) {
    if (in.id == selfId_)
      continue;
    auto [it, inserted] = remotes_.try_emplace(in.id);
    RemoteEntity &r = it->second;
    r.id = in.id;
    EntityState merged = inserted ? EntityState{} : r.last;
    if (in.mask & field::Type) { merged.kind = in.kind; merged.type = in.type; merged.item = in.item; }
    if (in.mask & field::Pos) merged.pos = in.pos;
    if (in.mask & field::Vel) merged.vel = in.vel;
    if (in.mask & field::Yaw) merged.yaw = in.yaw;
    if (in.mask & field::Anim) {
      merged.locoAnim = in.locoAnim;
      merged.actionAnim = in.actionAnim;
      merged.actionSeq = in.actionSeq;
    }
    if (in.mask & field::Health) { merged.hp = in.hp; merged.maxHp = in.maxHp; }
    if (in.mask & field::Flags) merged.flags = in.flags;
    merged.id = in.id;

    if (!inserted && merged.actionSeq != r.lastActionSeq && merged.actionAnim != act::None) {
      WeaponType w = WeaponType::None;
      int creature = -1;
      if (merged.kind == EntityKind::Player) {
        const auto piece = r.appearance.pieces[size_t(atm::model::EquipSlot::MainHand)];
        w = WeaponType(models_.piece(piece).weaponType);
      } else if (merged.kind == EntityKind::Monster) {
        creature = creatureForMonster(merged.type);
      }
      playActionAnim(r.animator, merged.actionAnim, w, creature);
    }
    r.lastActionSeq = merged.actionSeq;
    r.last = merged;
    r.track.push({float(s.tick), merged.pos, merged.vel, merged.yaw, merged.flags});
  }
  for (EntityId gone : s.removed) {
    auto it = remotes_.find(gone);
    if (it != remotes_.end() && it->second.last.kind == EntityKind::Projectile) {
      // Impact burst where the arrow / bolt ended (hit a monster or a wall).
      const bool magic = it->second.last.type == uint8_t(WeaponType::Staff);
      Particles::Burst hit;
      hit.block = atm::voxel::blocks::Lamp;
      hit.tint = magic ? rgba(200, 140, 255) : rgba(255, 220, 120);
      hit.count = magic ? 26 : 16;
      hit.speed = magic ? 6.0f : 4.5f;
      hit.up = 1.5f;
      hit.size = magic ? 0.1f : 0.08f;
      hit.life = 0.4f;
      hit.gravity = magic ? 0.0f : 10.0f;
      hit.drag = 3.0f;
      particles_.burst(it->second.last.pos, hit);
    }
    remotes_.erase(gone);
  }
}

void App::onAppearance(const proto::AppearanceMsg &m) {
  if (m.entity == selfId_) {
    selfAppearance_ = m.appearance;
    return;
  }
  RemoteEntity &r = remotes_[m.entity];
  r.id = m.entity;
  r.appearance = m.appearance;
  r.name = m.name;
}

void App::onDamage(const proto::DamageEvent &m) {
  glm::dvec3 at = prediction_.current().pos + glm::dvec3(0, 2.2, 0);
  if (m.target != selfId_) {
    auto it = remotes_.find(m.target);
    if (it == remotes_.end())
      return;
    at = it->second.last.pos + glm::dvec3(0, 2.0, 0);
    it->second.hitFlash = 0.18f;
    if (m.killed) {
      const int creature = it->second.last.kind == EntityKind::Monster
                               ? creatureForMonster(it->second.last.type)
                               : -1;
      playActionAnim(it->second.animator, act::Death, WeaponType::None, creature);
    }
  }
  const uint32_t color = m.target == selfId_ ? rgba(255, 80, 80)
                         : m.critical        ? rgba(255, 200, 40)
                                             : rgba(255, 255, 255);
  // Spread numbers so rapid hits do not stack on top of each other.
  const float jx = float(int(m.amount * 7919u + floating_.size() * 104729u) % 100) / 100.0f - 0.5f;
  const float jz = float(int(m.amount * 104729u + floating_.size() * 7919u) % 100) / 100.0f - 0.5f;
  at += glm::dvec3(jx * 0.7, 0.0, jz * 0.7);
  addFloatingText(at, (m.critical ? std::to_string(m.amount) + "!" : std::to_string(m.amount)), color);
  if (sfx_ && m.source == selfId_)
    sfx_->play(GameSound::MonsterHit);
  // Impact VFX: glowing sparks + chunks in the victim's colour, a big burst
  // on kills, red sparks when we are the one getting hit.
  {
    const bool self = m.target == selfId_;
    uint32_t body = rgba(200, 200, 205);
    if (!self) {
      auto it = remotes_.find(m.target);
      if (it != remotes_.end() && it->second.last.kind == EntityKind::Monster) {
        switch (it->second.last.type) {
        case monsters::Slime: body = rgba(110, 220, 110); break;
        case monsters::Wolf: body = rgba(150, 150, 160); break;
        case monsters::Golem: body = rgba(150, 140, 128); break;
        default: break;
        }
      }
    }
    const glm::dvec3 c = at - glm::dvec3(0.0, self ? 1.2 : 1.0, 0.0);
    Particles::Burst sp;
    sp.block = atm::voxel::blocks::Lamp;
    sp.tint = self ? rgba(255, 70, 60) : (m.critical ? rgba(255, 190, 60) : rgba(255, 240, 200));
    sp.count = m.critical ? 26 : 14;
    sp.speed = m.critical ? 9.0f : 6.0f;
    sp.up = 1.5f;
    sp.size = 0.08f;
    sp.life = 0.35f;
    sp.gravity = 8.0f;
    sp.drag = 3.0f;
    particles_.burst(c, sp);
    if (!self) {
      Particles::Burst ch;
      ch.block = atm::voxel::blocks::Snow;
      ch.tint = body;
      ch.count = m.killed ? 40 : 7;
      ch.speed = m.killed ? 6.0f : 3.5f;
      ch.up = m.killed ? 5.0f : 3.0f;
      ch.size = m.killed ? 0.2f : 0.13f;
      ch.life = m.killed ? 1.1f : 0.6f;
      ch.gravity = 20.0f;
      ch.spread = m.killed ? 0.6f : 0.3f;
      particles_.burst(c, ch);
      if (m.killed) { // poof
        Particles::Burst poof;
        poof.block = atm::voxel::blocks::Snow;
        poof.tint = rgba(240, 240, 240);
        poof.count = 24;
        poof.speed = 2.0f;
        poof.up = 1.5f;
        poof.size = 0.28f;
        poof.life = 0.7f;
        poof.gravity = -1.0f;
        poof.drag = 2.5f;
        poof.spread = 0.5f;
        particles_.burst(c, poof);
      }
    }
  }

  // Feel: a small kick when our hit lands, more on crits and when we get hit.
  if (m.source == selfId_)
    shake_ = std::max(shake_, m.critical ? 0.9f : 0.45f);
  if (m.target == selfId_)
    shake_ = std::max(shake_, 1.0f);
}

void App::onInventory(const proto::InventoryMsg &m) {
  for (size_t i = 0; i < inventory_.size(); ++i)
    inventory_[i] = i < m.slots.size() ? m.slots[i] : ItemStack{};
  for (size_t i = 0; i < equipped_.size(); ++i)
    equipped_[i] = i < m.equipped.size() ? m.equipped[i] : 0;
  // Local appearance follows equipment immediately.
  for (size_t i = 0; i < equipped_.size() && i < selfAppearance_.pieces.size(); ++i) {
    const ItemDef &d = itemDef(ItemId(equipped_[i]));
    selfAppearance_.pieces[i] = d.piece ? models_.findPiece(d.piece) : atm::model::kNoPiece;
  }
}

void App::onXpGain(const proto::XpGain &m) {
  if (m.skill >= kSkillCount)
    return;
  const uint32_t before = levelForXp(skillXp_[m.skill]);
  skillXp_[m.skill] = m.totalXp;
  const uint32_t after = levelForXp(m.totalXp);
  xpDrops_.push_back({Skill(m.skill), m.amount, 0.0f});
  if (after > before) {
    // Golden level-up fountain around the player.
    Particles::Burst lb;
    lb.count = 60;
    lb.block = atm::voxel::blocks::Lamp;
    lb.tint = rgba(255, 214, 90);
    lb.speed = 3.0f;
    lb.up = 7.0f;
    lb.size = 0.1f;
    lb.life = 1.3f;
    lb.gravity = 6.0f;
    lb.spread = 0.8f;
    particles_.burst(prediction_.current().pos + glm::dvec3(0.0, 0.3, 0.0), lb);
    banners_.push_back({"Congratulations! " + std::string(skillName(Skill(m.skill))) +
                            " level " + std::to_string(after),
                        0.0f});
    addChatLine("You advanced a " + std::string(skillName(Skill(m.skill))) + " level: " +
                std::to_string(after));
    if (sfx_)
      sfx_->play(GameSound::LevelUp);
  }
}

void App::onLoot(const proto::LootMsg &m) {
  const ItemDef &d = itemDef(m.item);
  addChatLine(std::string(m.rare ? "RARE DROP: " : "Loot: ") + std::string(d.display) + " x" +
              std::to_string(m.count));
  if (m.rare)
    banners_.push_back({"Rare drop: " + std::string(d.display) + "!", 0.0f});
  if (sfx_)
    sfx_->play(GameSound::Pickup);
}

void App::onChat(const proto::ChatMsg &m) { addChatLine(m.from + ": " + m.text); }


} // namespace ao::client
