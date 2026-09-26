// App: the frame (environment, models, debug views) and HUD text feeds.

#include "app/AppInternal.h"

#include <chrono>

namespace ao::client {

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void App::render(float alpha, float dt) {
  // Look settings from the graphics panel (F10); sun from azimuth/elevation.
  cfg_.render.fovYDegrees = look_.fov;
  renderer_.setVsync(look_.vsync);
  decor_.setDensity(look_.decorDensity);
  atm::render::Environment env = look_.env;
  {
    const float az = glm::radians(look_.sunAzimuth), el = glm::radians(look_.sunElevation);
    env.sunDirection = glm::vec3(std::sin(az) * std::cos(el), std::sin(el), -std::cos(az) * std::cos(el));
  }
  if (world_) {
    const atm::voxel::BlockPos cb{int32_t(std::floor(camPos_.x)), int32_t(std::floor(camPos_.y)),
                                  int32_t(std::floor(camPos_.z))};
    env.underwater = blocks_.get(world_->blockAt(cb)).liquid;
  }

  bool frameOk = false;
  {
    ATM_PROFILE_SCOPE("Renderer: begin frame (GPU wait)");
    frameOk = renderer_.beginFrame(camera_, env);
  }
  if (!frameOk) {
    SDL_Delay(10); // minimised: don't spin; ImGui::NewFrame is skipped too
    return;
  }
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();

  if (welcomed_ && world_) {
    // Local player.
    const MoveState &me = prediction_.current();
    // Visual body yaw (Trove-style): turn toward the movement direction; face
    // the aim while attacking / mining; keep the last facing when idle, so
    // the camera can orbit around to the character's face.
    {
      const float hs = std::sqrt(me.vel.x * me.vel.x + me.vel.z * me.vel.z);
      float target = bodyYaw_;
      if (attackCooldown_ > 0.05f || mineProgress_ > 0.0f)
        target = camYaw_;
      else if (hs > 0.6f)
        target = std::atan2(-me.vel.x, -me.vel.z);
      float d = std::fmod(target - bodyYaw_ + 3.14159265f, 6.28318531f);
      if (d < 0.0f) d += 6.28318531f;
      d -= 3.14159265f;
      bodyYaw_ += d * std::min(1.0f, dt * 14.0f);
    }
    // Hide our own model when the camera is pushed right against it (tight
    // spaces, underwater): otherwise it fills the screen as a dark wall.
    const glm::dvec3 selfFeet = prediction_.renderPosition(alpha);
    if (glm::length(camPos_ - (selfFeet + glm::dvec3(0.0, 1.1, 0.0))) > 1.2)
      drawCharacter(selfAppearance_, selfAnim_, selfFeet, bodyYaw_,
                    hp_ == 0 ? rgba(120, 120, 120) : 0xFFFFFFFFu);
    drawStructures();
    drawWorldProps();

    ATM_PROFILE_SCOPE("Entities draw");
    // Remote entities.
    const float renderTick = serverTickEstimate_ - kInterpDelayTicks;
    for (auto &[id, r] : remotes_) {
      if (r.track.empty())
        continue; // known only from an AppearanceMsg so far: no position yet
      const RemoteSample s = r.track.sample(renderTick);
      const uint32_t tint = r.hitFlash > 0.0f ? rgba(255, 120, 120) : 0xFFFFFFFFu;
      switch (r.last.kind) {
      case EntityKind::Player:
        drawCharacter(r.appearance, r.animator, s.pos, s.yaw, tint);
        break;
      case EntityKind::Monster:
        drawMonster(r.last.type, r.animator, s.pos, s.yaw, tint);
        break;
      case EntityKind::DroppedItem: {
        drawItemEntity(r.last.item, s.pos, renderTick * 0.08f);
        // Loot beam for rare and better drops (RuneLite-style), pulsing.
        const ItemDef &idef = itemDef(r.last.item);
        if (idef.rarity >= Rarity::Rare && lootBeamMesh_ != atm::render::kInvalidModelMesh) {
          const float pulse = 0.75f + 0.25f * std::sin(renderTick * 0.35f + float(id % 7));
          const ImU32 c = ui::rarityColor(idef.rarity); // bytes R,G,B,A like model tints
          auto ch = [&](int sh) { return uint32_t(float((c >> sh) & 0xFF) * pulse); };
          atm::render::ModelInstance beam;
          beam.mesh = lootBeamMesh_;
          beam.origin = s.pos;
          beam.pivot = {0.5f, 0.0f, 0.5f};
          beam.voxelScale = idef.rarity >= Rarity::Epic ? 0.16f : 0.12f;
          beam.tint = ch(0) | (ch(8) << 8) | (ch(16) << 16) | 0xFF000000u;
          beam.flags = atm::render::kInstanceNoRim | atm::render::kInstanceNoShadow;
          renderer_.drawModel(beam);
        }
        break;
      }
      case EntityKind::Projectile: {
        drawProjectile(s.pos, s.vel);
        // Trove-style glow: a bright head plus a spark trail spawned by
        // distance travelled (same density at any frame rate).
        const bool magic = r.last.type == uint8_t(WeaponType::Staff);
        const uint32_t glow = magic ? rgba(190, 120, 255) : rgba(255, 214, 110);
        if (atm::voxel::blocks::Lamp < blockItemMeshes_.size()) {
          atm::render::ModelInstance head;
          head.mesh = blockItemMeshes_[atm::voxel::blocks::Lamp];
          head.origin = s.pos;
          head.pivot = {2.0f, 2.0f, 2.0f};
          head.voxelScale = (magic ? 0.34f : 0.18f) / 4.0f;
          head.rotation = glm::angleAxis(renderTick * 0.6f, glm::normalize(glm::vec3(1.0f, 1.0f, 0.3f)));
          head.tint = glow;
          head.flags = atm::render::kInstanceNoRim | atm::render::kInstanceNoShadow;
          renderer_.drawModel(head);
        }
        if (r.trailFrom.x > 1e29)
          r.trailFrom = s.pos;
        const glm::dvec3 seg = s.pos - r.trailFrom;
        const double len = glm::length(seg);
        const double step = 0.12;
        const int n = std::min(int(len / step), 16);
        for (int i = 1; i <= n; ++i)
          particles_.trail(r.trailFrom + seg * (double(i) * step / len), glow, magic ? 0.12f : 0.08f);
        if (n > 0)
          r.trailFrom += seg * (double(n) * step / len);
        break;
      }
      }
    }

    {
      ATM_PROFILE_SCOPE("Decoration draw");
      decor_.draw(renderer_);
    }
    {
      ATM_PROFILE_SCOPE("Particles draw");
      particles_.draw(renderer_, blockItemMeshes_);
    }

    // F8 hitbox view: the volumes the server actually tests, at the positions
    // this client renders. Green = body (0.6 x 1.8 collision box), red =
    // monster hit zone (bounds of its per-type hit capsule),
    // yellow = projectile (a point) plus a marker along its velocity.
    if (showHitboxes_) {
      const MoveTuning mt = moveTuning();
      const glm::dvec3 hw(mt.halfWidth, 0.0, mt.halfWidth);
      auto body = [&](const glm::dvec3 &feet, uint32_t col) {
        renderer_.drawDebugBox(feet - hw, feet + hw + glm::dvec3(0.0, mt.height, 0.0), col);
      };
      body(prediction_.renderPosition(alpha), rgba(80, 255, 120));
      for (auto &[id, r] : remotes_) {
        if (r.track.empty() || (r.last.flags & 4u))
          continue;
        const RemoteSample s = r.track.sample(renderTick);
        if (r.last.kind == EntityKind::Player || r.last.kind == EntityKind::Monster) {
          body(s.pos, rgba(80, 255, 120));
          if (r.last.kind == EntityKind::Monster) {
            // Bounds of the per-type hit capsule.
            const MonsterHitShape hs = monsterHitShape(r.last.type);
            renderer_.drawDebugBox(s.pos + glm::dvec3(-hs.radius, hs.bottom - hs.radius, -hs.radius),
                                   s.pos + glm::dvec3(hs.radius, hs.top + hs.radius, hs.radius),
                                   rgba(255, 70, 60));
          }
        } else if (r.last.kind == EntityKind::Projectile) {
          renderer_.drawDebugBox(s.pos - glm::dvec3(0.1), s.pos + glm::dvec3(0.1), rgba(255, 230, 60));
          const float sp = glm::length(s.vel);
          if (sp > 0.01f) {
            const glm::dvec3 ahead = s.pos + glm::dvec3(s.vel / sp) * 0.6;
            renderer_.drawDebugBox(ahead - glm::dvec3(0.04), ahead + glm::dvec3(0.04), rgba(255, 230, 60));
          }
        } else if (r.last.kind == EntityKind::DroppedItem) {
          renderer_.drawDebugBox(s.pos - glm::dvec3(0.2, 0.0, 0.2), s.pos + glm::dvec3(0.2, 0.4, 0.2),
                                 rgba(120, 200, 255));
        }
      }
    }

    // Block outline; skipped when the block nearly touches the camera (its
    // edges would stretch across the whole screen as long diagonal lines).
    if (hasTarget_ && resourceForBlock(world_->blockAt(targetBlock_))) { // RuneScape-style: only gatherables
      const glm::dvec3 bc(targetBlock_.x + 0.5, targetBlock_.y + 0.5, targetBlock_.z + 0.5);
      if (glm::length(bc - camPos_) > 1.6)
        renderer_.drawBlockHighlight(targetBlock_);
    }
  }

  {
    ATM_PROFILE_SCOPE("HUD (ImGui build)");
    drawHud(dt);
  }
  {
    ATM_PROFILE_SCOPE("Renderer: submit frame");
    renderer_.endFrame();
  }
}

void App::addFloatingText(const glm::dvec3 &at, std::string text, uint32_t color) {
  if (floating_.size() > 64)
    floating_.erase(floating_.begin());
  floating_.push_back({at, std::move(text), color, 0.0f, 1.2f});
}

void App::addChatLine(std::string line) {
  chat_.push_back(std::move(line));
  chatIdle_ = 0.0f;
  while (chat_.size() > 50)
    chat_.pop_front();
}


} // namespace ao::client

namespace ao::client {

// ---------------------------------------------------------------------------
// Decoration props
// ---------------------------------------------------------------------------

void App::buildWorldProps() {
  worldProps_.clear();
  clearStructures();
  if (!world_) return;
  const auto &town = ao::world::homeTown(world_->generator());
  // Queue every 32^3 tile of every structure, nearest to the spawn first.
  const auto &models = ao::world::townStructures(town);
  for (size_t i = 0; i < models.size(); ++i) {
    const auto &m = models[i];
    for (int ty = 0; ty * 32 < m.sy; ++ty)
      for (int tz = 0; tz * 32 < m.sz; ++tz)
        for (int tx = 0; tx * 32 < m.sx; ++tx) pendingTiles_.push_back({int(i), tx, ty, tz});
  }
  std::stable_sort(pendingTiles_.begin(), pendingTiles_.end(), [&](const PendingTile &a, const PendingTile &b) {
    auto d = [&](const PendingTile &p) {
      const auto &m = models[size_t(p.model)];
      const double x = m.bx + p.tx * 8 + 4, z = m.bz + p.tz * 8 + 4;
      return x * x + z * z;
    };
    return d(a) < d(b);
  });
  for (const ao::world::TownProp &p : ao::world::townProps(town)) {
    const int part = models_.findPart(std::string("prop_") + p.name);
    if (part < 0) continue;
    PlacedProp w;
    w.part = part;
    w.pos = glm::dvec3(town.centerX + double(p.x), town.groundY + double(p.y), town.centerZ + double(p.z));
    w.yaw = p.yaw;
    worldProps_.push_back(w);
  }
}

void App::drawWorldProps() {
  ATM_PROFILE_SCOPE("Props draw");
  constexpr double kRange = 110.0; // blocks; small props, same idea as Decor's radius
  for (const PlacedProp &p : worldProps_) {
    const glm::dvec3 d = p.pos - camPos_;
    if (d.x * d.x + d.z * d.z > kRange * kRange) continue;
    if (size_t(p.part) >= partMeshes_.size() || partMeshes_[size_t(p.part)] == atm::render::kInvalidModelMesh)
      continue;
    atm::render::ModelInstance inst;
    inst.mesh = partMeshes_[size_t(p.part)];
    inst.origin = p.pos;
    inst.pivot = models_.parts()[size_t(p.part)].pivot;
    inst.rotation = glm::angleAxis(p.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    inst.voxelScale = 1.0f / 16.0f;
    inst.flags = atm::render::kInstanceNoRim;
    renderer_.drawModel(inst);
  }
}

} // namespace ao::client

namespace ao::client {

// ---------------------------------------------------------------------------
// Fine-voxel structures
// ---------------------------------------------------------------------------

void App::clearStructures() {
  for (const StructureTile &t : structureTiles_)
    if (t.mesh != atm::render::kInvalidModelMesh) renderer_.destroyModelMesh(t.mesh);
  structureTiles_.clear();
  pendingTiles_.clear();
  pendingTileCursor_ = 0;
}

void App::buildStructureMeshes(float budgetMs) {
  if (pendingTileCursor_ >= pendingTiles_.size() || !world_) return;
  ATM_PROFILE_SCOPE("Structure meshing");
  const auto t0 = std::chrono::steady_clock::now();
  const auto &town = ao::world::homeTown(world_->generator());
  const auto &models = ao::world::townStructures(town);
  constexpr int T = 32; // voxels per tile side (mesher limit)
  atm::model::VoxelPart part;
  part.palette = ao::world::microPalette();
  part.emissiveFrom = ao::world::kMicroGlowFrom;
  atm::voxel::ChunkMeshData mesh;
  while (pendingTileCursor_ < pendingTiles_.size()) {
    const PendingTile p = pendingTiles_[pendingTileCursor_++];
    const ao::world::MicroModel &m = models[size_t(p.model)];
    const int x0 = p.tx * T, y0 = p.ty * T, z0 = p.tz * T;
    part.sx = std::min(T, m.sx - x0), part.sy = std::min(T, m.sy - y0), part.sz = std::min(T, m.sz - z0);
    part.voxels.assign(size_t(part.sx) * part.sy * part.sz, 0);
    bool any = false;
    for (int y = 0; y < part.sy; ++y)
      for (int z = 0; z < part.sz; ++z)
        for (int x = 0; x < part.sx; ++x) {
          const uint8_t c = m.at(x0 + x, y0 + y, z0 + z);
          part.at(x, y, z) = c;
          any |= c != 0;
        }
    if (any) {
      mesh.clear();
      atm::model::meshPart(part, uint16_t(microMaterialBase_), mesh);
      if (!mesh.empty()) {
        StructureTile t;
        t.mesh = renderer_.createModelMesh(mesh);
        const double s = 1.0 / ao::world::kMicro;
        t.origin = glm::dvec3(town.centerX + m.bx + x0 * s, town.groundY + m.by + y0 * s, town.centerZ + m.bz + z0 * s);
        t.center = t.origin + glm::dvec3(part.sx, part.sy, part.sz) * (0.5 * s);
        t.radius = 0.5 * s * std::sqrt(double(part.sx * part.sx + part.sy * part.sy + part.sz * part.sz));
        structureTiles_.push_back(t);
      }
    }
    const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms > budgetMs) break;
  }
}

void App::drawStructures() {
  buildStructureMeshes(3.0f);
  ATM_PROFILE_SCOPE("Structures draw");
  const glm::dvec3 fwd = glm::dvec3(aimDirection());
  constexpr double kRange = 280.0;
  for (const StructureTile &t : structureTiles_) {
    const glm::dvec3 d = t.center - camPos_;
    const double dist2 = glm::dot(d, d);
    if (dist2 > (kRange + t.radius) * (kRange + t.radius)) continue;
    if (glm::dot(d, fwd) < -t.radius) continue; // behind the camera
    atm::render::ModelInstance inst;
    inst.mesh = t.mesh;
    inst.origin = t.origin;
    inst.pivot = glm::vec3(0.0f);
    inst.voxelScale = 1.0f / float(ao::world::kMicro);
    inst.flags = atm::render::kInstanceNoRim;
    renderer_.drawModel(inst);
  }
}

} // namespace ao::client
