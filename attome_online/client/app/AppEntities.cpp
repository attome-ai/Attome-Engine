// App: local / remote entity animation, particles, decoration, HUD timers.

#include "app/AppInternal.h"

namespace ao::client {

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

void App::updateEntities(float dt) {
  // Local player animation.
  const MoveState &me = prediction_.current();
  selectLocomotion(selfAnim_, me.vel,
                   uint8_t((me.gliding ? 1 : 0) | (me.inWater ? 2 : 0) | (me.onGround ? 8 : 0)),
                   true);
  selfAnim_.update(anims_, dt);

  for (auto &[id, r] : remotes_) {
    const RemoteSample s = r.track.sample(serverTickEstimate_ - kInterpDelayTicks);
    const int creature = r.last.kind == EntityKind::Monster ? creatureForMonster(r.last.type) : -1;
    if (r.last.kind == EntityKind::Player || r.last.kind == EntityKind::Monster)
      selectLocomotion(r.animator, s.vel, s.flags, false, creature);
    r.animator.update(anims_, dt);
    r.hitFlash = std::max(0.0f, r.hitFlash - dt);
    if (sfx_ && r.last.kind == EntityKind::Player && (s.flags & 8)) {
      r.stepDistance += glm::length(glm::vec2(s.vel.x, s.vel.z)) * dt;
      if (r.stepDistance > 2.1f) {
        r.stepDistance = 0.0f;
        const glm::dvec3 d = s.pos - prediction_.current().pos;
        if (d.x * d.x + d.y * d.y + d.z * d.z < 24.0 * 24.0)
          sfx_->playAt(GameSound::Footstep, float(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z)));
      }
    }
  }

  {
    ATM_PROFILE_SCOPE("Particles update");
    particles_.update(dt);
  }
  if (welcomed_)
    particles_.ambient(me.pos, dt);
  if (welcomed_ && world_) {
    {
      ATM_PROFILE_SCOPE("Decoration scatter");
      decor_.update(*world_, me.pos, dt);
    }
    {
      ATM_PROFILE_SCOPE("Map scan");
      mapCache_.update(*world_, blocks_, me.pos, 180, 0.15f, dt);
    }
    // Entering a new map region: banner + chat line (RuneScape-style).
    const MapRegion *reg = regionAt(me.pos.x, me.pos.z);
    if (reg && reg != lastRegion_) {
      if (lastRegion_) {
        std::string text = std::string(reg->name);
        if (!reg->levels.empty())
          text += "  -  levels " + std::string(reg->levels);
        banners_.push_back({text, 0.0f});
        addChatLine("You enter " + std::string(reg->name) + ".");
      }
      lastRegion_ = reg;
    }
  }

  // Floating texts, XP drops, banners age out.
  for (auto &f : floating_) f.age += dt;
  floating_.erase(std::remove_if(floating_.begin(), floating_.end(),
                                 [](const FloatingText &f) { return f.age >= f.life; }),
                  floating_.end());
  for (auto &x : xpDrops_) x.age += dt;
  xpDrops_.erase(std::remove_if(xpDrops_.begin(), xpDrops_.end(),
                                [](const XpDrop &x) { return x.age >= 1.8f; }),
                 xpDrops_.end());
  for (auto &b : banners_) b.age += dt;
  banners_.erase(std::remove_if(banners_.begin(), banners_.end(),
                                [](const Banner &b) { return b.age >= 3.5f; }),
                 banners_.end());
}


} // namespace ao::client
