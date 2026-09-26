// App: world streaming, camera, screen projection.

#include "app/AppInternal.h"

namespace ao::client {

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------

void App::updateWorld() {
  if (!world_)
    return;
  const glm::dvec3 p = prediction_.current().pos;
  world_->clearFoci();
  world_->addFocus(p.x, p.y, p.z);
  world_->update();

  meshResults_.clear();
  world_->takeMeshResults(meshResults_);
  for (const auto &r : meshResults_)
    renderer_.setChunkMesh(r.coord, r.mesh);

  unloaded_.clear();
  world_->takeUnloaded(unloaded_);
  for (const auto &c : unloaded_)
    renderer_.removeChunk(c);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

glm::dvec3 App::eyePosition(float alpha) const {
  glm::dvec3 p = prediction_.renderPosition(alpha);
  p.y += moveTuning().eyeHeight;
  return p;
}

glm::vec3 App::aimDirection() const { return lookDir(camYaw_, camPitch_); }

glm::vec3 App::crosshairAim(const glm::dvec3 &from) const {
  const glm::dvec3 d(aimDirection());
  double best = 150.0; // nothing hit: aim at a far point along the crosshair
  // Blocks: nearest point of the first solid block on the camera ray.
  if (world_) {
    BlockPos hit{};
    FaceDir face = FaceDir::PosY;
    if (raycastBlock(*world_, blocks_, camPos_, aimDirection(), float(best), hit, face)) {
      const glm::dvec3 c(hit.x + 0.5, hit.y + 0.5, hit.z + 0.5);
      best = std::max(0.5, glm::dot(c - camPos_, d) - 0.5);
    }
  }
  // Monsters / players: closest one the ray passes through (body ~0.9 wide).
  const float renderTick = serverTickEstimate_ - kInterpDelayTicks;
  for (const auto &[id, r] : remotes_) {
    if ((r.last.kind != EntityKind::Monster && r.last.kind != EntityKind::Player) || (r.last.flags & 4u) ||
        r.track.empty())
      continue;
    const glm::dvec3 feet = r.track.sample(renderTick).pos;
    if (r.last.kind == EntityKind::Monster) {
      // Per-type hit capsule (same test as the server).
      const MonsterHitShape hs = monsterHitShape(r.last.type);
      const glm::dvec3 c = feet + glm::dvec3(0.0, 0.5 * (hs.bottom + hs.top), 0.0);
      const double t = glm::dot(c - camPos_, d);
      if (t <= 0.0 || t >= best)
        continue;
      const glm::dvec3 q = camPos_ + d * t;
      if (monsterHitDistance2(r.last.type, feet.x, feet.y, feet.z, q.x, q.y, q.z) < double(hs.radius) * hs.radius)
        best = t;
      continue;
    }
    const glm::dvec3 c = feet + glm::dvec3(0.0, 0.9, 0.0);
    const double t = glm::dot(c - camPos_, d);
    if (t <= 0.0 || t >= best)
      continue;
    if (glm::length(c - (camPos_ + d * t)) < 0.9)
      best = t;
  }
  const glm::dvec3 target = camPos_ + d * best;
  const glm::dvec3 dir = target - from;
  const double len = glm::length(dir);
  return len > 1e-4 ? glm::vec3(dir / len) : glm::vec3(d);
}

void App::updateCamera(float dt) {
  const float alpha = float(accumulator_ / kSimDt);
  const glm::dvec3 target = eyePosition(alpha) + glm::dvec3(0.0, 0.4, 0.0);
  const glm::vec3 fwd = aimDirection();
  const glm::vec3 right{std::cos(camYaw_), 0.0f, -std::sin(camYaw_)};

  // Over-the-shoulder offset, then pull in when blocks are in the way.
  const glm::dvec3 shoulder = target + glm::dvec3(right) * 0.95; // over the shoulder: crosshair clears the character
  float dist = camDistance_;
  if (world_) {
    const glm::dvec3 back = -glm::dvec3(fwd);
    const float step = 0.2f;
    for (float t = 0.3f; t <= camDistance_; t += step) {
      const glm::dvec3 q = shoulder + back * double(t);
      const atm::voxel::BlockPos bp{int32_t(std::floor(q.x)), int32_t(std::floor(q.y)),
                                    int32_t(std::floor(q.z))};
      if (blocks_.solid(world_->blockAt(bp))) {
        dist = std::max(0.3f, t - 0.35f);
        break;
      }
    }
  }
  camPos_ = shoulder - glm::dvec3(fwd) * double(dist);

  // Impact shake (hits dealt / taken): decaying, smooth pseudo-random jitter.
  if (shake_ > 0.0f) {
    const float t = float(SDL_GetTicks()) * 0.001f;
    const float a = shake_ * shake_ * 0.18f;
    camPos_ += glm::dvec3(right) * double(a * std::sin(t * 61.0f)) +
               glm::dvec3(0.0, double(a * std::sin(t * 47.0f + 1.3f)), 0.0);
    shake_ = std::max(0.0f, shake_ - dt * 3.5f);
  }

  camera_.position = camPos_;
  camera_.yaw = camYaw_;
  camera_.pitch = camPitch_;
  camera_.fovYDegrees = cfg_.render.fovYDegrees;

  int w = 0, h = 0;
  SDL_GetWindowSizeInPixels(window_, &w, &h);
  if (w > 0 && h > 0)
    aspect_ = float(w) / float(h);
}

bool App::worldToScreen(const glm::dvec3 &p, float &sx, float &sy) const {
  const glm::vec3 rel = glm::vec3(p - camPos_);
  const glm::vec3 fwd = aimDirection();
  const glm::mat4 view = glm::lookAt(glm::vec3(0.0f), fwd, glm::vec3(0, 1, 0));
  const glm::mat4 proj = glm::perspective(glm::radians(camera_.fovYDegrees), aspect_, 0.05f, 1000.0f);
  const glm::vec4 clip = proj * view * glm::vec4(rel, 1.0f);
  if (clip.w <= 0.01f)
    return false;
  const ImVec2 size = ImGui::GetIO().DisplaySize;
  sx = (clip.x / clip.w * 0.5f + 0.5f) * size.x;
  sy = (1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * size.y;
  return true;
}


} // namespace ao::client
