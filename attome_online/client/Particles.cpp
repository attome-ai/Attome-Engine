#include "Particles.h"

#include <algorithm>
#include <cmath>

namespace ao::client {

namespace vb = atm::voxel::blocks;

float Particles::rnd() {
  seed_ ^= seed_ << 13;
  seed_ ^= seed_ >> 17;
  seed_ ^= seed_ << 5;
  return float(seed_ & 0xFFFFFFu) / float(0xFFFFFF);
}

void Particles::burst(const glm::dvec3 &at, const Burst &b) {
  if (p_.size() > 6000) return;
  for (int i = 0; i < b.count; ++i) {
    P p;
    glm::vec3 d(rnds(), rnds(), rnds());
    const float l = glm::length(d);
    d = l > 1e-3f ? d / l : glm::vec3(0, 1, 0);
    p.pos = at + glm::dvec3(d * (b.spread * rnd()));
    p.vel = d * (b.speed * (0.4f + 0.6f * rnd())) + glm::vec3(0.0f, b.up * (0.5f + 0.5f * rnd()), 0.0f);
    p.axis = glm::normalize(glm::vec3(rnds(), rnds(), rnds()) + glm::vec3(0.0f, 0.01f, 0.0f));
    p.spin = rnds() * 12.0f;
    p.life = b.life * (0.6f + 0.8f * rnd());
    p.size = b.size * (0.6f + 0.8f * rnd());
    p.gravity = b.gravity;
    p.drag = b.drag;
    p.tint = b.tint;
    p.block = b.block;
    p_.push_back(p);
  }
}

void Particles::slashArc(const glm::dvec3 &center, float yaw, uint32_t tint, float radius) {
  // yaw 0 faces -Z; right = +X rotated by yaw.
  const glm::vec3 fwd(-std::sin(yaw), 0.0f, -std::cos(yaw));
  const glm::vec3 right(std::cos(yaw), 0.0f, -std::sin(yaw));
  constexpr int kN = 22;
  for (int i = 0; i < kN; ++i) {
    const float t = float(i) / float(kN - 1);
    const float a = (t - 0.5f) * 2.3f;                // -66 .. +66 degrees, left to right
    const glm::vec3 dir = fwd * std::cos(a) + right * std::sin(a);
    for (int layer = 0; layer < 2; ++layer) {
      P p;
      const float r = radius * (layer ? 0.78f : 1.0f);
      p.pos = center + glm::dvec3(dir * r) + glm::dvec3(0.0, 0.25 * (t - 0.5), 0.0);
      p.vel = dir * 1.5f + right * 0.8f;
      p.age = -t * 0.12f;                             // appears as the blade sweeps
      p.life = 0.22f;
      p.size = layer ? 0.09f : 0.13f;
      p.gravity = 0.0f;
      p.drag = 4.0f;
      p.spin = 6.0f;
      p.tint = tint;
      p.block = vb::Lamp;
      p_.push_back(p);
    }
  }
}

void Particles::ambient(const glm::dvec3 &around, float dt) {
  // Keep ~48 motes alive around the player; spawn a few per second.
  static float acc = 0.0f;
  acc += dt;
  while (acc > 0.05f && motes_ < 48) {
    acc -= 0.05f;
    P p;
    p.mote = true;
    p.pos = around + glm::dvec3(rnds() * 14.0, 0.3 + rnd() * 4.5, rnds() * 14.0);
    p.vel = glm::vec3(rnds() * 0.3f, 0.15f + rnd() * 0.2f, rnds() * 0.3f);
    p.life = 4.0f + rnd() * 4.0f;
    p.size = 0.05f + rnd() * 0.04f;
    p.gravity = 0.0f;
    p.drag = 0.0f;
    p.spin = rnds() * 2.0f;
    p.tint = rnd() < 0.7f ? 0xFF80F0FFu : 0xFFA0FFD0u; // warm gold / pale green
    p.block = vb::Lamp;
    p_.push_back(p);
    ++motes_;
  }
  if (acc > 0.05f) acc = 0.05f;
}

void Particles::update(float dt) {
  for (P &p : p_) {
    p.age += dt;
    if (p.age < 0.0f) continue;
    p.vel.y -= p.gravity * dt;
    p.vel *= std::max(0.0f, 1.0f - p.drag * dt);
    if (p.mote) { // gentle wander
      p.vel.x += std::sin(p.age * 1.7f + float(p.pos.z)) * 0.25f * dt;
      p.vel.z += std::cos(p.age * 1.3f + float(p.pos.x)) * 0.25f * dt;
    }
    p.pos += glm::dvec3(p.vel * dt);
  }
  for (const P &p : p_)
    if (p.mote && p.age >= p.life) --motes_;
  p_.erase(std::remove_if(p_.begin(), p_.end(), [](const P &p) { return p.age >= p.life; }), p_.end());
}

void Particles::draw(atm::render::Renderer &renderer,
                     const std::vector<atm::render::ModelMeshId> &blockMeshes) const {
  for (const P &p : p_) {
    if (p.age < 0.0f || p.block >= blockMeshes.size() || blockMeshes[p.block] == atm::render::kInvalidModelMesh)
      continue;
    const float t = p.age / p.life;
    // Motes fade in and out; everything else pops then shrinks away.
    const float k = p.mote ? std::sin(t * 3.14159f) : (t < 0.1f ? 0.6f + 4.0f * t : 1.0f - (t - 0.1f) / 0.9f * 0.85f);
    atm::render::ModelInstance inst;
    inst.mesh = blockMeshes[p.block];
    inst.origin = p.pos;
    inst.rotation = glm::angleAxis(p.spin * p.age, p.axis);
    inst.pivot = {2.0f, 2.0f, 2.0f};
    inst.voxelScale = std::max(0.0f, p.size * k) / 4.0f; // item cubes are 4 voxels wide
    inst.tint = p.tint;
    renderer.drawModel(inst);
  }
}

} // namespace ao::client
