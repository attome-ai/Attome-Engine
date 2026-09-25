#pragma once

// Trove-style voxel particles: tiny cubes drawn as model instances of the
// per-block item cube meshes (tinted). Glowing sparks use the lamp block's
// emissive cube, so they bloom. Client-side only, purely cosmetic.

#include "../../engine/render/Renderer.h"
#include "../../engine/voxel/BlockRegistry.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <vector>

namespace ao::client {

class Particles {
public:
  struct Burst {
    int count = 10;
    atm::voxel::BlockId block = atm::voxel::blocks::Snow; // mesh (colour source)
    uint32_t tint = 0xFFFFFFFFu;  // bytes R,G,B,A
    float speed = 4.0f;           // blocks/s, random direction
    float up = 2.0f;              // extra upward velocity
    float size = 0.12f;           // edge length in blocks
    float life = 0.6f;            // seconds
    float gravity = 16.0f;
    float drag = 1.5f;            // 1/s
    float spread = 0.25f;         // spawn jitter radius
  };

  void burst(const glm::dvec3 &at, const Burst &b);
  // Glowing sweep in front of a swing: an arc of sparks appearing left to
  // right over ~0.12 s around `center` facing `yaw` (0 = -Z).
  void slashArc(const glm::dvec3 &center, float yaw, uint32_t tint, float radius = 1.7f);
  // Floating firefly motes around the player (call every frame).
  void ambient(const glm::dvec3 &around, float dt);
  void update(float dt);
  void draw(atm::render::Renderer &renderer, const std::vector<atm::render::ModelMeshId> &blockMeshes) const;

private:
  struct P {
    glm::dvec3 pos{0.0};
    glm::vec3 vel{0.0f};
    glm::vec3 axis{0.0f, 1.0f, 0.0f};
    float age = 0.0f, life = 1.0f, size = 0.1f, gravity = 0.0f, drag = 0.0f, spin = 0.0f;
    uint32_t tint = 0xFFFFFFFFu;
    atm::voxel::BlockId block = 0;
    bool mote = false;
  };
  float rnd();          // 0..1
  float rnds() { return rnd() * 2.0f - 1.0f; }
  std::vector<P> p_;
  uint32_t seed_ = 0x9E3779B9u;
  int motes_ = 0;
};

} // namespace ao::client
