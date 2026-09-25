#pragma once

// Ground decoration (Trove-style clutter): grass tufts and flowers on grass,
// pebbles and shells on sand. Tiny voxel models placed deterministically
// (hash of the block position) around the player and drawn as model
// instances. Client-side and purely cosmetic.

#include "../../engine/render/Renderer.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/voxel/VoxelWorld.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace ao::client {

class Decor {
public:
  // Colours appended to the material table; decoration palette index k uses
  // material (materialBase + k - 1).
  static std::vector<uint32_t> palette();
  // Builds the decoration meshes. `materialBase` = index of palette()[0].
  void init(atm::render::Renderer &renderer, uint32_t materialBase);
  void shutdown(atm::render::Renderer &renderer);
  // Re-scatters around the player when it moved or periodically (chunks load).
  void update(const atm::voxel::VoxelWorld &world, const glm::dvec3 &player, float dt);
  void draw(atm::render::Renderer &renderer) const;

private:
  enum Kind : uint8_t { TuftA, TuftB, FlowerRed, FlowerYellow, FlowerWhite, FlowerBlue, Pebble, Shell, KindCount };
  struct Item {
    glm::dvec3 pos;
    float yaw;
    float scale;
    uint8_t kind;
  };
  std::vector<atm::render::ModelMeshId> meshes_;
  std::vector<glm::vec3> pivots_;
  std::vector<Item> items_;
  glm::dvec3 center_{1e30};
  float timer_ = 0.0f;
};

} // namespace ao::client
