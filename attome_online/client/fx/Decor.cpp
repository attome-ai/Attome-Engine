#include "Decor.h"

#include "../../../engine/model/Character.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>

namespace ao::client {

namespace {

namespace vb = atm::voxel::blocks;

constexpr uint32_t rgba(int r, int g, int b) {
  return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | 0xFF000000u;
}

// Palette indices (1-based, 0 = empty voxel).
enum : uint8_t {
  kGrassA = 1, kGrassB, kGrassTip, kStem, kRed, kYellow, kWhite, kBlue, kCenter, kPebble, kPebbleDark, kShell,
  kPaletteSize
};

constexpr int kRadius = 72;       // blocks around the player
constexpr float kFadeBand = 14.0f; // items grow in / shrink out over the outer band
constexpr float kVoxel = 1.0f / 8.0f;

uint32_t hash(int x, int y, int z, uint32_t salt) {
  uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ uint32_t(z) * 0xcb1ab31fu ^ salt;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

float unit(uint32_t h) { return float(h & 0xFFFFu) / 65535.0f; }

atm::model::VoxelPart makePart(int sx, int sy, int sz) {
  atm::model::VoxelPart p;
  p.name = "decor";
  p.sx = sx;
  p.sy = sy;
  p.sz = sz;
  p.voxels.assign(size_t(sx) * size_t(sy) * size_t(sz), 0);
  p.palette.assign(kPaletteSize, 0u); // colours come from the materials
  p.pivot = {sx * 0.5f, 0.0f, sz * 0.5f};
  return p;
}

void column(atm::model::VoxelPart &p, int x, int z, int h, uint8_t body, uint8_t tip) {
  for (int y = 0; y < h && y < p.sy; ++y)
    p.at(x, y, z) = y == h - 1 ? tip : body;
}

atm::model::VoxelPart tuft(bool big) {
  atm::model::VoxelPart p = makePart(5, 5, 5);
  // Blades of different heights; lighter tips.
  const int blades[][4] = {{2, 2, 5, kGrassB}, {1, 1, 4, kGrassA}, {3, 3, 4, kGrassB}, {0, 2, 3, kGrassA},
                           {4, 2, 3, kGrassA}, {2, 4, 3, kGrassB}, {2, 0, 3, kGrassA}, {3, 1, 2, kGrassB},
                           {1, 3, 2, kGrassB}};
  const int n = big ? 9 : 6;
  for (int i = 0; i < n; ++i) {
    const int h = big ? blades[i][2] : std::max(1, blades[i][2] - 1);
    column(p, blades[i][0], blades[i][1], h, uint8_t(blades[i][3]), kGrassTip);
  }
  return p;
}

atm::model::VoxelPart flower(uint8_t petal) {
  atm::model::VoxelPart p = makePart(3, 6, 3);
  column(p, 1, 1, 4, kStem, kStem);
  p.at(0, 1, 1) = kStem; // leaf
  p.at(2, 2, 1) = kStem;
  p.at(1, 4, 0) = petal;
  p.at(0, 4, 1) = petal;
  p.at(2, 4, 1) = petal;
  p.at(1, 4, 2) = petal;
  p.at(1, 4, 1) = kCenter;
  p.at(1, 5, 1) = petal;
  return p;
}

atm::model::VoxelPart pebble() {
  atm::model::VoxelPart p = makePart(3, 2, 3);
  for (int z = 0; z < 2; ++z)
    for (int x = 0; x < 3; ++x)
      p.at(x, 0, z) = kPebble;
  p.at(1, 0, 2) = kPebbleDark;
  p.at(1, 1, 1) = kPebbleDark;
  p.at(0, 1, 0) = kPebble;
  return p;
}

atm::model::VoxelPart shell() {
  atm::model::VoxelPart p = makePart(3, 1, 3);
  p.at(1, 0, 0) = kShell;
  p.at(0, 0, 1) = kShell;
  p.at(1, 0, 1) = kShell;
  p.at(2, 0, 1) = kShell;
  p.at(1, 0, 2) = kShell;
  return p;
}

} // namespace

std::vector<uint32_t> Decor::palette() {
  // Order matches the palette index enum (index 1 first).
  return {rgba(100, 168, 60),  rgba(114, 184, 68), rgba(150, 204, 92), rgba(80, 140, 52),
          rgba(236, 62, 72),   rgba(252, 212, 58), rgba(246, 246, 240), rgba(96, 150, 250),
          rgba(255, 186, 40),  rgba(150, 150, 160), rgba(112, 112, 124), rgba(246, 214, 196)};
}

void Decor::init(atm::render::Renderer &renderer, uint32_t materialBase) {
  const atm::model::VoxelPart parts[KindCount] = {tuft(true),      tuft(false),     flower(kRed), flower(kYellow),
                                                  flower(kWhite), flower(kBlue), pebble(),     shell()};
  meshes_.assign(KindCount, atm::render::kInvalidModelMesh);
  pivots_.assign(KindCount, glm::vec3(0.0f));
  atm::voxel::ChunkMeshData mesh;
  for (int k = 0; k < KindCount; ++k) {
    atm::model::meshPart(parts[k], uint16_t(materialBase - 1), mesh);
    if (!mesh.empty())
      meshes_[size_t(k)] = renderer.createModelMesh(mesh);
    pivots_[size_t(k)] = parts[k].pivot;
  }
}

void Decor::shutdown(atm::render::Renderer &renderer) {
  for (auto id : meshes_)
    if (id != atm::render::kInvalidModelMesh)
      renderer.destroyModelMesh(id);
  meshes_.clear();
  items_.clear();
}

const Decor::Column &Decor::column(const atm::voxel::VoxelWorld &world, int x, int z, int py) {
  const uint64_t key = (uint64_t(uint32_t(x)) << 32) | uint32_t(z);
  auto it = cache_.find(key);
  if (it != cache_.end())
    return it->second;
  static const Column kPending{};
  // Only remember columns whose chunks are loaded (else retry later).
  if (!world.isLoaded(atm::voxel::chunkOf({x, py + 14, z})) ||
      !world.isLoaded(atm::voxel::chunkOf({x, py - 20, z})))
    return kPending;
  Column c;
  // Surface: first solid block from above with air on top.
  atm::voxel::BlockId above = world.blockAt({x, py + 14, z});
  for (int y = py + 13; y >= py - 20; --y) {
    const atm::voxel::BlockId b = world.blockAt({x, y, z});
    if (b != atm::voxel::kAir && above == atm::voxel::kAir) {
      const uint32_t h = hash(x, y, z, 0x5eedu);
      const float r = unit(h);
      int kind = -1;
      if (b == vb::Grass) {
        if (r < 0.03f * density_) kind = FlowerRed + int((h >> 20) & 3u);
        else if (r < 0.13f * density_) kind = TuftA;
        else if (r < 0.20f * density_) kind = TuftB;
      } else if (b == vb::Sand) {
        if (r < 0.02f * density_) kind = Pebble;
        else if (r < 0.035f * density_) kind = Shell;
      } else if (b == vb::Stone && r < 0.03f * density_) {
        kind = Pebble;
      }
      if (kind >= 0) {
        const uint32_t h2 = hash(x, y, z, 0xdec0u);
        c.has = true;
        c.item.pos = glm::dvec3(x + 0.5 + (unit(h2) - 0.5) * 0.5, y + 1.0, z + 0.5 + (unit(h2 >> 16) - 0.5) * 0.5);
        c.item.yaw = unit(h2 * 3u) * 6.2831853f;
        c.item.scale = 0.85f + unit(h2 * 7u) * 0.35f;
        c.item.kind = uint8_t(kind);
      }
      break;
    }
    above = b;
  }
  return cache_.emplace(key, c).first->second;
}

void Decor::update(const atm::voxel::VoxelWorld &world, const glm::dvec3 &player, float dt) {
  player_ = player;
  timer_ += dt;
  flushTimer_ += dt;
  // Forget everything now and then: picks up dug / placed blocks and keeps
  // the cache from growing while exploring.
  if (flushTimer_ > 20.0f) {
    flushTimer_ = 0.0f;
    cache_.clear();
    timer_ = 1e9f;
  }
  const glm::dvec3 d = player - center_;
  if (d.x * d.x + d.z * d.z < 16.0 && timer_ < 1.0f)
    return;
  timer_ = 0.0f;
  center_ = player;
  items_.clear();
  const int px = int(std::floor(player.x)), py = int(std::floor(player.y)), pz = int(std::floor(player.z));
  const int r2 = kRadius * kRadius;
  for (int z = pz - kRadius; z <= pz + kRadius; ++z)
    for (int x = px - kRadius; x <= px + kRadius; ++x) {
      const int dx = x - px, dz = z - pz;
      if (dx * dx + dz * dz > r2)
        continue; // round area: the fade band is the same in every direction
      const Column &c = column(world, x, z, py);
      if (c.has)
        items_.push_back(c.item);
    }
}

void Decor::setDensity(float d) {
  d = std::clamp(d, 0.0f, 4.0f);
  if (d == density_)
    return;
  density_ = d;
  cache_.clear(); // placement depends on the density
  timer_ = 1e9f;
}

void Decor::draw(atm::render::Renderer &renderer) const {
  for (const Item &it : items_) {
    const auto mesh = meshes_[it.kind];
    if (mesh == atm::render::kInvalidModelMesh)
      continue;
    // Grow in / shrink out near the edge of the area instead of popping.
    const double dx = it.pos.x - player_.x, dz = it.pos.z - player_.z;
    const float dist = float(std::sqrt(dx * dx + dz * dz));
    const float fade = 1.0f - std::clamp((dist - (float(kRadius) - kFadeBand)) / kFadeBand, 0.0f, 1.0f);
    if (fade <= 0.02f)
      continue;
    atm::render::ModelInstance inst;
    inst.mesh = mesh;
    inst.origin = it.pos;
    inst.rotation = glm::angleAxis(it.yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    inst.pivot = pivots_[it.kind];
    inst.voxelScale = kVoxel * it.scale * fade;
    inst.flags = atm::render::kInstanceNoRim | atm::render::kInstanceNoShadow;
    renderer.drawModel(inst);
  }
}

} // namespace ao::client
