#include "MapCache.h"

#include "../../shared/world/Town.h"

#include <algorithm>
#include <cmath>
#include <chrono>

namespace ao::client {

namespace {

constexpr int kSeaLevel = 64; // water fills y < 64 (WorldGenerator)

uint32_t scale(uint32_t c, float k) {
  auto ch = [&](int s) { return uint32_t(std::clamp(float((c >> s) & 0xFF) * k, 0.0f, 255.0f)); };
  return ch(0) | (ch(8) << 8) | (ch(16) << 16) | 0xFF000000u;
}

} // namespace

uint32_t MapCache::scanColumn(const atm::voxel::VoxelWorld &world, const atm::voxel::BlockRegistry &blocks,
                              int32_t x, int32_t z) const {
  const int h = ao::world::groundHeight(world.generator(), x, z);
  // Town columns: the colour of the fine-voxel roof / paving on top.
  if (uint32_t tc; ao::world::townMapColor(ao::world::homeTown(world.generator()), x, z, tc)) {
    const int hn = ao::world::groundHeight(world.generator(), x - 1, z - 1);
    return scale(tc, std::clamp(1.0f + float(h - hn) * 0.08f, 0.75f, 1.2f));
  }
  // Top non-air block near the generated surface (catches trees and edits).
  int top = -1;
  atm::voxel::BlockId b = atm::voxel::kAir;
  for (int y = std::max(h, kSeaLevel) + 10; y >= h - 6 && y >= 0; --y) {
    b = world.blockAt({x, y, z});
    if (b != atm::voxel::kAir) {
      top = y;
      break;
    }
  }
  if (top < 0)
    return 0;
  const auto &def = blocks.get(b);
  uint32_t c = def.colorTop | 0xFF000000u;
  if (def.liquid) { // deeper water is darker
    const float depth = float(std::max(0, top - h));
    return scale(c, std::clamp(1.1f - depth * 0.06f, 0.55f, 1.1f));
  }
  // Hill shading from the slope toward the north-west (light from there).
  const int hn = ao::world::groundHeight(world.generator(), x - 1, z - 1);
  const float k = std::clamp(1.0f + float(h - hn) * 0.12f, 0.72f, 1.25f);
  return scale(c, k);
}

void MapCache::update(const atm::voxel::VoxelWorld &world, const atm::voxel::BlockRegistry &blocks,
                      const glm::dvec3 &player, int radius, float budgetMs, float dt) {
  if (ringRadius_ != radius) {
    ring_.clear();
    for (int dz = -radius; dz <= radius; ++dz)
      for (int dx = -radius; dx <= radius; ++dx)
        if (dx * dx + dz * dz <= radius * radius)
          ring_.push_back({dx, dz});
    std::sort(ring_.begin(), ring_.end(), [](const glm::ivec2 &a, const glm::ivec2 &b) {
      return a.x * a.x + a.y * a.y < b.x * b.x + b.y * b.y;
    });
    ringRadius_ = radius;
    cursor_ = 0;
  }
  // Restart near the player after moving a few blocks, so the view around
  // them fills first and changes get picked up.
  const glm::ivec2 c(int(std::floor(player.x)), int(std::floor(player.z)));
  if (std::abs(c.x - center_.x) + std::abs(c.y - center_.y) > 6) {
    center_ = c;
    cursor_ = 0;
    idle_ = false;
  }
  // After a full pass, rest (no work at all) until the player moves or ~10 s
  // pass (to pick up block edits and chunks that loaded late).
  if (idle_) {
    idleTime_ += dt;
    if (idleTime_ < 10.0f)
      return;
    idle_ = false;
    cursor_ = 0;
  }
  const auto t0 = std::chrono::steady_clock::now();
  for (int n = 0; cursor_ < ring_.size(); ++n, ++cursor_) {
    // Check the clock every 32 columns (a column costs ~0.5 us).
    if ((n & 31) == 31 &&
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count() > budgetMs)
      break;
    const int32_t x = center_.x + ring_[cursor_].x, z = center_.y + ring_[cursor_].y;
    // Skip columns whose chunks are not loaded yet (retried on the next pass).
    if (!world.isLoaded(atm::voxel::chunkOf({x, kSeaLevel, z})))
      continue;
    const int32_t tx = tileOf(x), tz = tileOf(z);
    Tile &t = tiles_[key(tx, tz)];
    uint32_t &slot = t.color[size_t(z - tz * kTile) * kTile + size_t(x - tx * kTile)];
    const uint32_t col = scanColumn(world, blocks, x, z);
    if ((slot >> 24) == 0 && (col >> 24) != 0)
      ++t.scanned;
    if ((col >> 24) != 0)
      slot = col;
  }
  if (cursor_ >= ring_.size()) {
    idle_ = true;
    idleTime_ = 0.0f;
  }
}

uint32_t MapCache::colorAt(int32_t x, int32_t z) const {
  const int32_t tx = tileOf(x), tz = tileOf(z);
  const auto it = tiles_.find(key(tx, tz));
  if (it == tiles_.end())
    return 0;
  return it->second.color[size_t(z - tz * kTile) * kTile + size_t(x - tx * kTile)];
}

} // namespace ao::client
