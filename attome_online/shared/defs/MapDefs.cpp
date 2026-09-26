#include "DefsInternal.h"

namespace ao {

namespace defs_detail {

std::vector<MapRegion> &regionTable() {
  static std::vector<MapRegion> t;
  return t;
}

std::string &mapNameStorage() {
  static std::string s;
  return s;
}

} // namespace defs_detail

std::string_view mapName() { return defs_detail::mapNameStorage(); }

const std::vector<MapRegion> &mapRegions() { return defs_detail::regionTable(); }

const MapRegion *regionAt(double x, double z) {
  const auto &t = defs_detail::regionTable();
  for (const MapRegion &r : t) {
    if (!r.hasShape)
      return &r;
    const double dx = x - r.centerX, dz = z - r.centerZ;
    const double d2 = dx * dx + dz * dz;
    if (d2 >= r.minRadius * r.minRadius && d2 < r.maxRadius * r.maxRadius)
      return &r;
  }
  return t.empty() ? nullptr : &t.back();
}

int pickSpawn(const MapRegion &region, float r01) {
  if (region.spawns.empty() || region.totalWeight <= 0.0f)
    return -1;
  float pick = r01 * region.totalWeight;
  for (const MapRegion::Spawn &s : region.spawns) {
    if (pick < s.weight)
      return s.npc;
    pick -= s.weight;
  }
  return region.spawns.back().npc;
}

} // namespace ao
