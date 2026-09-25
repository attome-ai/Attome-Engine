// Skills, XP curve and combat level (GAME_DESIGN §9.4, §9.5).
// Items and monsters live in Content.cpp.

#include "GameTypes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ao {

std::string_view skillName(Skill s) {
  static constexpr std::array<std::string_view, kSkillCount> kNames = {
      "Attack",    "Strength",   "Defence",  "Ranged",   "Magic",    "Hitpoints", "Prayer",
      "Mining",    "Woodcutting", "Fishing", "Hunter",   "Farming",  "Smithing",  "Crafting",
      "Fletching", "Cooking",    "Herblore", "Construction", "Agility", "Slayer", "Taming"};
  const size_t i = size_t(s);
  return i < kNames.size() ? kNames[i] : std::string_view("Unknown");
}

// ---------------------------------------------------------------------------
// XP curve: RuneScape formula to 99, then a fixed 1,228,825 XP per level.
//   XP(L) = floor( 1/4 * sum_{x=1}^{L-1} floor(x + 300 * 2^(x/7)) )
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kCurveTop = 99;
constexpr uint64_t kXpPerLevelAfter99 = 1228825;

struct XpTable {
  std::array<uint64_t, kCurveTop + 1> xp{}; // xp[L] for L = 1..99 (xp[0] = 0)
  XpTable() {
    uint64_t points = 0; // running sum of floor(x + 300 * 2^(x/7))
    xp[0] = 0;
    xp[1] = 0;
    for (uint32_t level = 2; level <= kCurveTop; ++level) {
      const uint32_t x = level - 1;
      points += uint64_t(std::floor(double(x) + 300.0 * std::pow(2.0, double(x) / 7.0)));
      xp[level] = points / 4; // floor(points / 4)
    }
  }
};

const XpTable &xpTable() {
  static const XpTable t;
  return t;
}

} // namespace

uint64_t xpForLevel(uint32_t level) {
  if (level <= 1)
    return 0;
  const XpTable &t = xpTable();
  if (level <= kCurveTop)
    return t.xp[level];
  // Linear after 99; saturate instead of overflowing (level ~1.5e13 needed).
  const uint64_t extra = uint64_t(level - kCurveTop);
  const uint64_t top = t.xp[kCurveTop];
  if (extra > (std::numeric_limits<uint64_t>::max() - top) / kXpPerLevelAfter99)
    return std::numeric_limits<uint64_t>::max();
  return top + extra * kXpPerLevelAfter99;
}

uint32_t levelForXp(uint64_t xp) {
  const XpTable &t = xpTable();
  const uint64_t top = t.xp[kCurveTop];
  if (xp >= top) {
    const uint64_t extra = (xp - top) / kXpPerLevelAfter99;
    const uint64_t level = uint64_t(kCurveTop) + extra;
    return level > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                         : uint32_t(level);
  }
  // Highest L in 1..98 with xp[L] <= xp.
  uint32_t level = 1;
  while (level + 1 <= kCurveTop && t.xp[level + 1] <= xp)
    ++level;
  return level;
}

// ---------------------------------------------------------------------------
// Combat level (Old School RuneScape shape, levels capped at 99), computed in
// integer thousandths so the floor is exact:
//   base   = 0.25  * (Def + HP + floor(Prayer / 2))
//   melee  = 0.325 * (Att + Str)
//   ranged = 0.325 * floor(1.5 * Ranged)
//   magic  = 0.325 * floor(1.5 * Magic)
//   combat = floor(base + max(melee, ranged, magic))
// ---------------------------------------------------------------------------
uint32_t combatLevel(const std::array<uint32_t, kSkillCount> &levels) {
  auto lv = [&](Skill s) -> uint64_t {
    const uint32_t v = levels[size_t(s)];
    return uint64_t(std::clamp<uint32_t>(v, 1u, 99u));
  };
  const uint64_t base = 250 * (lv(Skill::Defence) + lv(Skill::Hitpoints) + lv(Skill::Prayer) / 2);
  const uint64_t melee = 325 * (lv(Skill::Attack) + lv(Skill::Strength));
  const uint64_t ranged = 325 * ((lv(Skill::Ranged) * 3) / 2);
  const uint64_t magic = 325 * ((lv(Skill::Magic) * 3) / 2);
  return uint32_t((base + std::max({melee, ranged, magic})) / 1000);
}

} // namespace ao
