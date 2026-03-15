#pragma once

#include "Constants.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tower_swarm::evolution {

inline int killsNeededForNextTier(int tier) {
  const int t = std::max(1, tier);
  switch (t) {
  case 1:
    return kKillsTier1To2;
  case 2:
    return kKillsTier2To3;
  case 3:
    return kKillsTier3To4;
  case 4:
    return kKillsTier4To5;
  default:
    break;
  }

  const double base = static_cast<double>(kKillsInfiniteBase);
  const double growth = static_cast<double>(kKillsInfiniteGrowth);
  const double v = base * std::pow(growth, static_cast<double>(t - 1));
  if (!std::isfinite(v) ||
      v >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return std::max(0, static_cast<int>(std::floor(v)));
}

inline float sizeMultiplierForTier(int tier) {
  const int t = std::max(1, tier);
  float mult = kTierSize1To3;
  if (t >= 20) {
    mult = kTierSize20Plus;
  } else if (t >= 16) {
    mult = kTierSize16To19;
  } else if (t >= 13) {
    mult = kTierSize13To15;
  } else if (t >= 10) {
    mult = kTierSize10To12;
  } else if (t >= 7) {
    mult = kTierSize7To9;
  } else if (t >= 4) {
    mult = kTierSize4To6;
  }
  return std::clamp(mult, 0.0f, kTierSizeCap);
}

inline int visualBandIndexForTier(int tier) {
  const int t = std::max(1, tier);
  if (t >= 20) {
    return 6;
  }
  if (t >= 16) {
    return 5;
  }
  if (t >= 13) {
    return 4;
  }
  if (t >= 10) {
    return 3;
  }
  if (t >= 7) {
    return 2;
  }
  if (t >= 4) {
    return 1;
  }
  return 0;
}

inline int sizePxForTier(int base_size_px, int tier, float extra_scale = 1.0f) {
  const float scale =
      std::max(0.0f, sizeMultiplierForTier(tier) * std::max(0.0f, extra_scale));
  const float size_f =
      static_cast<float>(std::max(1, base_size_px)) * scale;
  if (!std::isfinite(size_f)) {
    return std::max(1, base_size_px);
  }
  const int size_px = static_cast<int>(std::lround(size_f));
  return std::max(1, size_px);
}

inline int creatureSizePxForTier(int tier, float extra_scale = 1.0f) {
  return sizePxForTier(kCreatureBaseSizePx, tier, extra_scale);
}

} // namespace tower_swarm::evolution
