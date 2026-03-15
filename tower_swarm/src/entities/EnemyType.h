#pragma once

#include <cstdint>
#include <string_view>

namespace tower_swarm {

enum class EnemyType : std::uint8_t {
  Grub = 0,
  Hulk = 1,
  Scuttle = 2,
  Driftwing = 3,
  Divide = 4,
  Vanguard = 5,
  Mender = 6,
  SiegeLord = 7,
  Count = 8
};

inline constexpr std::string_view to_string(EnemyType t) {
  switch (t) {
  case EnemyType::Grub:
    return "Grub";
  case EnemyType::Hulk:
    return "Hulk";
  case EnemyType::Scuttle:
    return "Scuttle";
  case EnemyType::Driftwing:
    return "Driftwing";
  case EnemyType::Divide:
    return "Divide";
  case EnemyType::Vanguard:
    return "Vanguard";
  case EnemyType::Mender:
    return "Mender";
  case EnemyType::SiegeLord:
    return "Siege Lord";
  case EnemyType::Count:
    return "Count";
  }
  return "Unknown";
}

} // namespace tower_swarm

