#pragma once

#include <cstdint>
#include <string_view>

namespace tower_swarm {

enum class CharacterId : std::uint8_t {
  Brix = 0,
  Flara = 1,
  Mossling = 2,
  Glitch = 3,
  Ironjaw = 4,
  Wraith = 5,
  Crystalis = 6,
  Vex = 7,
  Orin = 8,
  NullSeed = 9,
  Count = 10
};

enum class CharacterRole : std::uint8_t {
  Shooter = 0,
  Splasher = 1,
  Support = 2,
  Trapper = 3,
  Charger = 4,
  Sniper = 5,
  Hybrid = 6,
  Chaos = 7,
  Titan = 8,
  Nullifier = 9
};

enum class Rarity : std::uint8_t {
  Common = 0,
  Rare = 1,
  Epic = 2,
  Legendary = 3
};

inline constexpr std::string_view to_string(CharacterId id) {
  switch (id) {
  case CharacterId::Brix:
    return "Brix";
  case CharacterId::Flara:
    return "Flara";
  case CharacterId::Mossling:
    return "Mossling";
  case CharacterId::Glitch:
    return "Glitch";
  case CharacterId::Ironjaw:
    return "Ironjaw";
  case CharacterId::Wraith:
    return "Wraith";
  case CharacterId::Crystalis:
    return "Crystalis";
  case CharacterId::Vex:
    return "Vex";
  case CharacterId::Orin:
    return "Orin";
  case CharacterId::NullSeed:
    return "Null";
  case CharacterId::Count:
    return "Count";
  }
  return "Unknown";
}

inline bool from_string(std::string_view s, CharacterId &out) {
  if (s == "Brix") {
    out = CharacterId::Brix;
    return true;
  }
  if (s == "Flara") {
    out = CharacterId::Flara;
    return true;
  }
  if (s == "Mossling") {
    out = CharacterId::Mossling;
    return true;
  }
  if (s == "Glitch") {
    out = CharacterId::Glitch;
    return true;
  }
  if (s == "Ironjaw") {
    out = CharacterId::Ironjaw;
    return true;
  }
  if (s == "Wraith") {
    out = CharacterId::Wraith;
    return true;
  }
  if (s == "Crystalis") {
    out = CharacterId::Crystalis;
    return true;
  }
  if (s == "Vex") {
    out = CharacterId::Vex;
    return true;
  }
  if (s == "Orin") {
    out = CharacterId::Orin;
    return true;
  }
  if (s == "Null") {
    out = CharacterId::NullSeed;
    return true;
  }
  return false;
}

} // namespace tower_swarm

