#pragma once

#include <cstdint>
#include <string_view>

namespace tower_swarm {

enum class RelicId : std::uint8_t {
  IronCore = 0,
  Bloodshard = 1,
  EssenceMagnet = 2,
  MergersGift = 3,
  WarpedTime = 4,
  PackInstinct = 5,
  EruptionCore = 6,
  ChainStrike = 7,
  VoidLens = 8,
  LivingWall = 9,
  ApexHunger = 10,
  TwinPulse = 11,
  ColdBloom = 12,
  ResonantGrowth = 13,
  ChaosSpark = 14,
  EternalEcho = 15,
  RecursiveMerge = 16,
  ShardHunger = 17,
  DeathBloom = 18,
  TheQuiet = 19,
  Count = 20,
  None = 255,
};

inline constexpr std::string_view to_string(RelicId id) {
  switch (id) {
  case RelicId::IronCore:
    return "Iron Core";
  case RelicId::Bloodshard:
    return "Bloodshard";
  case RelicId::EssenceMagnet:
    return "Essence Magnet";
  case RelicId::MergersGift:
    return "Merger's Gift";
  case RelicId::WarpedTime:
    return "Warped Time";
  case RelicId::PackInstinct:
    return "Pack Instinct";
  case RelicId::EruptionCore:
    return "Eruption Core";
  case RelicId::ChainStrike:
    return "Chain Strike";
  case RelicId::VoidLens:
    return "Void Lens";
  case RelicId::LivingWall:
    return "Living Wall";
  case RelicId::ApexHunger:
    return "Apex Hunger";
  case RelicId::TwinPulse:
    return "Twin Pulse";
  case RelicId::ColdBloom:
    return "Cold Bloom";
  case RelicId::ResonantGrowth:
    return "Resonant Growth";
  case RelicId::ChaosSpark:
    return "Chaos Spark";
  case RelicId::EternalEcho:
    return "Eternal Echo";
  case RelicId::RecursiveMerge:
    return "Recursive Merge";
  case RelicId::ShardHunger:
    return "Shard Hunger";
  case RelicId::DeathBloom:
    return "Death Bloom";
  case RelicId::TheQuiet:
    return "The Quiet";
  case RelicId::Count:
    return "Count";
  case RelicId::None:
    return "None";
  }
  return "None";
}

inline bool from_string(std::string_view s, RelicId &out) {
  if (s == "Iron Core") {
    out = RelicId::IronCore;
    return true;
  }
  if (s == "Bloodshard") {
    out = RelicId::Bloodshard;
    return true;
  }
  if (s == "Essence Magnet") {
    out = RelicId::EssenceMagnet;
    return true;
  }
  if (s == "Merger's Gift") {
    out = RelicId::MergersGift;
    return true;
  }
  if (s == "Warped Time") {
    out = RelicId::WarpedTime;
    return true;
  }
  if (s == "Pack Instinct") {
    out = RelicId::PackInstinct;
    return true;
  }
  if (s == "Eruption Core") {
    out = RelicId::EruptionCore;
    return true;
  }
  if (s == "Chain Strike") {
    out = RelicId::ChainStrike;
    return true;
  }
  if (s == "Void Lens") {
    out = RelicId::VoidLens;
    return true;
  }
  if (s == "Living Wall") {
    out = RelicId::LivingWall;
    return true;
  }
  if (s == "Apex Hunger") {
    out = RelicId::ApexHunger;
    return true;
  }
  if (s == "Twin Pulse") {
    out = RelicId::TwinPulse;
    return true;
  }
  if (s == "Cold Bloom") {
    out = RelicId::ColdBloom;
    return true;
  }
  if (s == "Resonant Growth") {
    out = RelicId::ResonantGrowth;
    return true;
  }
  if (s == "Chaos Spark") {
    out = RelicId::ChaosSpark;
    return true;
  }
  if (s == "Eternal Echo") {
    out = RelicId::EternalEcho;
    return true;
  }
  if (s == "Recursive Merge") {
    out = RelicId::RecursiveMerge;
    return true;
  }
  if (s == "Shard Hunger") {
    out = RelicId::ShardHunger;
    return true;
  }
  if (s == "Death Bloom") {
    out = RelicId::DeathBloom;
    return true;
  }
  if (s == "The Quiet") {
    out = RelicId::TheQuiet;
    return true;
  }
  if (s == "None") {
    out = RelicId::None;
    return true;
  }
  return false;
}

} // namespace tower_swarm

