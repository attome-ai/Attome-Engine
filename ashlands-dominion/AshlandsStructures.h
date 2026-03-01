#ifndef ASHLANDS_STRUCTURES_H
#define ASHLANDS_STRUCTURES_H

#include <string>
#include <vector>


enum class StructureType {
  CentralForge = 0,
  WarGrounds,
  Barracks,
  BeastPen,
  SiegeWorks,
  WarCollege,
  Armorsmith,
  TradingPost,
  EmbassyHall,
  Vault,
  EmberPool,
  GrandVault,
  GrandEmberPool,
  HiddenCache,
  FestivalHall,
  Outpost,
  Citadel,
  RelicVault,
  CaravanOffice,
  CommandersLodge,
  Rampart,
  SnarePit,
  ProvingGrounds,
  Runestone,
  LookoutTower,
  RiftConduit,
  NUM_STRUCTURES
};

struct Requirement {
  StructureType type;
  uint8_t level;
};

struct StructureDef {
  std::string name;
  double base_time;
  double base_timber;
  double base_stone;
  double base_ore;
  double base_ember;
  double cost_multiplier;
  double time_multiplier;
  std::vector<Requirement> requirements;
  int max_level;
};

class StructureDB {
public:
  static std::vector<StructureDef> getDefs() {
    std::vector<StructureDef> defs(
        static_cast<int>(StructureType::NUM_STRUCTURES));

    // Note: Prices scaled around Travian-like values.
    defs[(int)StructureType::CentralForge] = {
        "Central Forge", 100.0, 70, 40, 60, 20, 1.28, 1.20, {}, 20};

    defs[(int)StructureType::WarGrounds] = {"War Grounds",
                                            150.0,
                                            210,
                                            140,
                                            260,
                                            110,
                                            1.28,
                                            1.20,
                                            {{StructureType::CentralForge, 3}},
                                            20};

    defs[(int)StructureType::Barracks] = {
        "Barracks",
        180.0,
        100,
        130,
        150,
        70,
        1.28,
        1.20,
        {{StructureType::WarGrounds, 1}, {StructureType::CentralForge, 3}},
        20};

    defs[(int)StructureType::Vault] = {
        "Vault", 120.0, 130,
        160,     90,    40,
        1.28,    1.20,  {{StructureType::CentralForge, 1}},
        20};

    defs[(int)StructureType::EmberPool] = {"Ember Pool",
                                           120.0,
                                           130,
                                           160,
                                           90,
                                           40,
                                           1.28,
                                           1.20,
                                           {{StructureType::CentralForge, 1}},
                                           20};
    // Add more structures later as needed. For now the basic 5 are here.
    return defs;
  }
};

struct BuildJob {
  StructureType type;
  uint8_t target_level;
  float time_remaining;
};

#endif
