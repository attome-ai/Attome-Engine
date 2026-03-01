#ifndef ASHLANDS_MILITARY_H
#define ASHLANDS_MILITARY_H

#include "../engine/ATMEngine.h"
#include <array>
#include <string>
#include <vector>


// 10 unit types per faction
enum class UnitType {
  InfantryTier1 = 0,
  InfantryTier2,
  InfantryTier3, // Anti-Cav
  CavalryTier1,
  CavalryTier2,
  CavalryTier3, // Heavy
  Scout,
  Ram,
  Catapult,
  Commander,
  NUM_UNITS
};

struct UnitDef {
  std::string name;
  int attack;
  int defense_infantry;
  int defense_cavalry;
  int speed; // tiles per hour
  int carry_capacity;
  int ember_upkeep;
  int cost_timber;
  int cost_stone;
  int cost_ore;
  int cost_ember;
  int build_time; // base seconds
};

class MilitaryDB {
public:
  static const UnitDef &getUnitDef(Faction faction, UnitType type) {
    static std::vector<std::vector<UnitDef>> defs = initializeDefs();
    return defs[(int)faction][(int)type];
  }

private:
  static std::vector<std::vector<UnitDef>> initializeDefs() {
    std::vector<std::vector<UnitDef>> all_defs(4); // 3 playable + None
    for (int f = 0; f < 4; ++f) {
      all_defs[f].resize((int)UnitType::NUM_UNITS);
    }

    // Ironborn (Roman-esque)
    // High def, high cost
    all_defs[(int)Faction::Ironborn][(int)UnitType::InfantryTier1] = {
        "Legionnaire", 40, 35, 50, 6, 40, 1, 120, 100, 150, 30, 1600};
    all_defs[(int)Faction::Ironborn][(int)UnitType::InfantryTier2] = {
        "Praetorian", 30, 65, 35, 5, 20, 1, 100, 130, 160, 70, 1700};
    all_defs[(int)Faction::Ironborn][(int)UnitType::InfantryTier3] = {
        "Imperian", 70, 40, 25, 7, 50, 1, 150, 160, 210, 80, 1900};
    all_defs[(int)Faction::Ironborn][(int)UnitType::CavalryTier1] = {
        "Equites Legati", 0, 20, 10, 16, 0, 2, 140, 160, 20, 40, 2200 // Scout
    };
    all_defs[(int)Faction::Ironborn][(int)UnitType::CavalryTier2] = {
        "Equites Imperatoris",
        120,
        65,
        50,
        14,
        100,
        3,
        550,
        440,
        320,
        100,
        3200};
    all_defs[(int)Faction::Ironborn][(int)UnitType::CavalryTier3] = {
        "Equites Caesaris", 180, 80, 105, 10, 70, 4, 550, 640, 800, 180, 4400};
    all_defs[(int)Faction::Ironborn][(int)UnitType::Ram] = {
        "Battering Ram", 60, 30, 75, 4, 0, 3, 900, 360, 500, 70, 4600};
    all_defs[(int)Faction::Ironborn][(int)UnitType::Catapult] = {
        "Fire Catapult", 75, 60, 10, 3, 0, 6, 950, 1350, 600, 90, 9000};

    // TODO: Map out Thornkin, Ashwalkers
    // Using Ironborn copies for now to prevent crashes
    all_defs[(int)Faction::Thornkin] = all_defs[(int)Faction::Ironborn];
    all_defs[(int)Faction::Ashwalkers] = all_defs[(int)Faction::Ironborn];
    all_defs[(int)Faction::None] = all_defs[(int)Faction::Ironborn];

    return all_defs;
  }
};

// Represents an army on the move or stationed
struct TroopCount {
  std::array<int, (int)UnitType::NUM_UNITS> counts = {0};

  int getTotalSupplyUpkeep(Faction faction) const {
    int total = 0;
    for (int i = 0; i < (int)UnitType::NUM_UNITS; ++i) {
      if (counts[i] > 0) {
        total += MilitaryDB::getUnitDef(faction, (UnitType)i).ember_upkeep *
                 counts[i];
      }
    }
    return total;
  }
};

// Represents a troop movement (Attack, Raid, Reinforcement, Return)
enum class MovementType { Attack, Raid, Reinforcement, Return };

#endif
