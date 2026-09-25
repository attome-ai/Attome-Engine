#pragma once

// Game tuning values. Every ATM_TUNABLE below is a mutable global whose
// compiled-in value is the default; config/tower_swarm.json overrides it at
// startup and while the game runs (atm::Tunables::reloadIfChanged()). JSON
// keys are "<section>.<name>", e.g. "level.kBaseHp"; values declared directly
// in namespace tower_swarm live in section "general". Colors are [r, g, b, a].
//
// Values that are only read once at startup (window/world/tile size, pool
// capacities) need a restart to take effect.

#include "ATMConfig.h"

#include <cstdint>

namespace tower_swarm {

struct Rgba8 {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

} // namespace tower_swarm

// Colors are stored as [r, g, b, a] arrays with components in 0..255.
template <> struct atm::TunableTraits<tower_swarm::Rgba8> {
  static bool fromJson(const atm::Json &j, tower_swarm::Rgba8 &out) {
    if (!j.isArray() || j.size() != 4)
      return false;
    std::uint8_t c[4] = {};
    for (int i = 0; i < 4; ++i) {
      const atm::Json &v = j.asArray()[static_cast<std::size_t>(i)];
      if (!atm::TunableTraits<std::uint8_t>::fromJson(v, c[i]))
        return false;
    }
    out = tower_swarm::Rgba8{c[0], c[1], c[2], c[3]};
    return true;
  }
  static atm::Json toJson(const tower_swarm::Rgba8 &v) {
    atm::Json j = atm::Json::array();
    j.push(static_cast<int>(v.r));
    j.push(static_cast<int>(v.g));
    j.push(static_cast<int>(v.b));
    j.push(static_cast<int>(v.a));
    return j;
  }
};

namespace tower_swarm {
ATM_TUNABLE_SECTION("general");

// -----------------------------------------------------------------------------
// Phase 0/1 — Bootstrap + World/Camera
// -----------------------------------------------------------------------------

ATM_TUNABLE(int, kWindowWidthPx, 1280);
ATM_TUNABLE(int, kWindowHeightPx, 720);

ATM_TUNABLE(int, kWorldWidthPx, 5120);
ATM_TUNABLE(int, kWorldHeightPx, 2880);

ATM_TUNABLE(int, kTileSizePx, 64);

ATM_TUNABLE(float, kMaxFrameDtSec, 0.05f);
ATM_TUNABLE(float, kDefaultFrameDtSec, 1.0f / 60.0f);

ATM_TUNABLE(float, kCameraPanSpeedPxPerSec, 1500.0f);
ATM_TUNABLE(float, kCameraSmoothRate, 14.0f);
ATM_TUNABLE(float, kCameraDefaultZoom, 1.0f);
ATM_TUNABLE(float, kMinCameraZoomEpsilon, 0.0001f);

ATM_TUNABLE(int, kHudTopBarHeightPx, 44);
ATM_TUNABLE(int, kHudPaddingPx, 12);
ATM_TUNABLE(int, kBaseSizePx, 144);
ATM_TUNABLE(float, kBaseRadiusPx, 72.0f);
ATM_TUNABLE(int, kCreatureBaseSizePx, 48);
ATM_TUNABLE(int, kEnemyBaseSizePx, 44);
ATM_TUNABLE(int, kProjectileSizePx, 10);
ATM_TUNABLE(int, kPickupSizePx, 12);

ATM_TUNABLE(float, kProjectileHitRadiusPx, 8.0f);
ATM_TUNABLE(float, kProjectileDefaultLifetimeSec, 3.5f);
ATM_TUNABLE(float, kProjectileMinLifetimeSec, 0.05f);
ATM_TUNABLE(float, kProjectileEnemyHitPaddingFactor, 0.25f);
ATM_TUNABLE(float, kProjectileSpeedPxPerSec, 900.0f);
ATM_TUNABLE(float, kCreaturePickRadiusFactor, 0.60f);
ATM_TUNABLE(float, kPickupFloatUpSec, 2.0f);
ATM_TUNABLE(float, kPickupFloatUpPx, 30.0f);
ATM_TUNABLE(float, kPickupAttractRadiusPx, 80.0f);
ATM_TUNABLE(float, kPickupHomingSpeedPxPerSec, 520.0f);
ATM_TUNABLE(float, kPickupCollectDistancePx, 16.0f);

ATM_TUNABLE(int, kCreaturePoolCapacity, 2048);
ATM_TUNABLE(int, kEnemyPoolCapacity, 8192);
ATM_TUNABLE(int, kProjectilePoolCapacity, 10000);
ATM_TUNABLE(int, kPickupPoolCapacity, 4096);

ATM_TUNABLE(int, kBaseHpBarWidthPx, 420);
ATM_TUNABLE(int, kBaseHpBarHeightPx, 18);
ATM_TUNABLE(int, kBaseHpBarMarginBottomPx, 18);
ATM_TUNABLE(int, kBaseHpBarInsetPx, 2);
ATM_TUNABLE(int, kBaseHpBarLabelOffsetYPx, 16);
ATM_TUNABLE(int, kConfirmDialogWidthPx, 360);
ATM_TUNABLE(int, kConfirmDialogHeightPx, 160);
ATM_TUNABLE(int, kConfirmDialogButtonWidthPx, 140);
ATM_TUNABLE(int, kConfirmDialogButtonHeightPx, 36);
ATM_TUNABLE(int, kConfirmDialogButtonGapPx, 16);
ATM_TUNABLE(int, kHudSecondaryTextOffsetYPx, 6);
ATM_TUNABLE(int, kModalPanelTextInsetXPx, 16);
ATM_TUNABLE(int, kModalPanelTextInsetYPx, 20);
ATM_TUNABLE(int, kModalPanelLineStepPx, 18);
ATM_TUNABLE(int, kModalButtonTextInsetXPx, 12);
ATM_TUNABLE(int, kModalButtonTextInsetYPx, 10);
ATM_TUNABLE(int, kSelectedCreaturePanelWidthPx, 360);
ATM_TUNABLE(int, kSelectedCreaturePanelHeightPx, 86);
ATM_TUNABLE(int, kSelectedCreaturePanelBarHeightPx, 10);
ATM_TUNABLE(int, kSelectedCreaturePanelBarInsetPx, 2);

ATM_TUNABLE(int, kEnemyHpBarHeightPx, 5);
ATM_TUNABLE(int, kEnemyHpBarOffsetYPx, 6);
ATM_TUNABLE(int, kEnemyHpBarInsetPx, 1);

ATM_TUNABLE(int, kRangeIndicatorSegments, 48);

ATM_TUNABLE(int, kWaveShopCardWidthPx, 260);
ATM_TUNABLE(int, kWaveShopCardHeightPx, 140);
ATM_TUNABLE(int, kWaveShopCardGapPx, 18);
ATM_TUNABLE(int, kWaveShopBottomMarginPx, 24);
ATM_TUNABLE(int, kWaveShopCardTextInsetXPx, 14);
ATM_TUNABLE(int, kWaveShopCardTextInsetYPx, 16);

ATM_TUNABLE(Rgba8, kClearColor, {6, 10, 24, 255});
ATM_TUNABLE(Rgba8, kHudTopBarColor, {10, 14, 26, 220});
ATM_TUNABLE(Rgba8, kHudBorderColor, {255, 255, 255, 18});
ATM_TUNABLE(Rgba8, kHudTextColor, {206, 234, 255, 255});
ATM_TUNABLE(Rgba8, kDebugGridColor, {255, 255, 255, 22});
ATM_TUNABLE(Rgba8, kBaseColor, {26, 36, 64, 255});
ATM_TUNABLE(Rgba8, kBaseHpBarBackColor, {8, 10, 18, 210});
ATM_TUNABLE(Rgba8, kBaseHpBarFillColor, {64, 204, 128, 235});
ATM_TUNABLE(Rgba8, kBaseHpBarOutlineColor, {255, 255, 255, 30});
ATM_TUNABLE(Rgba8, kBaseHpBarMarkerColor, {255, 255, 255, 44});
ATM_TUNABLE(Rgba8, kModalOverlayColor, {0, 0, 0, 160});
ATM_TUNABLE(Rgba8, kModalPanelColor, {14, 18, 30, 235});
ATM_TUNABLE(Rgba8, kModalButtonColor, {32, 44, 86, 235});
ATM_TUNABLE(Rgba8, kModalButtonHoverColor, {48, 70, 140, 245});
ATM_TUNABLE(Rgba8, kModalButtonTextColor, {230, 242, 255, 255});
ATM_TUNABLE(Rgba8, kGhostValidColor, {80, 220, 160, 90});
ATM_TUNABLE(Rgba8, kGhostInvalidColor, {220, 80, 80, 90});

ATM_TUNABLE(Rgba8, kEnemyHpBarBackColor, {8, 10, 18, 160});
ATM_TUNABLE(Rgba8, kEnemyHpBarFillColor, {220, 80, 80, 220});
ATM_TUNABLE(Rgba8, kEnemyHpBarOutlineColor, {255, 255, 255, 24});

ATM_TUNABLE(Rgba8, kRangeIndicatorColor, {120, 220, 255, 60});
ATM_TUNABLE(Rgba8, kEvolutionPulseColor, {255, 255, 255, 90});
ATM_TUNABLE(Rgba8, kSelectedCreatureBarBackColor, {8, 10, 18, 200});
ATM_TUNABLE(Rgba8, kSelectedCreatureBarFillColor, {120, 220, 255, 220});
ATM_TUNABLE(Rgba8, kMergeLinkColor, {255, 190, 90, 110});
ATM_TUNABLE(float, kMergeLinkPulseHz, 2.0f);

ATM_TUNABLE(Rgba8, kWaveShopCardColor, {20, 26, 42, 230});
ATM_TUNABLE(Rgba8, kWaveShopCardHoverColor, {30, 40, 70, 240});
ATM_TUNABLE(Rgba8, kWaveShopCardBorderColor, {255, 255, 255, 28});

ATM_TUNABLE(std::uint8_t, kZIndexTiles, 0);
ATM_TUNABLE(std::uint8_t, kZIndexBase, 1);
ATM_TUNABLE(std::uint8_t, kZIndexCreatures, 2);
ATM_TUNABLE(std::uint8_t, kZIndexEnemies, 3);
ATM_TUNABLE(std::uint8_t, kZIndexProjectiles, 4);
ATM_TUNABLE(std::uint8_t, kZIndexPickups, 5);

ATM_TUNABLE(Rgba8, kBiomeTileColorVerdantFields, {22, 64, 34, 255});
ATM_TUNABLE(Rgba8, kBiomeTileColorAshlands, {78, 46, 30, 255});
ATM_TUNABLE(Rgba8, kBiomeTileColorFrostmarsh, {34, 70, 92, 255});
ATM_TUNABLE(Rgba8, kBiomeTileColorDeepcore, {64, 24, 44, 255});
ATM_TUNABLE(Rgba8, kBiomeTileColorTheVoid, {12, 12, 18, 255});

ATM_TUNABLE(Rgba8, kCreatureColorBrix, {170, 170, 180, 255});
ATM_TUNABLE(Rgba8, kCreatureColorFlara, {240, 120, 40, 255});
ATM_TUNABLE(Rgba8, kCreatureColorMossling, {60, 200, 90, 255});
ATM_TUNABLE(Rgba8, kCreatureColorGlitch, {180, 60, 220, 255});
ATM_TUNABLE(Rgba8, kCreatureColorIronjaw, {200, 200, 160, 255});
ATM_TUNABLE(Rgba8, kCreatureColorWraith, {90, 90, 120, 255});
ATM_TUNABLE(Rgba8, kCreatureColorCrystalis, {80, 220, 230, 255});
ATM_TUNABLE(Rgba8, kCreatureColorVex, {230, 70, 150, 255});
ATM_TUNABLE(Rgba8, kCreatureColorOrin, {240, 210, 80, 255});
ATM_TUNABLE(Rgba8, kCreatureColorNull, {30, 30, 40, 255});

ATM_TUNABLE(Rgba8, kEnemyColorGrub, {100, 220, 120, 255});
ATM_TUNABLE(Rgba8, kEnemyColorHulk, {150, 100, 70, 255});
ATM_TUNABLE(Rgba8, kEnemyColorScuttle, {80, 200, 200, 255});
ATM_TUNABLE(Rgba8, kEnemyColorDriftwing, {120, 180, 240, 255});
ATM_TUNABLE(Rgba8, kEnemyColorDivide, {180, 130, 220, 255});
ATM_TUNABLE(Rgba8, kEnemyColorVanguard, {80, 120, 220, 255});
ATM_TUNABLE(Rgba8, kEnemyColorMender, {100, 240, 180, 255});
ATM_TUNABLE(Rgba8, kEnemyColorBoss, {220, 60, 60, 255});

ATM_TUNABLE(Rgba8, kProjectileColor, {240, 240, 255, 255});
ATM_TUNABLE(Rgba8, kPickupColor, {255, 224, 80, 255});

// -----------------------------------------------------------------------------
// GDD constants — gameplay math + tuning (used by later phases too)
// -----------------------------------------------------------------------------

enum class Biome : std::uint8_t {
  VerdantFields = 0,
  Ashlands = 1,
  Frostmarsh = 2,
  Deepcore = 3,
  TheVoid = 4,
  Count = 5
};

enum class MapTemplate : std::uint8_t {
  OpenField = 0,
  Chokepoint = 1,
  SplitPath = 2,
  Island = 3,
  Maze = 4,
  Count = 5
};

namespace level {
ATM_TUNABLE_SECTION("level");
ATM_TUNABLE(int, kBaseHp, 100);
ATM_TUNABLE(float, kStar3Threshold, 0.70f);
ATM_TUNABLE(float, kStar2Threshold, 0.30f);

ATM_TUNABLE(int, kBaseWaveCount, 5);
ATM_TUNABLE(float, kWaveCountPerLevel, 0.5f);

ATM_TUNABLE(float, kDifficultyBase, 1.18f);

ATM_TUNABLE(int, kMapTemplateCount, 5);
ATM_TUNABLE(int, kBiomeLevelsPer, 10);
ATM_TUNABLE(int, kEliteEveryLevels, 5);

ATM_TUNABLE(float, kWaveEnemyCountBase, 3.0f);
ATM_TUNABLE(float, kWaveEnemyCountLinear, 1.8f);
ATM_TUNABLE(float, kWaveEnemyCountQuadratic, 0.05f);
ATM_TUNABLE(float, kWaveEnemyCountWaveFactor, 0.15f);

ATM_TUNABLE(float, kWaveEnemyHpWaveFactor, 0.10f);

ATM_TUNABLE(float, kWaveEnemySpeedExponent, 0.4f);
ATM_TUNABLE(float, kWaveEnemySpeedWaveFactor, 0.04f);
ATM_TUNABLE(float, kWaveEnemySpeedMax, 3.0f);

// Wave spawner pacing.
ATM_TUNABLE(float, kInterSpawnDelaySec, 0.08f);

// Banners (Production TODO Â§7.1).
ATM_TUNABLE(float, kWaveStartBannerDurationSec, 1.5f);
ATM_TUNABLE(float, kWaveClearBannerDurationSec, 2.0f);
ATM_TUNABLE(float, kBossWaveBannerDurationSec, 3.0f);
ATM_TUNABLE(float, kLevelStartBannerDurationSec, 2.0f);
ATM_TUNABLE(float, kLevelClearBannerDurationSec, 2.0f);
ATM_TUNABLE(float, kLevelFailedBannerDurationSec, 2.0f);

ATM_TUNABLE(int, kMilestoneTutorialLevel, 1);
ATM_TUNABLE(int, kMilestoneFirstEliteLevel, 5);
ATM_TUNABLE(int, kMilestoneBiome2Level, 10);
ATM_TUNABLE(int, kMilestoneChargerShopLevel, 15);
ATM_TUNABLE(int, kMilestoneMidBossLevel, 25);
ATM_TUNABLE(int, kMilestoneVoidPreviewLevel, 50);
ATM_TUNABLE(int, kMilestoneHallOfFameLevel, 100);
ATM_TUNABLE(int, kMilestoneMasteryTagLevel, 200);
} // namespace level

namespace evolution {
ATM_TUNABLE_SECTION("evolution");

// compile-time (not in JSON): std::array bound in CreatureContainer/TowerSwarmGame.
constexpr int kVisualBandCount = 7;

ATM_TUNABLE(int, kKillsTier1To2, 10);
ATM_TUNABLE(int, kKillsTier2To3, 30);
ATM_TUNABLE(int, kKillsTier3To4, 80);
ATM_TUNABLE(int, kKillsTier4To5, 200);
ATM_TUNABLE(float, kKillsInfiniteBase, 10.0f);
ATM_TUNABLE(float, kKillsInfiniteGrowth, 2.5f);

ATM_TUNABLE(float, kHpExponent, 1.4f);
ATM_TUNABLE(float, kDamageExponent, 1.3f);
ATM_TUNABLE(float, kRangeExponent, 0.5f);
ATM_TUNABLE(float, kRangeCapPx, 600.0f);
ATM_TUNABLE(float, kAttackRateExponent, 0.4f);
ATM_TUNABLE(float, kAttackRateCapPerSec, 8.0f);
ATM_TUNABLE(float, kMoveSpeedExponent, 0.2f);

ATM_TUNABLE(float, kTierSize1To3, 1.0f);
ATM_TUNABLE(float, kTierSize4To6, 1.3f);
ATM_TUNABLE(float, kTierSize7To9, 1.6f);
ATM_TUNABLE(float, kTierSize10To12, 2.0f);
ATM_TUNABLE(float, kTierSize13To15, 2.4f);
ATM_TUNABLE(float, kTierSize16To19, 2.8f);
ATM_TUNABLE(float, kTierSize20Plus, 3.0f);
ATM_TUNABLE(float, kTierSizeCap, 3.0f);

ATM_TUNABLE(float, kEvolutionPulseScale, 1.5f);
ATM_TUNABLE(float, kEvolutionAnimSec, 0.8f);
ATM_TUNABLE(float, kEvolutionFloatingTextSec, 1.25f);
ATM_TUNABLE(float, kEvolutionFloatingTextRisePxPerSec, 40.0f);
ATM_TUNABLE(float, kScreenEdgeGlowSec, 1.0f);
} // namespace evolution

namespace merge {
ATM_TUNABLE_SECTION("merge");
ATM_TUNABLE(float, kCooldownSec, 3.0f);
ATM_TUNABLE(float, kAutoMergeIdleSec, 6.0f);
ATM_TUNABLE(float, kAnimationSec, 0.8f);
ATM_TUNABLE(int, kEssenceBonus, 10);
ATM_TUNABLE(float, kKillInheritanceDivisor, 2.0f);
} // namespace merge

namespace movement_ai {
ATM_TUNABLE_SECTION("movement_ai");
ATM_TUNABLE(float, kRecalcIntervalSec, 3.0f);

ATM_TUNABLE(float, kThreatRadiusNearPx, 200.0f);
ATM_TUNABLE(float, kThreatRadiusMidPx, 400.0f);
ATM_TUNABLE(float, kThreatRadiusFarPx, 600.0f);
ATM_TUNABLE(float, kThreatWeightNear, 3.0f);
ATM_TUNABLE(float, kThreatWeightMid, 2.0f);
ATM_TUNABLE(float, kThreatWeightFar, 1.0f);

ATM_TUNABLE(float, kSupportRepelRadiusPx, 96.0f);

ATM_TUNABLE(float, kDesiredMoveThresholdPx, 64.0f);
ATM_TUNABLE(float, kWaypointInterpSec, 1.5f);

ATM_TUNABLE(float, kPlayerDragStartThresholdPx, 8.0f);
ATM_TUNABLE(float, kPlayerDragStunSec, 0.5f);
} // namespace movement_ai

namespace characters {
ATM_TUNABLE_SECTION("characters");
ATM_TUNABLE(int, kEvolutionStage1MaxTier, 3);
ATM_TUNABLE(int, kEvolutionStage2MaxTier, 6);
ATM_TUNABLE(int, kEvolutionStage3MaxTier, 9);
ATM_TUNABLE(int, kEvolutionStage4MinTier, 10);

namespace base_stats {
ATM_TUNABLE_SECTION("characters.base_stats");
ATM_TUNABLE(float, kBrixBaseHp, 70.0f);
ATM_TUNABLE(float, kBrixBaseDamage, 12.0f);
ATM_TUNABLE(float, kBrixBaseRangePx, 220.0f);
ATM_TUNABLE(float, kBrixBaseAttackRatePerSec, 1.5f);
ATM_TUNABLE(float, kBrixBaseMoveSpeedPxPerSec, 120.0f);

ATM_TUNABLE(float, kFlaraBaseHp, 55.0f);
ATM_TUNABLE(float, kFlaraBaseDamage, 8.0f);
ATM_TUNABLE(float, kFlaraBaseRangePx, 180.0f);
ATM_TUNABLE(float, kFlaraBaseAttackRatePerSec, 0.8f);
ATM_TUNABLE(float, kFlaraBaseMoveSpeedPxPerSec, 125.0f);
ATM_TUNABLE(float, kFlaraSplashRadiusPx, 80.0f);

ATM_TUNABLE(float, kMosslingBaseHp, 85.0f);
ATM_TUNABLE(float, kMosslingBaseDamage, 4.0f);
ATM_TUNABLE(float, kMosslingBaseRangePx, 160.0f);
ATM_TUNABLE(float, kMosslingBaseAttackRatePerSec, 1.0f);
ATM_TUNABLE(float, kMosslingBaseMoveSpeedPxPerSec, 120.0f);

ATM_TUNABLE(float, kGlitchBaseHp, 65.0f);
ATM_TUNABLE(float, kGlitchBaseDamage, 6.0f);
ATM_TUNABLE(float, kGlitchBaseRangePx, 200.0f);
ATM_TUNABLE(float, kGlitchBaseAttackRatePerSec, 0.7f);
ATM_TUNABLE(float, kGlitchBaseMoveSpeedPxPerSec, 120.0f);
ATM_TUNABLE(float, kGlitchSlowFieldRadiusPx, 60.0f);

ATM_TUNABLE(float, kIronjawBaseHp, 110.0f);
ATM_TUNABLE(float, kIronjawBaseDamage, 10.0f);
ATM_TUNABLE(float, kIronjawBaseRangePx, 80.0f);
ATM_TUNABLE(float, kIronjawBaseAttackRatePerSec, 1.2f);
ATM_TUNABLE(float, kIronjawBaseMoveSpeedPxPerSec, 160.0f);
ATM_TUNABLE(float, kIronjawChargeRangePx, 300.0f);
ATM_TUNABLE(float, kIronjawChargeDamage, 30.0f);
ATM_TUNABLE(float, kIronjawChargeKnockbackPx, 80.0f);

ATM_TUNABLE(float, kWraithBaseHp, 45.0f);
ATM_TUNABLE(float, kWraithBaseDamage, 40.0f);
ATM_TUNABLE(float, kWraithBaseRangePx, 500.0f);
ATM_TUNABLE(float, kWraithBaseAttackRatePerSec, 0.3f);
ATM_TUNABLE(float, kWraithBaseMoveSpeedPxPerSec, 120.0f);

ATM_TUNABLE(float, kCrystalisBaseHp, 80.0f);
ATM_TUNABLE(float, kCrystalisBaseDamage, 15.0f);
ATM_TUNABLE(float, kCrystalisBaseRangePx, 280.0f);
ATM_TUNABLE(float, kCrystalisBaseAttackRatePerSec, 0.9f);
ATM_TUNABLE(float, kCrystalisBaseMoveSpeedPxPerSec, 120.0f);
ATM_TUNABLE(float, kCrystalisAuraRangeBoost, 0.15f);

ATM_TUNABLE(float, kVexBaseHp, 75.0f);
ATM_TUNABLE(float, kVexBaseDamage, 10.0f);
ATM_TUNABLE(float, kVexBaseRangePx, 220.0f);
ATM_TUNABLE(float, kVexBaseAttackRatePerSec, 0.9f);
ATM_TUNABLE(float, kVexBaseMoveSpeedPxPerSec, 130.0f);

ATM_TUNABLE(float, kOrinBaseHp, 160.0f);
ATM_TUNABLE(float, kOrinBaseDamage, 18.0f);
ATM_TUNABLE(float, kOrinBaseRangePx, 240.0f);
ATM_TUNABLE(float, kOrinBaseAttackRatePerSec, 0.6f);
ATM_TUNABLE(float, kOrinBaseMoveSpeedPxPerSec, 110.0f);

ATM_TUNABLE(float, kNullBaseHp, 140.0f);
ATM_TUNABLE(float, kNullBaseDamage, 14.0f);
ATM_TUNABLE(float, kNullBaseRangePx, 200.0f);
ATM_TUNABLE(float, kNullBaseAttackRatePerSec, 0.7f);
ATM_TUNABLE(float, kNullBaseMoveSpeedPxPerSec, 115.0f);
ATM_TUNABLE(float, kNullDrainRadiusPx, 180.0f);
} // namespace base_stats

namespace brix {
ATM_TUNABLE_SECTION("characters.brix");
ATM_TUNABLE(int, kPierceStage2, 1);
ATM_TUNABLE(int, kPierceStage3, 3);
ATM_TUNABLE(float, kStage3RangeBonus, 0.20f);
ATM_TUNABLE(float, kStage4SplashRadiusPx, 60.0f);
ATM_TUNABLE(float, kSignatureCooldownSec, 15.0f);
ATM_TUNABLE(float, kSignatureLineLengthPx, 150.0f);
} // namespace brix

namespace flara {
ATM_TUNABLE_SECTION("characters.flara");

// compile-time (not in JSON): std::array bound in CreatureContainer.cpp.
constexpr int kStage4SimultaneousTargets = 3;

ATM_TUNABLE(float, kBurningGroundStage2Sec, 2.0f);
ATM_TUNABLE(float, kBurningGroundStage3Sec, 4.0f);
ATM_TUNABLE(float, kSignatureCooldownSec, 20.0f);
ATM_TUNABLE(float, kSignatureRadiusPx, 300.0f);
ATM_TUNABLE(float, kSignatureDamageMultiplier, 5.0f);
} // namespace flara

namespace mossling {
ATM_TUNABLE_SECTION("characters.mossling");
ATM_TUNABLE(float, kAuraRadiusStage1Px, 96.0f);
ATM_TUNABLE(float, kAuraAttackSpeedStage1, 0.05f);
ATM_TUNABLE(float, kAuraAttackSpeedStage2, 0.10f);
ATM_TUNABLE(float, kAuraDamageStage2, 0.08f);
ATM_TUNABLE(float, kAuraHealStage3HpPerSec, 2.0f);
ATM_TUNABLE(float, kAuraRadiusStage4Px, 200.0f);
ATM_TUNABLE(float, kSignatureCooldownSec, 25.0f);
} // namespace mossling

namespace glitch {
ATM_TUNABLE_SECTION("characters.glitch");
ATM_TUNABLE(float, kSlowFieldSpeedMultiplier, 0.50f);
ATM_TUNABLE(float, kSlowFieldDurationSec, 3.0f);
ATM_TUNABLE(float, kOrbDetonateAfterSec, 4.0f);
ATM_TUNABLE(float, kSignatureCooldownSec, 18.0f);
ATM_TUNABLE(float, kSignatureFreezeRadiusPx, 250.0f);
ATM_TUNABLE(float, kSignatureFreezeDurationSec, 2.5f);
} // namespace glitch

namespace ironjaw {
ATM_TUNABLE_SECTION("characters.ironjaw");
ATM_TUNABLE(float, kSignatureCooldownSec, 22.0f);
ATM_TUNABLE(float, kSignatureFrenzyDurationSec, 4.0f);
ATM_TUNABLE(float, kSignatureAttackSpeedMultiplier, 3.0f);
} // namespace ironjaw

namespace wraith {
ATM_TUNABLE_SECTION("characters.wraith");
ATM_TUNABLE(float, kArmorIgnoreFraction, 0.30f);
ATM_TUNABLE(float, kExecuteBelowHpFraction, 0.15f);
ATM_TUNABLE(float, kSignatureCooldownSec, 30.0f);
ATM_TUNABLE(float, kSignatureMarkDurationSec, 4.0f);
} // namespace wraith

namespace crystalis {
ATM_TUNABLE_SECTION("characters.crystalis");
ATM_TUNABLE(int, kStage2RefractTargets, 2);
ATM_TUNABLE(int, kStage3RefractTargets, 4);
ATM_TUNABLE(float, kSignatureCooldownSec, 20.0f);
} // namespace crystalis

namespace vex {
ATM_TUNABLE_SECTION("characters.vex");
ATM_TUNABLE(float, kRandomAbilityIntervalSec, 5.0f);
ATM_TUNABLE(float, kStage3AbilityStrengthMultiplier, 2.0f);
ATM_TUNABLE(float, kSignatureCooldownSec, 25.0f);
} // namespace vex

namespace orin {
ATM_TUNABLE_SECTION("characters.orin");
ATM_TUNABLE(float, kPassiveIgnoreDamageChanceStage1, 0.05f);
ATM_TUNABLE(float, kPassiveIgnoreDamageChanceStage2, 0.15f);
ATM_TUNABLE(float, kPassiveBaseShieldStage3, 0.25f);
ATM_TUNABLE(float, kSignatureCooldownSec, 60.0f);
ATM_TUNABLE(float, kSignatureFreezeDurationSec, 5.0f);
} // namespace orin

namespace null_seed {
ATM_TUNABLE_SECTION("characters.null_seed");
ATM_TUNABLE(float, kDrainDamageStage1, 0.10f);
ATM_TUNABLE(float, kDrainDamageStage2, 0.25f);
ATM_TUNABLE(float, kDrainSpeedStage2, 0.15f);
ATM_TUNABLE(float, kDrainDamageStage3, 1.00f);
ATM_TUNABLE(float, kSignatureCooldownSec, 45.0f);
} // namespace null_seed
} // namespace characters

namespace enemies {
ATM_TUNABLE_SECTION("enemies");
ATM_TUNABLE(int, kIntroLevelGrub, 1);
ATM_TUNABLE(int, kIntroLevelHulk, 2);
ATM_TUNABLE(int, kIntroLevelScuttle, 4);
ATM_TUNABLE(int, kIntroLevelDriftwing, 7);
ATM_TUNABLE(int, kIntroLevelDivide, 11);
ATM_TUNABLE(int, kIntroLevelVanguard, 16);
ATM_TUNABLE(int, kIntroLevelMender, 22);

ATM_TUNABLE(float, kHulkFrontDamageTakenMultiplier, 0.50f);

ATM_TUNABLE(int, kScuttlePackMin, 15);
ATM_TUNABLE(int, kScuttlePackMax, 30);

ATM_TUNABLE(int, kDivideChildrenCount, 2);
ATM_TUNABLE(float, kDivideChildHpFactor, 0.40f);

ATM_TUNABLE(float, kVanguardFrontResist, 0.80f);

ATM_TUNABLE(float, kMenderHealHpPerSec, 8.0f);
ATM_TUNABLE(float, kMenderHealRadiusPx, 120.0f);

ATM_TUNABLE(float, kBossHpMultiplier, 50.0f);
ATM_TUNABLE(float, kBossPhase2Threshold, 0.66f);
ATM_TUNABLE(float, kBossPhase3Threshold, 0.33f);
ATM_TUNABLE(int, kBossPhase2SpawnGrubs, 20);
ATM_TUNABLE(int, kBossPhase3SpawnHulks, 10);
ATM_TUNABLE(float, kBossPhase2SpeedBonus, 0.30f);
ATM_TUNABLE(float, kBossBaseSpeedMultiplier, 0.65f);
ATM_TUNABLE(int, kBossDeathEssenceBase, 150);
ATM_TUNABLE(int, kBossDeathEssencePerLevel, 8);
ATM_TUNABLE(float, kBossBaseDamageToBase, 15.0f);
ATM_TUNABLE(float, kBossStompRadiusPx, 120.0f);
ATM_TUNABLE(float, kBossStompIntervalSec, 2.25f);
ATM_TUNABLE(float, kBossStompInitialDelaySec, 0.75f);
ATM_TUNABLE(float, kBossStompDamageToCreatureMultiplier, 1.25f);

ATM_TUNABLE(float, kEliteHpMultiplier, 1.50f);
ATM_TUNABLE(float, kEliteSpeedMultiplier, 1.20f);

ATM_TUNABLE(float, kScalingSpeedExponent, 0.4f);
ATM_TUNABLE(float, kScalingSpeedCap, 3.0f);
ATM_TUNABLE(float, kScalingDamagePerLevel, 0.05f);
ATM_TUNABLE(float, kScalingRewardPerLevel, 0.08f);

ATM_TUNABLE(float, kNewEnemyIntroPauseSec, 2.0f);

ATM_TUNABLE(float, kFrontHitDotThreshold, 0.25f);

ATM_TUNABLE(float, kDivideChildSpawnOffsetXPx, 12.0f);
ATM_TUNABLE(float, kDivideChildSpawnOffsetYPx, 10.0f);

ATM_TUNABLE(float, kGrubBaseHp, 30.0f);
ATM_TUNABLE(float, kGrubBaseSpeedPxPerSec, 90.0f);
ATM_TUNABLE(int, kGrubBaseRewardEssence, 5);
ATM_TUNABLE(float, kGrubBaseDamageToBase, 1.0f);

ATM_TUNABLE(float, kHulkBaseHp, 250.0f);
ATM_TUNABLE(float, kHulkBaseSpeedPxPerSec, 35.0f);
ATM_TUNABLE(int, kHulkBaseRewardEssence, 20);
ATM_TUNABLE(float, kHulkBaseDamageToBase, 4.0f);

ATM_TUNABLE(float, kScuttleBaseHp, 12.0f);
ATM_TUNABLE(float, kScuttleBaseSpeedPxPerSec, 110.0f);
ATM_TUNABLE(int, kScuttleBaseRewardEssence, 2);
ATM_TUNABLE(float, kScuttleBaseDamageToBase, 1.0f);

ATM_TUNABLE(float, kDriftwingBaseHp, 60.0f);
ATM_TUNABLE(float, kDriftwingBaseSpeedPxPerSec, 70.0f);
ATM_TUNABLE(int, kDriftwingBaseRewardEssence, 12);
ATM_TUNABLE(float, kDriftwingBaseDamageToBase, 2.0f);

ATM_TUNABLE(float, kDivideBaseHp, 80.0f);
ATM_TUNABLE(float, kDivideBaseSpeedPxPerSec, 55.0f);
ATM_TUNABLE(int, kDivideBaseRewardEssence, 15);
ATM_TUNABLE(float, kDivideBaseDamageToBase, 2.0f);

ATM_TUNABLE(float, kVanguardBaseHp, 150.0f);
ATM_TUNABLE(float, kVanguardBaseSpeedPxPerSec, 50.0f);
ATM_TUNABLE(int, kVanguardBaseRewardEssence, 18);
ATM_TUNABLE(float, kVanguardBaseDamageToBase, 3.0f);

ATM_TUNABLE(float, kMenderBaseHp, 40.0f);
ATM_TUNABLE(float, kMenderBaseSpeedPxPerSec, 40.0f);
ATM_TUNABLE(int, kMenderBaseRewardEssence, 10);
ATM_TUNABLE(float, kMenderBaseDamageToBase, 2.0f);
} // namespace enemies

namespace wave_shop {
ATM_TUNABLE_SECTION("wave_shop");

// compile-time (not in JSON): used in the constexpr card table in
// WaveBuffShop.cpp.
constexpr int kSurgeDurationWaves = 4;
constexpr int kFrenziedBloodDurationWaves = 3;
constexpr int kSlowTideDurationWaves = 1;
constexpr int kForesightDurationWaves = 1;
constexpr int kEchoStrikeDurationWaves = 3;
constexpr int kIronSkinDurationWaves = 2;
constexpr int kApexHunterDurationWaves = 1;
constexpr int kVoidPulseDurationWaves = 1;

ATM_TUNABLE(int, kCardPoolSize, 12);
ATM_TUNABLE(int, kCardsDrawnPerWaveClear, 3);
ATM_TUNABLE(float, kSelectionTimerSec, 5.0f);

ATM_TUNABLE(float, kSurgeAttackSpeedBonus, 0.25f);

ATM_TUNABLE(int, kFortifyBaseHpBonus, 15);

ATM_TUNABLE(int, kFrenziedBloodEssencePerKill, 1);

ATM_TUNABLE(float, kSlowTideSpeedMultiplier, 0.65f);

ATM_TUNABLE(float, kMendHealFraction, 0.50f);

ATM_TUNABLE(int, kWildSeedTier, 2);

ATM_TUNABLE(float, kEchoStrikeDamageRepeatFraction, 0.20f);
ATM_TUNABLE(float, kEchoStrikeRepeatDelaySec, 0.30f);

ATM_TUNABLE(float, kEssenceCacheFraction, 0.30f);

ATM_TUNABLE(float, kIronSkinDamageTakenMultiplier, 0.80f);

ATM_TUNABLE(float, kApexHunterDamageBonus, 0.50f);

ATM_TUNABLE(int, kVoidPulseKillInterval, 10);
ATM_TUNABLE(float, kVoidPulseRadiusPx, 80.0f);
ATM_TUNABLE(float, kVoidPulseDamageFractionOfKilledEnemyMaxHp, 0.50f);
} // namespace wave_shop

namespace inter_level_shop {
ATM_TUNABLE_SECTION("inter_level_shop");

// compile-time (not in JSON): std::array bound in TowerSwarmGame.h.
constexpr int kBazaarOfferCount = 4;

ATM_TUNABLE(int, kRerollCostEssence, 15);

ATM_TUNABLE(int, kSeedCommonBaseCost, 20);
ATM_TUNABLE(int, kSeedCommonPerLevelCost, 2);
ATM_TUNABLE(int, kSeedRareBaseCost, 60);
ATM_TUNABLE(int, kSeedRarePerLevelCost, 4);
ATM_TUNABLE(int, kSeedEpicBaseCost, 150);
ATM_TUNABLE(int, kSeedEpicPerLevelCost, 6);
ATM_TUNABLE(int, kSeedLegendaryBaseCost, 400);
ATM_TUNABLE(int, kSeedLegendaryPerLevelCost, 10);

ATM_TUNABLE(float, kSellRefundFraction, 0.50f);

ATM_TUNABLE(float, kUpgradeStrikeDamagePerRank, 0.15f);
ATM_TUNABLE(int, kUpgradeStrikeMaxRanks, 5);
ATM_TUNABLE(float, kUpgradeVitalityHpPerRank, 0.20f);
ATM_TUNABLE(int, kUpgradeVitalityMaxRanks, 5);
ATM_TUNABLE(float, kUpgradeReachRangePerRank, 0.10f);
ATM_TUNABLE(int, kUpgradeReachMaxRanks, 3);
ATM_TUNABLE(float, kUpgradeTempoAttackSpeedPerRank, 0.08f);
ATM_TUNABLE(int, kUpgradeTempoMaxRanks, 3);
ATM_TUNABLE(int, kUpgradeSignatureMaxRanks, 3);

ATM_TUNABLE(int, kUpgradeCostBase, 15);

ATM_TUNABLE(int, kRepairRestore20Hp, 20);
ATM_TUNABLE(int, kRepairRestore20Cost, 40);
ATM_TUNABLE(int, kRepairRestore50Hp, 50);
ATM_TUNABLE(int, kRepairRestore50Cost, 90);
ATM_TUNABLE(int, kRepairFullRestoreHp, 100);
ATM_TUNABLE(int, kRepairFullRestoreCost, 160);
} // namespace inter_level_shop

namespace armory {
ATM_TUNABLE_SECTION("armory");
ATM_TUNABLE(int, kCharacterGlitchShardCost, 80);
ATM_TUNABLE(int, kCharacterIronjawShardCost, 120);
ATM_TUNABLE(int, kCharacterWraithShardCost, 150);
ATM_TUNABLE(int, kCharacterCrystalisShardCost, 250);
ATM_TUNABLE(int, kCharacterVexShardCost, 300);
ATM_TUNABLE(int, kCharacterOrinShardCost, 500);
ATM_TUNABLE(int, kCharacterNullShardCost, 800);

ATM_TUNABLE(int, kEchoFoundationStartEssencePerRank, 20);
ATM_TUNABLE(int, kEchoFoundationRanks, 3);
ATM_TUNABLE(int, kEchoFoundationCostR1, 50);
ATM_TUNABLE(int, kEchoFoundationCostR2, 75);
ATM_TUNABLE(int, kEchoFoundationCostR3, 100);

ATM_TUNABLE(int, kNexusVaultStartHpPerRank, 10);
ATM_TUNABLE(int, kNexusVaultRanks, 3);
ATM_TUNABLE(int, kNexusVaultCostR1, 60);
ATM_TUNABLE(int, kNexusVaultCostR2, 90);
ATM_TUNABLE(int, kNexusVaultCostR3, 120);

ATM_TUNABLE(float, kRapidGrowthKillThresholdReductionPerRank, 0.05f);
ATM_TUNABLE(int, kRapidGrowthRanks, 3);
ATM_TUNABLE(int, kRapidGrowthCostR1, 80);
ATM_TUNABLE(int, kRapidGrowthCostR2, 110);
ATM_TUNABLE(int, kRapidGrowthCostR3, 150);

ATM_TUNABLE(float, kKineticSwarmMoveSpeedBonusPerRank, 0.05f);
ATM_TUNABLE(int, kKineticSwarmRanks, 3);
ATM_TUNABLE(int, kKineticSwarmCostR1, 40);
ATM_TUNABLE(int, kKineticSwarmCostR2, 60);
ATM_TUNABLE(int, kKineticSwarmCostR3, 80);

ATM_TUNABLE(float, kSynthesisMergeCooldownReductionSecPerRank, 1.0f);
ATM_TUNABLE(int, kSynthesisMasteryRanks, 3);
ATM_TUNABLE(int, kSynthesisMasteryCostR1, 70);
ATM_TUNABLE(int, kSynthesisMasteryCostR2, 100);
ATM_TUNABLE(int, kSynthesisMasteryCostR3, 140);

ATM_TUNABLE(float, kIronResolveHpBonusPerLevelAbove20PerRank, 0.05f);
ATM_TUNABLE(int, kIronResolveRanks, 3);
ATM_TUNABLE(int, kIronResolveCostR1, 100);
ATM_TUNABLE(int, kIronResolveCostR2, 150);
ATM_TUNABLE(int, kIronResolveCostR3, 200);

ATM_TUNABLE(float, kVoidAppetiteEssenceDropBonusPerRank, 0.08f);
ATM_TUNABLE(int, kVoidAppetiteRanks, 3);
ATM_TUNABLE(int, kVoidAppetiteCostR1, 45);
ATM_TUNABLE(int, kVoidAppetiteCostR2, 65);
ATM_TUNABLE(int, kVoidAppetiteCostR3, 90);

ATM_TUNABLE(int, kShardEyeBonusShardsPerRank, 1);
ATM_TUNABLE(int, kShardEyeRanks, 2);
ATM_TUNABLE(int, kShardEyeCostR1, 120);
ATM_TUNABLE(int, kShardEyeCostR2, 200);

ATM_TUNABLE(int, kCosmeticCharacterSkinMinCost, 100);
ATM_TUNABLE(int, kCosmeticCharacterSkinMaxCost, 200);
ATM_TUNABLE(int, kCosmeticBaseSkinCost, 80);
ATM_TUNABLE(int, kCosmeticParticleThemeCost, 60);
ATM_TUNABLE(int, kCosmeticHudThemeCost, 40);
} // namespace armory

namespace relics {
ATM_TUNABLE_SECTION("relics");

// compile-time (not in JSON): std::array bound in GameState/SaveState.
constexpr int kSlotCount = 3;

ATM_TUNABLE(int, kSlot2UnlockPlayerLevel, 3);
ATM_TUNABLE(int, kSlot3UnlockPlayerLevel, 12);

ATM_TUNABLE(float, kIronCoreHpBonus, 0.10f);
ATM_TUNABLE(float, kBloodshardDamagePerTier, 0.03f);
ATM_TUNABLE(float, kEssenceMagnetDropBonus, 0.15f);
ATM_TUNABLE(float, kMergersGiftProgressInheritance, 0.40f);
ATM_TUNABLE(float, kWarpedTimeGraceBonusSec, 3.0f);
ATM_TUNABLE(float, kPackInstinctAttackSpeedPer3SameType, 0.08f);
ATM_TUNABLE(float, kEruptionCoreBurningGroundSec, 3.0f);
ATM_TUNABLE(float, kChainStrikeShockwaveRadiusPx, 60.0f);
ATM_TUNABLE(float, kVoidLensHpBarRevealRangeMultiplier, 2.0f);
ATM_TUNABLE(int, kLivingWallHpPerWave, 20);
ATM_TUNABLE(float, kApexHungerDamageBonus, 0.20f);
ATM_TUNABLE(float, kTwinPulseAuraRadiusBonusPx, 40.0f);
ATM_TUNABLE(float, kColdBloomHealReceivedMultiplier, 0.0f);
ATM_TUNABLE(float, kResonantGrowthEvolutionRateBonus, 0.10f);
ATM_TUNABLE(int, kChaosSparkExtraOptionsPerLevelAbove30, 1);
ATM_TUNABLE(float, kRecursiveMergeSecondMergeChance, 0.10f);
ATM_TUNABLE(int, kShardHungerBonusShardsPer100Kills, 1);
ATM_TUNABLE(int, kShardHungerKillsStep, 100);
ATM_TUNABLE(float, kDeathBloomRadiusPx, 150.0f);
ATM_TUNABLE(int, kTheQuietBonusStarsCosmetic, 3);
} // namespace relics

namespace relic_unlocks {
// compile-time (not in JSON): all of these feed the constexpr relic table in
// RelicSystem.cpp.
constexpr int kEssenceMagnetShardCost = 60;
constexpr int kMergersGiftShardCost = 80;
constexpr int kWarpedTimeShardCost = 70;
constexpr int kPackInstinctShardCost = 90;
constexpr int kEruptionCoreShardCost = 100;
constexpr int kChainStrikeShardCost = 100;
constexpr int kVoidLensShardCost = 80;
constexpr int kLivingWallShardCost = 60;
constexpr int kApexHungerShardCost = 110;
constexpr int kTwinPulseShardCost = 90;
constexpr int kColdBloomShardCost = 110;
constexpr int kResonantGrowthShardCost = 130;
constexpr int kChaosSparkShardCost = 150;
constexpr int kEternalEchoShardCost = 200;
constexpr int kRecursiveMergeShardCost = 180;
constexpr int kShardHungerShardCost = 160;
constexpr int kDeathBloomShardCost = 200;
constexpr int kTheQuietShardCost = 250;
} // namespace relic_unlocks

namespace unlocks {
ATM_TUNABLE_SECTION("unlocks");
ATM_TUNABLE(int, kGlitchShopLevel, 6);
ATM_TUNABLE(int, kIronjawShopLevel, 10);
ATM_TUNABLE(int, kWraithShopLevel, 15);
ATM_TUNABLE(int, kCrystalisShopLevel, 22);
ATM_TUNABLE(int, kVexShopLevel, 30);
ATM_TUNABLE(int, kOrinUnlockLevel, 50);
ATM_TUNABLE(int, kNullUnlockLevel, 100);
} // namespace unlocks

namespace economy {
ATM_TUNABLE_SECTION("economy");
ATM_TUNABLE(int, kEssenceKillGrub, 5);
ATM_TUNABLE(int, kEssenceKillHulk, 20);
ATM_TUNABLE(int, kEssenceKillScuttle, 2);
ATM_TUNABLE(int, kEssenceKillDriftwing, 12);
ATM_TUNABLE(int, kEssenceKillDivideParent, 15);
ATM_TUNABLE(int, kEssenceKillVanguard, 18);
ATM_TUNABLE(int, kEssenceKillMender, 10);
ATM_TUNABLE(int, kEssenceKillBossBase, 150);
ATM_TUNABLE(int, kEssenceKillBossPerLevel, 8);
ATM_TUNABLE(float, kWaveClearBonusBase, 10.0f);
ATM_TUNABLE(float, kWaveClearBonusPerLevel, 1.5f);

ATM_TUNABLE(int, kEssenceLevelComplete1Star, 50);
ATM_TUNABLE(int, kEssenceLevelComplete2Star, 100);
ATM_TUNABLE(int, kEssenceLevelComplete3Star, 175);

ATM_TUNABLE(int, kInterestThresholdEssence, 100);
ATM_TUNABLE(float, kInterestRate, 0.05f);

ATM_TUNABLE(int, kShardsFirstTime3StarAnyLevel, 5);
ATM_TUNABLE(int, kShardsFirstTimeCompleteAnyLevel, 2);
ATM_TUNABLE(int, kShardsDailyChallengeAnyStars, 3);
ATM_TUNABLE(int, kShardsDailyChallenge3StarBonus, 5);
ATM_TUNABLE(int, kShardsDailyLoginStreakMinDays, 3);
ATM_TUNABLE(int, kShardsDailyLoginStreakBonusPerDay, 1);
ATM_TUNABLE(int, kShardsDailyLoginStreakBigMinDays, 7);
ATM_TUNABLE(int, kShardsDailyLoginStreakBigBonusPerDay, 3);
} // namespace economy

namespace meta {
ATM_TUNABLE_SECTION("meta");
ATM_TUNABLE(int, kXpPerLevelCompleted, 10);
ATM_TUNABLE(int, kXpPerStar, 5);
ATM_TUNABLE(int, kXpPerBossKilled, 8);
ATM_TUNABLE(int, kXpPerMerge, 2);

ATM_TUNABLE(int, kXpPerPlayerLevel, 100);

ATM_TUNABLE(int, kPlayerLevelUnlockRelicSlot2, 3);
ATM_TUNABLE(int, kPlayerLevelUnlockReroll, 7);
ATM_TUNABLE(int, kPlayerLevelUnlockRelicSlot3, 12);
ATM_TUNABLE(int, kPlayerLevelUnlockBrutalMode, 20);
ATM_TUNABLE(int, kPlayerLevelUnlockSkinSlot, 30);
ATM_TUNABLE(int, kPlayerLevelUnlockHallOfFame, 50);

ATM_TUNABLE(int, kSeasonLengthDays, 30);
ATM_TUNABLE(int, kSeasonTopRankCosmeticCount, 100);
} // namespace meta

namespace achievements {
ATM_TUNABLE_SECTION("achievements");
ATM_TUNABLE(int, kFirstBloodShards, 2);
ATM_TUNABLE(int, kEvolverShards, 3);
ATM_TUNABLE(int, kMergerShards, 3);
ATM_TUNABLE(int, kTenForwardShards, 5);
ATM_TUNABLE(int, kStarboundShards, 5);
ATM_TUNABLE(int, kBossSlayerShards, 8);
ATM_TUNABLE(int, kLevel25Shards, 10);
ATM_TUNABLE(int, kPerfectDefenseShards, 10);
ATM_TUNABLE(int, kCollectorShards, 8);
ATM_TUNABLE(int, kArmyOfOneShards, 15);
ATM_TUNABLE(int, kMergeChainShards, 10);
ATM_TUNABLE(int, kEvolutionGodShards, 20);
ATM_TUNABLE(int, kCenturyShards, 50);
ATM_TUNABLE(int, kLegendaryShards, 20);
ATM_TUNABLE(int, kNoDamageRunShards, 25);
} // namespace achievements

namespace biomes {
ATM_TUNABLE_SECTION("biomes");
ATM_TUNABLE(int, kVerdantMinLevel, 1);
ATM_TUNABLE(int, kVerdantMaxLevel, 10);
ATM_TUNABLE(int, kAshlandsMinLevel, 11);
ATM_TUNABLE(int, kAshlandsMaxLevel, 20);
ATM_TUNABLE(int, kFrostmarshMinLevel, 21);
ATM_TUNABLE(int, kFrostmarshMaxLevel, 35);
ATM_TUNABLE(int, kDeepcoreMinLevel, 36);
ATM_TUNABLE(int, kDeepcoreMaxLevel, 60);
ATM_TUNABLE(int, kVoidMinLevel, 61);

ATM_TUNABLE(int, kVoidVariantStartLevel, 61);
ATM_TUNABLE(float, kVoidVariantStatMultiplier, 1.30f);

ATM_TUNABLE(float, kObstacleDensityVariance, 0.20f);
ATM_TUNABLE(int, kSpawnEdgeCount, 4);
} // namespace biomes

namespace viral {
ATM_TUNABLE_SECTION("viral");
ATM_TUNABLE(int, kDailyLeaderboardResetHourUtc, 0);
ATM_TUNABLE(int, kWeeklyLeaderboardResetWeekday, 1);
} // namespace viral

namespace prototype {
ATM_TUNABLE_SECTION("prototype");
ATM_TUNABLE(float, kInitialBrixDeployOffsetPx, 220.0f);
ATM_TUNABLE(int, kInitialGrubCount, 5);
} // namespace prototype

} // namespace tower_swarm
