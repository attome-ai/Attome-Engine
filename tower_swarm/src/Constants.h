#pragma once

#include <cstdint>

namespace tower_swarm {

struct Rgba8 {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

// -----------------------------------------------------------------------------
// Phase 0/1 — Bootstrap + World/Camera
// -----------------------------------------------------------------------------

constexpr int kWindowWidthPx = 1280;
constexpr int kWindowHeightPx = 720;

constexpr int kWorldWidthPx = 5120;
constexpr int kWorldHeightPx = 2880;

constexpr int kTileSizePx = 64;

constexpr float kMaxFrameDtSec = 0.05f;
constexpr float kDefaultFrameDtSec = 1.0f / 60.0f;

constexpr float kCameraPanSpeedPxPerSec = 1500.0f;
constexpr float kCameraSmoothRate = 14.0f;
constexpr float kCameraDefaultZoom = 1.0f;
constexpr float kMinCameraZoomEpsilon = 0.0001f;

constexpr int kHudTopBarHeightPx = 44;
constexpr int kHudPaddingPx = 12;
constexpr int kBaseSizePx = 144;
constexpr float kBaseRadiusPx = 72.0f;
constexpr int kCreatureBaseSizePx = 48;
constexpr int kEnemyBaseSizePx = 44;
constexpr int kProjectileSizePx = 10;
constexpr int kPickupSizePx = 12;

constexpr float kProjectileHitRadiusPx = 8.0f;
constexpr float kProjectileDefaultLifetimeSec = 3.5f;
constexpr float kProjectileMinLifetimeSec = 0.05f;
constexpr float kProjectileEnemyHitPaddingFactor = 0.25f;
constexpr float kProjectileSpeedPxPerSec = 900.0f;
constexpr float kCreaturePickRadiusFactor = 0.60f;
constexpr float kPickupFloatUpSec = 2.0f;
constexpr float kPickupFloatUpPx = 30.0f;
constexpr float kPickupAttractRadiusPx = 80.0f;
constexpr float kPickupHomingSpeedPxPerSec = 520.0f;
constexpr float kPickupCollectDistancePx = 16.0f;

constexpr int kCreaturePoolCapacity = 2048;
constexpr int kEnemyPoolCapacity = 8192;
constexpr int kProjectilePoolCapacity = 10000;
constexpr int kPickupPoolCapacity = 4096;

constexpr int kBaseHpBarWidthPx = 420;
constexpr int kBaseHpBarHeightPx = 18;
constexpr int kBaseHpBarMarginBottomPx = 18;
constexpr int kBaseHpBarInsetPx = 2;
constexpr int kBaseHpBarLabelOffsetYPx = 16;
constexpr int kConfirmDialogWidthPx = 360;
constexpr int kConfirmDialogHeightPx = 160;
constexpr int kConfirmDialogButtonWidthPx = 140;
constexpr int kConfirmDialogButtonHeightPx = 36;
constexpr int kConfirmDialogButtonGapPx = 16;
constexpr int kHudSecondaryTextOffsetYPx = 6;
constexpr int kModalPanelTextInsetXPx = 16;
constexpr int kModalPanelTextInsetYPx = 20;
constexpr int kModalPanelLineStepPx = 18;
constexpr int kModalButtonTextInsetXPx = 12;
constexpr int kModalButtonTextInsetYPx = 10;
constexpr int kSelectedCreaturePanelWidthPx = 360;
constexpr int kSelectedCreaturePanelHeightPx = 86;
constexpr int kSelectedCreaturePanelBarHeightPx = 10;
constexpr int kSelectedCreaturePanelBarInsetPx = 2;

constexpr int kEnemyHpBarHeightPx = 5;
constexpr int kEnemyHpBarOffsetYPx = 6;
constexpr int kEnemyHpBarInsetPx = 1;

constexpr int kRangeIndicatorSegments = 48;

constexpr int kWaveShopCardWidthPx = 260;
constexpr int kWaveShopCardHeightPx = 140;
constexpr int kWaveShopCardGapPx = 18;
constexpr int kWaveShopBottomMarginPx = 24;
constexpr int kWaveShopCardTextInsetXPx = 14;
constexpr int kWaveShopCardTextInsetYPx = 16;

constexpr Rgba8 kClearColor = {6, 10, 24, 255};
constexpr Rgba8 kHudTopBarColor = {10, 14, 26, 220};
constexpr Rgba8 kHudBorderColor = {255, 255, 255, 18};
constexpr Rgba8 kHudTextColor = {206, 234, 255, 255};
constexpr Rgba8 kDebugGridColor = {255, 255, 255, 22};
constexpr Rgba8 kBaseColor = {26, 36, 64, 255};
constexpr Rgba8 kBaseHpBarBackColor = {8, 10, 18, 210};
constexpr Rgba8 kBaseHpBarFillColor = {64, 204, 128, 235};
constexpr Rgba8 kBaseHpBarOutlineColor = {255, 255, 255, 30};
constexpr Rgba8 kBaseHpBarMarkerColor = {255, 255, 255, 44};
constexpr Rgba8 kModalOverlayColor = {0, 0, 0, 160};
constexpr Rgba8 kModalPanelColor = {14, 18, 30, 235};
constexpr Rgba8 kModalButtonColor = {32, 44, 86, 235};
constexpr Rgba8 kModalButtonHoverColor = {48, 70, 140, 245};
constexpr Rgba8 kModalButtonTextColor = {230, 242, 255, 255};
constexpr Rgba8 kGhostValidColor = {80, 220, 160, 90};
constexpr Rgba8 kGhostInvalidColor = {220, 80, 80, 90};

constexpr Rgba8 kEnemyHpBarBackColor = {8, 10, 18, 160};
constexpr Rgba8 kEnemyHpBarFillColor = {220, 80, 80, 220};
constexpr Rgba8 kEnemyHpBarOutlineColor = {255, 255, 255, 24};

constexpr Rgba8 kRangeIndicatorColor = {120, 220, 255, 60};
constexpr Rgba8 kEvolutionPulseColor = {255, 255, 255, 90};
constexpr Rgba8 kSelectedCreatureBarBackColor = {8, 10, 18, 200};
constexpr Rgba8 kSelectedCreatureBarFillColor = {120, 220, 255, 220};
constexpr Rgba8 kMergeLinkColor = {255, 190, 90, 110};
constexpr float kMergeLinkPulseHz = 2.0f;

constexpr Rgba8 kWaveShopCardColor = {20, 26, 42, 230};
constexpr Rgba8 kWaveShopCardHoverColor = {30, 40, 70, 240};
constexpr Rgba8 kWaveShopCardBorderColor = {255, 255, 255, 28};

constexpr std::uint8_t kZIndexTiles = 0;
constexpr std::uint8_t kZIndexBase = 1;
constexpr std::uint8_t kZIndexCreatures = 2;
constexpr std::uint8_t kZIndexEnemies = 3;
constexpr std::uint8_t kZIndexProjectiles = 4;
constexpr std::uint8_t kZIndexPickups = 5;

constexpr Rgba8 kBiomeTileColorVerdantFields = {22, 64, 34, 255};
constexpr Rgba8 kBiomeTileColorAshlands = {78, 46, 30, 255};
constexpr Rgba8 kBiomeTileColorFrostmarsh = {34, 70, 92, 255};
constexpr Rgba8 kBiomeTileColorDeepcore = {64, 24, 44, 255};
constexpr Rgba8 kBiomeTileColorTheVoid = {12, 12, 18, 255};

constexpr Rgba8 kCreatureColorBrix = {170, 170, 180, 255};
constexpr Rgba8 kCreatureColorFlara = {240, 120, 40, 255};
constexpr Rgba8 kCreatureColorMossling = {60, 200, 90, 255};
constexpr Rgba8 kCreatureColorGlitch = {180, 60, 220, 255};
constexpr Rgba8 kCreatureColorIronjaw = {200, 200, 160, 255};
constexpr Rgba8 kCreatureColorWraith = {90, 90, 120, 255};
constexpr Rgba8 kCreatureColorCrystalis = {80, 220, 230, 255};
constexpr Rgba8 kCreatureColorVex = {230, 70, 150, 255};
constexpr Rgba8 kCreatureColorOrin = {240, 210, 80, 255};
constexpr Rgba8 kCreatureColorNull = {30, 30, 40, 255};

constexpr Rgba8 kEnemyColorGrub = {100, 220, 120, 255};
constexpr Rgba8 kEnemyColorHulk = {150, 100, 70, 255};
constexpr Rgba8 kEnemyColorScuttle = {80, 200, 200, 255};
constexpr Rgba8 kEnemyColorDriftwing = {120, 180, 240, 255};
constexpr Rgba8 kEnemyColorDivide = {180, 130, 220, 255};
constexpr Rgba8 kEnemyColorVanguard = {80, 120, 220, 255};
constexpr Rgba8 kEnemyColorMender = {100, 240, 180, 255};
constexpr Rgba8 kEnemyColorBoss = {220, 60, 60, 255};

constexpr Rgba8 kProjectileColor = {240, 240, 255, 255};
constexpr Rgba8 kPickupColor = {255, 224, 80, 255};

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
constexpr int kBaseHp = 100;
constexpr float kStar3Threshold = 0.70f;
constexpr float kStar2Threshold = 0.30f;

constexpr int kBaseWaveCount = 5;
constexpr float kWaveCountPerLevel = 0.5f;

constexpr float kDifficultyBase = 1.18f;

constexpr int kMapTemplateCount = 5;
constexpr int kBiomeLevelsPer = 10;
constexpr int kEliteEveryLevels = 5;

constexpr float kWaveEnemyCountBase = 3.0f;
constexpr float kWaveEnemyCountLinear = 1.8f;
constexpr float kWaveEnemyCountQuadratic = 0.05f;
constexpr float kWaveEnemyCountWaveFactor = 0.15f;

constexpr float kWaveEnemyHpWaveFactor = 0.10f;

constexpr float kWaveEnemySpeedExponent = 0.4f;
constexpr float kWaveEnemySpeedWaveFactor = 0.04f;
constexpr float kWaveEnemySpeedMax = 3.0f;

// Wave spawner pacing.
constexpr float kInterSpawnDelaySec = 0.08f;

// Banners (Production TODO Â§7.1).
constexpr float kWaveStartBannerDurationSec = 1.5f;
constexpr float kWaveClearBannerDurationSec = 2.0f;
constexpr float kBossWaveBannerDurationSec = 3.0f;
constexpr float kLevelStartBannerDurationSec = 2.0f;
constexpr float kLevelClearBannerDurationSec = 2.0f;
constexpr float kLevelFailedBannerDurationSec = 2.0f;

constexpr int kMilestoneTutorialLevel = 1;
constexpr int kMilestoneFirstEliteLevel = 5;
constexpr int kMilestoneBiome2Level = 10;
constexpr int kMilestoneChargerShopLevel = 15;
constexpr int kMilestoneMidBossLevel = 25;
constexpr int kMilestoneVoidPreviewLevel = 50;
constexpr int kMilestoneHallOfFameLevel = 100;
constexpr int kMilestoneMasteryTagLevel = 200;
} // namespace level

namespace evolution {
constexpr int kKillsTier1To2 = 10;
constexpr int kKillsTier2To3 = 30;
constexpr int kKillsTier3To4 = 80;
constexpr int kKillsTier4To5 = 200;
constexpr float kKillsInfiniteBase = 10.0f;
constexpr float kKillsInfiniteGrowth = 2.5f;

constexpr float kHpExponent = 1.4f;
constexpr float kDamageExponent = 1.3f;
constexpr float kRangeExponent = 0.5f;
constexpr float kRangeCapPx = 600.0f;
constexpr float kAttackRateExponent = 0.4f;
constexpr float kAttackRateCapPerSec = 8.0f;
  constexpr float kMoveSpeedExponent = 0.2f;

  constexpr int kVisualBandCount = 7;

  constexpr float kTierSize1To3 = 1.0f;
  constexpr float kTierSize4To6 = 1.3f;
  constexpr float kTierSize7To9 = 1.6f;
  constexpr float kTierSize10To12 = 2.0f;
constexpr float kTierSize13To15 = 2.4f;
constexpr float kTierSize16To19 = 2.8f;
constexpr float kTierSize20Plus = 3.0f;
constexpr float kTierSizeCap = 3.0f;

constexpr float kEvolutionPulseScale = 1.5f;
constexpr float kEvolutionAnimSec = 0.8f;
constexpr float kEvolutionFloatingTextSec = 1.25f;
constexpr float kEvolutionFloatingTextRisePxPerSec = 40.0f;
constexpr float kScreenEdgeGlowSec = 1.0f;
} // namespace evolution

namespace merge {
constexpr float kCooldownSec = 3.0f;
constexpr float kAutoMergeIdleSec = 6.0f;
constexpr float kAnimationSec = 0.8f;
constexpr int kEssenceBonus = 10;
constexpr float kKillInheritanceDivisor = 2.0f;
} // namespace merge

namespace movement_ai {
constexpr float kRecalcIntervalSec = 3.0f;

constexpr float kThreatRadiusNearPx = 200.0f;
constexpr float kThreatRadiusMidPx = 400.0f;
constexpr float kThreatRadiusFarPx = 600.0f;
constexpr float kThreatWeightNear = 3.0f;
constexpr float kThreatWeightMid = 2.0f;
constexpr float kThreatWeightFar = 1.0f;

constexpr float kSupportRepelRadiusPx = 96.0f;

constexpr float kDesiredMoveThresholdPx = 64.0f;
constexpr float kWaypointInterpSec = 1.5f;

constexpr float kPlayerDragStartThresholdPx = 8.0f;
constexpr float kPlayerDragStunSec = 0.5f;
} // namespace movement_ai

namespace characters {
constexpr int kEvolutionStage1MaxTier = 3;
constexpr int kEvolutionStage2MaxTier = 6;
constexpr int kEvolutionStage3MaxTier = 9;
constexpr int kEvolutionStage4MinTier = 10;

namespace base_stats {
constexpr float kBrixBaseHp = 70.0f;
constexpr float kBrixBaseDamage = 12.0f;
constexpr float kBrixBaseRangePx = 220.0f;
constexpr float kBrixBaseAttackRatePerSec = 1.5f;
constexpr float kBrixBaseMoveSpeedPxPerSec = 120.0f;

constexpr float kFlaraBaseHp = 55.0f;
constexpr float kFlaraBaseDamage = 8.0f;
constexpr float kFlaraBaseRangePx = 180.0f;
constexpr float kFlaraBaseAttackRatePerSec = 0.8f;
constexpr float kFlaraBaseMoveSpeedPxPerSec = 125.0f;
constexpr float kFlaraSplashRadiusPx = 80.0f;

constexpr float kMosslingBaseHp = 85.0f;
constexpr float kMosslingBaseDamage = 4.0f;
constexpr float kMosslingBaseRangePx = 160.0f;
constexpr float kMosslingBaseAttackRatePerSec = 1.0f;
constexpr float kMosslingBaseMoveSpeedPxPerSec = 120.0f;

constexpr float kGlitchBaseHp = 65.0f;
constexpr float kGlitchBaseDamage = 6.0f;
constexpr float kGlitchBaseRangePx = 200.0f;
constexpr float kGlitchBaseAttackRatePerSec = 0.7f;
constexpr float kGlitchBaseMoveSpeedPxPerSec = 120.0f;
constexpr float kGlitchSlowFieldRadiusPx = 60.0f;

constexpr float kIronjawBaseHp = 110.0f;
constexpr float kIronjawBaseDamage = 10.0f;
constexpr float kIronjawBaseRangePx = 80.0f;
constexpr float kIronjawBaseAttackRatePerSec = 1.2f;
constexpr float kIronjawBaseMoveSpeedPxPerSec = 160.0f;
constexpr float kIronjawChargeRangePx = 300.0f;
constexpr float kIronjawChargeDamage = 30.0f;
constexpr float kIronjawChargeKnockbackPx = 80.0f;

constexpr float kWraithBaseHp = 45.0f;
constexpr float kWraithBaseDamage = 40.0f;
constexpr float kWraithBaseRangePx = 500.0f;
constexpr float kWraithBaseAttackRatePerSec = 0.3f;
constexpr float kWraithBaseMoveSpeedPxPerSec = 120.0f;

constexpr float kCrystalisBaseHp = 80.0f;
constexpr float kCrystalisBaseDamage = 15.0f;
constexpr float kCrystalisBaseRangePx = 280.0f;
constexpr float kCrystalisBaseAttackRatePerSec = 0.9f;
constexpr float kCrystalisBaseMoveSpeedPxPerSec = 120.0f;
constexpr float kCrystalisAuraRangeBoost = 0.15f;

constexpr float kVexBaseHp = 75.0f;
constexpr float kVexBaseDamage = 10.0f;
constexpr float kVexBaseRangePx = 220.0f;
constexpr float kVexBaseAttackRatePerSec = 0.9f;
constexpr float kVexBaseMoveSpeedPxPerSec = 130.0f;

constexpr float kOrinBaseHp = 160.0f;
constexpr float kOrinBaseDamage = 18.0f;
constexpr float kOrinBaseRangePx = 240.0f;
constexpr float kOrinBaseAttackRatePerSec = 0.6f;
constexpr float kOrinBaseMoveSpeedPxPerSec = 110.0f;

constexpr float kNullBaseHp = 140.0f;
constexpr float kNullBaseDamage = 14.0f;
constexpr float kNullBaseRangePx = 200.0f;
constexpr float kNullBaseAttackRatePerSec = 0.7f;
constexpr float kNullBaseMoveSpeedPxPerSec = 115.0f;
constexpr float kNullDrainRadiusPx = 180.0f;
} // namespace base_stats

namespace brix {
constexpr int kPierceStage2 = 1;
constexpr int kPierceStage3 = 3;
constexpr float kStage3RangeBonus = 0.20f;
constexpr float kStage4SplashRadiusPx = 60.0f;
constexpr float kSignatureCooldownSec = 15.0f;
constexpr float kSignatureLineLengthPx = 150.0f;
} // namespace brix

namespace flara {
constexpr float kBurningGroundStage2Sec = 2.0f;
constexpr float kBurningGroundStage3Sec = 4.0f;
constexpr int kStage4SimultaneousTargets = 3;
constexpr float kSignatureCooldownSec = 20.0f;
constexpr float kSignatureRadiusPx = 300.0f;
constexpr float kSignatureDamageMultiplier = 5.0f;
} // namespace flara

namespace mossling {
constexpr float kAuraRadiusStage1Px = 96.0f;
constexpr float kAuraAttackSpeedStage1 = 0.05f;
constexpr float kAuraAttackSpeedStage2 = 0.10f;
constexpr float kAuraDamageStage2 = 0.08f;
constexpr float kAuraHealStage3HpPerSec = 2.0f;
constexpr float kAuraRadiusStage4Px = 200.0f;
constexpr float kSignatureCooldownSec = 25.0f;
} // namespace mossling

namespace glitch {
constexpr float kSlowFieldSpeedMultiplier = 0.50f;
constexpr float kSlowFieldDurationSec = 3.0f;
constexpr float kOrbDetonateAfterSec = 4.0f;
constexpr float kSignatureCooldownSec = 18.0f;
constexpr float kSignatureFreezeRadiusPx = 250.0f;
constexpr float kSignatureFreezeDurationSec = 2.5f;
} // namespace glitch

namespace ironjaw {
constexpr float kSignatureCooldownSec = 22.0f;
constexpr float kSignatureFrenzyDurationSec = 4.0f;
constexpr float kSignatureAttackSpeedMultiplier = 3.0f;
} // namespace ironjaw

namespace wraith {
constexpr float kArmorIgnoreFraction = 0.30f;
constexpr float kExecuteBelowHpFraction = 0.15f;
constexpr float kSignatureCooldownSec = 30.0f;
constexpr float kSignatureMarkDurationSec = 4.0f;
} // namespace wraith

namespace crystalis {
constexpr int kStage2RefractTargets = 2;
constexpr int kStage3RefractTargets = 4;
constexpr float kSignatureCooldownSec = 20.0f;
} // namespace crystalis

namespace vex {
constexpr float kRandomAbilityIntervalSec = 5.0f;
constexpr float kStage3AbilityStrengthMultiplier = 2.0f;
constexpr float kSignatureCooldownSec = 25.0f;
} // namespace vex

namespace orin {
constexpr float kPassiveIgnoreDamageChanceStage1 = 0.05f;
constexpr float kPassiveIgnoreDamageChanceStage2 = 0.15f;
constexpr float kPassiveBaseShieldStage3 = 0.25f;
constexpr float kSignatureCooldownSec = 60.0f;
constexpr float kSignatureFreezeDurationSec = 5.0f;
} // namespace orin

namespace null_seed {
constexpr float kDrainDamageStage1 = 0.10f;
constexpr float kDrainDamageStage2 = 0.25f;
constexpr float kDrainSpeedStage2 = 0.15f;
constexpr float kDrainDamageStage3 = 1.00f;
constexpr float kSignatureCooldownSec = 45.0f;
} // namespace null_seed
} // namespace characters

namespace enemies {
constexpr int kIntroLevelGrub = 1;
constexpr int kIntroLevelHulk = 2;
constexpr int kIntroLevelScuttle = 4;
constexpr int kIntroLevelDriftwing = 7;
constexpr int kIntroLevelDivide = 11;
constexpr int kIntroLevelVanguard = 16;
constexpr int kIntroLevelMender = 22;

constexpr float kHulkFrontDamageTakenMultiplier = 0.50f;

constexpr int kScuttlePackMin = 15;
constexpr int kScuttlePackMax = 30;

constexpr int kDivideChildrenCount = 2;
constexpr float kDivideChildHpFactor = 0.40f;

constexpr float kVanguardFrontResist = 0.80f;

constexpr float kMenderHealHpPerSec = 8.0f;
constexpr float kMenderHealRadiusPx = 120.0f;

constexpr float kBossHpMultiplier = 50.0f;
constexpr float kBossPhase2Threshold = 0.66f;
constexpr float kBossPhase3Threshold = 0.33f;
constexpr int kBossPhase2SpawnGrubs = 20;
constexpr int kBossPhase3SpawnHulks = 10;
constexpr float kBossPhase2SpeedBonus = 0.30f;
constexpr float kBossBaseSpeedMultiplier = 0.65f;
constexpr int kBossDeathEssenceBase = 150;
constexpr int kBossDeathEssencePerLevel = 8;
constexpr float kBossBaseDamageToBase = 15.0f;
constexpr float kBossStompRadiusPx = 120.0f;
constexpr float kBossStompIntervalSec = 2.25f;
constexpr float kBossStompInitialDelaySec = 0.75f;
constexpr float kBossStompDamageToCreatureMultiplier = 1.25f;

constexpr float kEliteHpMultiplier = 1.50f;
constexpr float kEliteSpeedMultiplier = 1.20f;

constexpr float kScalingSpeedExponent = 0.4f;
constexpr float kScalingSpeedCap = 3.0f;
constexpr float kScalingDamagePerLevel = 0.05f;
constexpr float kScalingRewardPerLevel = 0.08f;

constexpr float kNewEnemyIntroPauseSec = 2.0f;

constexpr float kFrontHitDotThreshold = 0.25f;

constexpr float kDivideChildSpawnOffsetXPx = 12.0f;
constexpr float kDivideChildSpawnOffsetYPx = 10.0f;

constexpr float kGrubBaseHp = 30.0f;
constexpr float kGrubBaseSpeedPxPerSec = 90.0f;
constexpr int kGrubBaseRewardEssence = 5;
constexpr float kGrubBaseDamageToBase = 1.0f;

constexpr float kHulkBaseHp = 250.0f;
constexpr float kHulkBaseSpeedPxPerSec = 35.0f;
constexpr int kHulkBaseRewardEssence = 20;
constexpr float kHulkBaseDamageToBase = 4.0f;

constexpr float kScuttleBaseHp = 12.0f;
constexpr float kScuttleBaseSpeedPxPerSec = 110.0f;
constexpr int kScuttleBaseRewardEssence = 2;
constexpr float kScuttleBaseDamageToBase = 1.0f;

constexpr float kDriftwingBaseHp = 60.0f;
constexpr float kDriftwingBaseSpeedPxPerSec = 70.0f;
constexpr int kDriftwingBaseRewardEssence = 12;
constexpr float kDriftwingBaseDamageToBase = 2.0f;

constexpr float kDivideBaseHp = 80.0f;
constexpr float kDivideBaseSpeedPxPerSec = 55.0f;
constexpr int kDivideBaseRewardEssence = 15;
constexpr float kDivideBaseDamageToBase = 2.0f;

constexpr float kVanguardBaseHp = 150.0f;
constexpr float kVanguardBaseSpeedPxPerSec = 50.0f;
constexpr int kVanguardBaseRewardEssence = 18;
constexpr float kVanguardBaseDamageToBase = 3.0f;

constexpr float kMenderBaseHp = 40.0f;
constexpr float kMenderBaseSpeedPxPerSec = 40.0f;
constexpr int kMenderBaseRewardEssence = 10;
constexpr float kMenderBaseDamageToBase = 2.0f;
} // namespace enemies

namespace wave_shop {
constexpr int kCardPoolSize = 12;
constexpr int kCardsDrawnPerWaveClear = 3;
constexpr float kSelectionTimerSec = 5.0f;

constexpr float kSurgeAttackSpeedBonus = 0.25f;
constexpr int kSurgeDurationWaves = 4;

constexpr int kFortifyBaseHpBonus = 15;

constexpr int kFrenziedBloodEssencePerKill = 1;
constexpr int kFrenziedBloodDurationWaves = 3;

constexpr float kSlowTideSpeedMultiplier = 0.65f;
constexpr int kSlowTideDurationWaves = 1;

constexpr int kForesightDurationWaves = 1;

constexpr float kMendHealFraction = 0.50f;

constexpr int kWildSeedTier = 2;

constexpr float kEchoStrikeDamageRepeatFraction = 0.20f;
constexpr float kEchoStrikeRepeatDelaySec = 0.30f;
constexpr int kEchoStrikeDurationWaves = 3;

constexpr float kEssenceCacheFraction = 0.30f;

constexpr float kIronSkinDamageTakenMultiplier = 0.80f;
constexpr int kIronSkinDurationWaves = 2;

constexpr float kApexHunterDamageBonus = 0.50f;
constexpr int kApexHunterDurationWaves = 1;

constexpr int kVoidPulseKillInterval = 10;
constexpr float kVoidPulseRadiusPx = 80.0f;
constexpr float kVoidPulseDamageFractionOfKilledEnemyMaxHp = 0.50f;
constexpr int kVoidPulseDurationWaves = 1;
} // namespace wave_shop

namespace inter_level_shop {
constexpr int kBazaarOfferCount = 4;
constexpr int kRerollCostEssence = 15;

constexpr int kSeedCommonBaseCost = 20;
constexpr int kSeedCommonPerLevelCost = 2;
constexpr int kSeedRareBaseCost = 60;
constexpr int kSeedRarePerLevelCost = 4;
constexpr int kSeedEpicBaseCost = 150;
constexpr int kSeedEpicPerLevelCost = 6;
constexpr int kSeedLegendaryBaseCost = 400;
constexpr int kSeedLegendaryPerLevelCost = 10;

constexpr float kSellRefundFraction = 0.50f;

constexpr float kUpgradeStrikeDamagePerRank = 0.15f;
constexpr int kUpgradeStrikeMaxRanks = 5;
constexpr float kUpgradeVitalityHpPerRank = 0.20f;
constexpr int kUpgradeVitalityMaxRanks = 5;
constexpr float kUpgradeReachRangePerRank = 0.10f;
constexpr int kUpgradeReachMaxRanks = 3;
constexpr float kUpgradeTempoAttackSpeedPerRank = 0.08f;
constexpr int kUpgradeTempoMaxRanks = 3;
constexpr int kUpgradeSignatureMaxRanks = 3;

constexpr int kUpgradeCostBase = 15;

constexpr int kRepairRestore20Hp = 20;
constexpr int kRepairRestore20Cost = 40;
constexpr int kRepairRestore50Hp = 50;
constexpr int kRepairRestore50Cost = 90;
constexpr int kRepairFullRestoreHp = 100;
constexpr int kRepairFullRestoreCost = 160;
} // namespace inter_level_shop

namespace armory {
constexpr int kCharacterGlitchShardCost = 80;
constexpr int kCharacterIronjawShardCost = 120;
constexpr int kCharacterWraithShardCost = 150;
constexpr int kCharacterCrystalisShardCost = 250;
constexpr int kCharacterVexShardCost = 300;
constexpr int kCharacterOrinShardCost = 500;
constexpr int kCharacterNullShardCost = 800;

constexpr int kEchoFoundationStartEssencePerRank = 20;
constexpr int kEchoFoundationRanks = 3;
constexpr int kEchoFoundationCostR1 = 50;
constexpr int kEchoFoundationCostR2 = 75;
constexpr int kEchoFoundationCostR3 = 100;

constexpr int kNexusVaultStartHpPerRank = 10;
constexpr int kNexusVaultRanks = 3;
constexpr int kNexusVaultCostR1 = 60;
constexpr int kNexusVaultCostR2 = 90;
constexpr int kNexusVaultCostR3 = 120;

constexpr float kRapidGrowthKillThresholdReductionPerRank = 0.05f;
constexpr int kRapidGrowthRanks = 3;
constexpr int kRapidGrowthCostR1 = 80;
constexpr int kRapidGrowthCostR2 = 110;
constexpr int kRapidGrowthCostR3 = 150;

constexpr float kKineticSwarmMoveSpeedBonusPerRank = 0.05f;
constexpr int kKineticSwarmRanks = 3;
constexpr int kKineticSwarmCostR1 = 40;
constexpr int kKineticSwarmCostR2 = 60;
constexpr int kKineticSwarmCostR3 = 80;

constexpr float kSynthesisMergeCooldownReductionSecPerRank = 1.0f;
constexpr int kSynthesisMasteryRanks = 3;
constexpr int kSynthesisMasteryCostR1 = 70;
constexpr int kSynthesisMasteryCostR2 = 100;
constexpr int kSynthesisMasteryCostR3 = 140;

constexpr float kIronResolveHpBonusPerLevelAbove20PerRank = 0.05f;
constexpr int kIronResolveRanks = 3;
constexpr int kIronResolveCostR1 = 100;
constexpr int kIronResolveCostR2 = 150;
constexpr int kIronResolveCostR3 = 200;

constexpr float kVoidAppetiteEssenceDropBonusPerRank = 0.08f;
constexpr int kVoidAppetiteRanks = 3;
constexpr int kVoidAppetiteCostR1 = 45;
constexpr int kVoidAppetiteCostR2 = 65;
constexpr int kVoidAppetiteCostR3 = 90;

constexpr int kShardEyeBonusShardsPerRank = 1;
constexpr int kShardEyeRanks = 2;
constexpr int kShardEyeCostR1 = 120;
constexpr int kShardEyeCostR2 = 200;

constexpr int kCosmeticCharacterSkinMinCost = 100;
constexpr int kCosmeticCharacterSkinMaxCost = 200;
constexpr int kCosmeticBaseSkinCost = 80;
constexpr int kCosmeticParticleThemeCost = 60;
constexpr int kCosmeticHudThemeCost = 40;
} // namespace armory

namespace relics {
constexpr int kSlotCount = 3;
constexpr int kSlot2UnlockPlayerLevel = 3;
constexpr int kSlot3UnlockPlayerLevel = 12;

constexpr float kIronCoreHpBonus = 0.10f;
constexpr float kBloodshardDamagePerTier = 0.03f;
constexpr float kEssenceMagnetDropBonus = 0.15f;
constexpr float kMergersGiftProgressInheritance = 0.40f;
constexpr float kWarpedTimeGraceBonusSec = 3.0f;
constexpr float kPackInstinctAttackSpeedPer3SameType = 0.08f;
constexpr float kEruptionCoreBurningGroundSec = 3.0f;
constexpr float kChainStrikeShockwaveRadiusPx = 60.0f;
constexpr float kVoidLensHpBarRevealRangeMultiplier = 2.0f;
constexpr int kLivingWallHpPerWave = 20;
constexpr float kApexHungerDamageBonus = 0.20f;
constexpr float kTwinPulseAuraRadiusBonusPx = 40.0f;
constexpr float kColdBloomHealReceivedMultiplier = 0.0f;
constexpr float kResonantGrowthEvolutionRateBonus = 0.10f;
constexpr int kChaosSparkExtraOptionsPerLevelAbove30 = 1;
constexpr float kRecursiveMergeSecondMergeChance = 0.10f;
constexpr int kShardHungerBonusShardsPer100Kills = 1;
constexpr int kShardHungerKillsStep = 100;
constexpr float kDeathBloomRadiusPx = 150.0f;
constexpr int kTheQuietBonusStarsCosmetic = 3;
} // namespace relics

namespace relic_unlocks {
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
constexpr int kGlitchShopLevel = 6;
constexpr int kIronjawShopLevel = 10;
constexpr int kWraithShopLevel = 15;
constexpr int kCrystalisShopLevel = 22;
constexpr int kVexShopLevel = 30;
constexpr int kOrinUnlockLevel = 50;
constexpr int kNullUnlockLevel = 100;
} // namespace unlocks

namespace economy {
constexpr int kEssenceKillGrub = 5;
constexpr int kEssenceKillHulk = 20;
constexpr int kEssenceKillScuttle = 2;
constexpr int kEssenceKillDriftwing = 12;
constexpr int kEssenceKillDivideParent = 15;
constexpr int kEssenceKillVanguard = 18;
constexpr int kEssenceKillMender = 10;
constexpr int kEssenceKillBossBase = 150;
constexpr int kEssenceKillBossPerLevel = 8;
constexpr float kWaveClearBonusBase = 10.0f;
constexpr float kWaveClearBonusPerLevel = 1.5f;

constexpr int kEssenceLevelComplete1Star = 50;
constexpr int kEssenceLevelComplete2Star = 100;
constexpr int kEssenceLevelComplete3Star = 175;

constexpr int kInterestThresholdEssence = 100;
constexpr float kInterestRate = 0.05f;

constexpr int kShardsFirstTime3StarAnyLevel = 5;
constexpr int kShardsFirstTimeCompleteAnyLevel = 2;
constexpr int kShardsDailyChallengeAnyStars = 3;
constexpr int kShardsDailyChallenge3StarBonus = 5;
constexpr int kShardsDailyLoginStreakMinDays = 3;
constexpr int kShardsDailyLoginStreakBonusPerDay = 1;
constexpr int kShardsDailyLoginStreakBigMinDays = 7;
constexpr int kShardsDailyLoginStreakBigBonusPerDay = 3;
} // namespace economy

namespace meta {
constexpr int kXpPerLevelCompleted = 10;
constexpr int kXpPerStar = 5;
constexpr int kXpPerBossKilled = 8;
constexpr int kXpPerMerge = 2;

constexpr int kXpPerPlayerLevel = 100;

constexpr int kPlayerLevelUnlockRelicSlot2 = 3;
constexpr int kPlayerLevelUnlockReroll = 7;
constexpr int kPlayerLevelUnlockRelicSlot3 = 12;
constexpr int kPlayerLevelUnlockBrutalMode = 20;
constexpr int kPlayerLevelUnlockSkinSlot = 30;
constexpr int kPlayerLevelUnlockHallOfFame = 50;

constexpr int kSeasonLengthDays = 30;
constexpr int kSeasonTopRankCosmeticCount = 100;
} // namespace meta

namespace achievements {
constexpr int kFirstBloodShards = 2;
constexpr int kEvolverShards = 3;
constexpr int kMergerShards = 3;
constexpr int kTenForwardShards = 5;
constexpr int kStarboundShards = 5;
constexpr int kBossSlayerShards = 8;
constexpr int kLevel25Shards = 10;
constexpr int kPerfectDefenseShards = 10;
constexpr int kCollectorShards = 8;
constexpr int kArmyOfOneShards = 15;
constexpr int kMergeChainShards = 10;
constexpr int kEvolutionGodShards = 20;
constexpr int kCenturyShards = 50;
constexpr int kLegendaryShards = 20;
constexpr int kNoDamageRunShards = 25;
} // namespace achievements

namespace biomes {
constexpr int kVerdantMinLevel = 1;
constexpr int kVerdantMaxLevel = 10;
constexpr int kAshlandsMinLevel = 11;
constexpr int kAshlandsMaxLevel = 20;
constexpr int kFrostmarshMinLevel = 21;
constexpr int kFrostmarshMaxLevel = 35;
constexpr int kDeepcoreMinLevel = 36;
constexpr int kDeepcoreMaxLevel = 60;
constexpr int kVoidMinLevel = 61;

constexpr int kVoidVariantStartLevel = 61;
constexpr float kVoidVariantStatMultiplier = 1.30f;

constexpr float kObstacleDensityVariance = 0.20f;
constexpr int kSpawnEdgeCount = 4;
} // namespace biomes

namespace viral {
constexpr int kDailyLeaderboardResetHourUtc = 0;
constexpr int kWeeklyLeaderboardResetWeekday = 1;
} // namespace viral

namespace prototype {
constexpr float kInitialBrixDeployOffsetPx = 220.0f;
constexpr int kInitialGrubCount = 5;
} // namespace prototype

} // namespace tower_swarm
