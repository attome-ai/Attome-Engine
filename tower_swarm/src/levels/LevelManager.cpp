#include "levels/LevelManager.h"

#include "Constants.h"
#include "entities/BaseEntity.h"
#include "entities/EnemyContainer.h"
#include "levels/GameState.h"
#include "levels/LevelScaler.h"

#include "ATMEngine.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <string_view>

namespace tower_swarm {
namespace {

std::uint32_t level_seed(std::int32_t level_number) {
  const std::uint32_t n =
      static_cast<std::uint32_t>(sanitize_level_number(level_number));
  return 0xA3C59AC3u ^ (n * 2654435761u);
}

EnemyType intro_enemy_for_wave(std::int32_t level_number, std::int32_t wave_index,
                               std::int32_t wave_count) {
  const std::int32_t lvl = sanitize_level_number(level_number);
  const std::int32_t w = std::max<std::int32_t>(0, wave_index);
  const std::int32_t wc = std::max<std::int32_t>(0, wave_count);

  if (is_boss_wave(w, wc)) {
    return EnemyType::Count;
  }

  if (lvl == enemies::kIntroLevelHulk && w == 0) {
    return EnemyType::Hulk;
  }

  if (lvl == enemies::kIntroLevelScuttle) {
    const std::int32_t last_normal_wave = std::max<std::int32_t>(0, wc - 2);
    for (std::int32_t wi = 0; wi <= last_normal_wave; ++wi) {
      if (wave_enemy_count(lvl, wi) >= enemies::kScuttlePackMin) {
        return (w == wi) ? EnemyType::Scuttle : EnemyType::Count;
      }
    }
  }

  if (lvl == enemies::kIntroLevelDriftwing && w == 0) {
    return EnemyType::Driftwing;
  }
  if (lvl == enemies::kIntroLevelDivide && w == 0) {
    return EnemyType::Divide;
  }
  if (lvl == enemies::kIntroLevelVanguard && w == 0) {
    return EnemyType::Vanguard;
  }
  if (lvl == enemies::kIntroLevelMender && w == 0) {
    return EnemyType::Mender;
  }

  return EnemyType::Count;
}

} // namespace

void LevelManager::bind(Engine *engine, BaseEntity *base, EntityHandle base_id,
                        EnemyContainer *enemies, GameState *game_state) {
  engine_ = engine;
  base_ = base;
  base_id_ = base_id;
  enemies_ = enemies;
  game_state_ = game_state;
  spawner_.bind(engine_, enemies_);
}

const char *LevelManager::bannerText() const {
  if (banner_remaining_sec_ <= 0.0f || banner_[0] == '\0') {
    return nullptr;
  }
  return banner_.data();
}

bool LevelManager::isBossWave() const {
  return is_boss_wave(wave_index_, def_.wave_count);
}

void LevelManager::setBanner(const char *fmt, ...) {
  if (!fmt) {
    banner_[0] = '\0';
    banner_remaining_sec_ = 0.0f;
    return;
  }

  va_list args;
  va_start(args, fmt);
  std::vsnprintf(banner_.data(), banner_.size(), fmt, args);
  va_end(args);
}

std::int32_t LevelManager::countVisibleEnemies() const {
  if (!enemies_) {
    return 0;
  }
  std::int32_t alive = 0;
  for (std::uint32_t slot = 0;
       slot < static_cast<std::uint32_t>(enemies_->count); ++slot) {
    if ((enemies_->flags[slot] & static_cast<std::uint8_t>(EntityFlag::VISIBLE)) ==
        0) {
      continue;
    }
    alive++;
  }
  return alive;
}

void LevelManager::beginLevel(std::int32_t level_number) {
  def_ = generate_level_definition(level_number);
  wave_index_ = 0;
  grace_remaining_sec_ = between_wave_grace_timer_sec(def_.level_number);
  if (game_state_ && game_state_->isRelicEquipped(RelicId::WarpedTime)) {
    grace_remaining_sec_ =
        std::max(0.0f, grace_remaining_sec_ + relics::kWarpedTimeGraceBonusSec);
  }
  state_ = LevelManagerState::WaveClear;
  last_level_stars_ = 0;

  if (spawner_.interSpawnDelaySec() <= 0.0f) {
    spawner_.setInterSpawnDelaySec(level::kInterSpawnDelaySec);
  }
  spawner_.beginLevel(def_, level_seed(def_.level_number));

  if (base_ && base_id_ != INVALID_ID) {
    int max_hp = level::kBaseHp;
    if (game_state_) {
      max_hp += std::max(0, game_state_->nexusVaultStartHpBonus());
    }
    base_->setMaxHp(base_id_, std::max(1, max_hp), false);
  }

  if (game_state_) {
    game_state_->level_number = def_.level_number;
    game_state_->wave_index = wave_index_;
    game_state_->wave_count = def_.wave_count;
    game_state_->is_elite = def_.is_elite;
    game_state_->level_time_sec = 0.0f;
    game_state_->enemies_killed_this_level = 0;
    game_state_->essence_earned_this_level = 0;
    game_state_->merges_this_level = 0;
    game_state_->evolutions_this_level = 0;
  }

  banner_remaining_sec_ = level::kLevelStartBannerDurationSec;
  if (def_.is_elite) {
    setBanner("ELITE LEVEL %d", def_.level_number);
  } else {
    setBanner("LEVEL %d", def_.level_number);
  }
}

std::int32_t LevelManager::computeStars() const {
  if (!base_ || base_id_ == INVALID_ID) {
    return 0;
  }
  const int hp = base_->getHp(base_id_);
  const int hp_max = base_->getHpMax(base_id_);
  if (hp_max <= 0) {
    return 0;
  }
  const float frac = static_cast<float>(hp) / static_cast<float>(hp_max);
  if (frac > level::kStar3Threshold) {
    return 3;
  }
  if (frac >= level::kStar2Threshold) {
    return 2;
  }
  return hp > 0 ? 1 : 0;
}

void LevelManager::enterWaveClear(std::int32_t cleared_wave) {
  state_ = LevelManagerState::WaveClear;
  grace_remaining_sec_ = between_wave_grace_timer_sec(def_.level_number);
  if (game_state_ && game_state_->isRelicEquipped(RelicId::WarpedTime)) {
    grace_remaining_sec_ =
        std::max(0.0f, grace_remaining_sec_ + relics::kWarpedTimeGraceBonusSec);
  }
  banner_remaining_sec_ = level::kWaveClearBannerDurationSec;
  setBanner("WAVE %d CLEAR", cleared_wave + 1);

  if (game_state_) {
    const float raw =
        economy::kWaveClearBonusBase +
        static_cast<float>(def_.level_number) * economy::kWaveClearBonusPerLevel;
    const int bonus = std::max(0, static_cast<int>(std::lround(raw)));
    if (bonus > 0) {
      game_state_->essence += bonus;
      game_state_->essence_earned_this_level += bonus;
    }
    game_state_->onWaveEnded();
  }
}

void LevelManager::enterLevelClear() {
  state_ = LevelManagerState::LevelClear;
  last_level_stars_ = computeStars();
  banner_remaining_sec_ = level::kLevelClearBannerDurationSec;
  setBanner("LEVEL %d COMPLETE", def_.level_number);

  if (game_state_) {
    const bool quiet =
        game_state_->isRelicEquipped(RelicId::TheQuiet) &&
        !game_state_->any_enemy_reached_base_this_level;
    game_state_->the_quiet_bonus_stars_cosmetic =
        quiet ? relics::kTheQuietBonusStarsCosmetic : 0;
  }

  if (game_state_) {
    game_state_->clearWaveBuffs();
  }

  if (game_state_) {
    const std::size_t idx = static_cast<std::size_t>(
        std::max<std::int32_t>(0, def_.level_number - 1));
    std::uint8_t prev_best = 0;
    if (idx < game_state_->stars_per_level.size()) {
      prev_best = game_state_->stars_per_level[idx];
    }

    game_state_->max_level_reached =
        std::max(game_state_->max_level_reached, def_.level_number + 1);

    if (game_state_->stars_per_level.size() <= idx) {
      game_state_->stars_per_level.resize(idx + 1, 0);
    }
    game_state_->stars_per_level[idx] = static_cast<std::uint8_t>(
        std::max<std::uint8_t>(game_state_->stars_per_level[idx],
                               static_cast<std::uint8_t>(last_level_stars_)));

    const int stars = std::clamp(last_level_stars_, 0, 3);
    int level_bonus = 0;
    if (stars >= 3) {
      level_bonus = economy::kEssenceLevelComplete3Star;
    } else if (stars == 2) {
      level_bonus = economy::kEssenceLevelComplete2Star;
    } else if (stars == 1) {
      level_bonus = economy::kEssenceLevelComplete1Star;
    }
    if (level_bonus > 0) {
      game_state_->essence += level_bonus;
      game_state_->essence_earned_this_level += level_bonus;
    }

    if (game_state_->essence >= economy::kInterestThresholdEssence) {
      const int interest = std::max(
          0,
          static_cast<int>(std::floor(static_cast<float>(game_state_->essence) *
                                      economy::kInterestRate)));
      if (interest > 0) {
        game_state_->essence += interest;
        game_state_->essence_earned_this_level += interest;
      }
    }

    const int mastery_essence = std::max(0, game_state_->echoFoundationEssenceBonus());
    if (mastery_essence > 0) {
      game_state_->essence += mastery_essence;
      game_state_->essence_earned_this_level += mastery_essence;
    }

    const int base_bonus = std::max(0, game_state_->nexusVaultStartHpBonus());
    if (base_bonus > 0) {
      const int max_hp = std::max(1, level::kBaseHp + base_bonus);
      const int healed =
          std::clamp(game_state_->base_hp + base_bonus, 0, max_hp);
      game_state_->base_hp = healed;
    }

    if (prev_best == 0 && stars > 0) {
      game_state_->shards += economy::kShardsFirstTimeCompleteAnyLevel;
      const int bonus = std::max(0, game_state_->shardEyeFirstTimeCompleteBonusShards());
      if (bonus > 0) {
        game_state_->shards += bonus;
      }
    }
    if (prev_best < 3 && stars == 3) {
      game_state_->shards += economy::kShardsFirstTime3StarAnyLevel;
    }

    if (game_state_->isRelicEquipped(RelicId::ShardHunger)) {
      const int kills = std::max(0, game_state_->enemies_killed_this_level);
      const int bonus = std::max(
          0, (kills / relics::kShardHungerKillsStep) *
                 relics::kShardHungerBonusShardsPer100Kills);
      if (bonus > 0) {
        game_state_->shards += bonus;
      }
    }

    game_state_->lifetime_levels_completed =
        std::max(0, game_state_->lifetime_levels_completed + 1);
    game_state_->lifetime_stars_earned =
        std::max(0, game_state_->lifetime_stars_earned + stars);
    game_state_->recomputeMetaProgression();
  }
}

void LevelManager::enterFailed() {
  state_ = LevelManagerState::Failed;
  banner_remaining_sec_ = level::kLevelFailedBannerDurationSec;
  setBanner("LEVEL FAILED");

  if (game_state_) {
    game_state_->clearWaveBuffs();
  }
}

void LevelManager::tick(float dt) {
  const float step = std::max(0.0f, dt);

  if (banner_remaining_sec_ > 0.0f) {
    banner_remaining_sec_ = std::max(0.0f, banner_remaining_sec_ - step);
    if (banner_remaining_sec_ <= 0.0f) {
      banner_[0] = '\0';
    }
  }

  if (game_state_) {
    game_state_->level_time_sec += step;
  }

  if (base_ && base_id_ != INVALID_ID) {
    if (base_->getHp(base_id_) <= 0 &&
        state_ != LevelManagerState::Failed) {
      enterFailed();
      return;
    }
  }

  switch (state_) {
  case LevelManagerState::WaveClear: {
    grace_remaining_sec_ = std::max(0.0f, grace_remaining_sec_ - step);
    if (grace_remaining_sec_ <= 0.0f) {
      state_ = LevelManagerState::Playing;
      if (game_state_) {
        game_state_->onWaveStarted();
      }

      bool wave_elite = def_.is_elite;
      bool spawn_boss = isBossWave();
      float extra_hp_mult = 1.0f;
      float extra_speed_mult = 1.0f;

      if (game_state_) {
        if (game_state_->hasWaveBuff(WaveBuffId::SlowTide)) {
          extra_speed_mult *= wave_shop::kSlowTideSpeedMultiplier;
        }
        if (game_state_->hasWaveBuff(WaveBuffId::Foresight)) {
          wave_elite = false;
          spawn_boss = false;
        }
      }

      const EnemyType intro =
          intro_enemy_for_wave(def_.level_number, wave_index_, def_.wave_count);
      const float intro_delay =
          (intro != EnemyType::Count) ? enemies::kNewEnemyIntroPauseSec : 0.0f;

      spawner_.startWave(wave_index_, spawn_boss, wave_elite, extra_hp_mult,
                         extra_speed_mult, intro_delay);

      if (intro != EnemyType::Count) {
        banner_remaining_sec_ = enemies::kNewEnemyIntroPauseSec;
        const std::string_view name = to_string(intro);
        setBanner("NEW ENEMY: %.*s", static_cast<int>(name.size()), name.data());
      } else {
        banner_remaining_sec_ = spawn_boss ? level::kBossWaveBannerDurationSec
                                             : level::kWaveStartBannerDurationSec;
        if (spawn_boss) {
          setBanner("BOSS WAVE");
        } else {
          setBanner("WAVE %d / %d", wave_index_ + 1, def_.wave_count);
        }
      }

      if (game_state_) {
        game_state_->wave_index = wave_index_;
      }
    }
    break;
  }
  case LevelManagerState::Playing: {
    spawner_.tick(step);
    const bool done_spawning = !spawner_.isSpawning();
    if (done_spawning && countVisibleEnemies() == 0) {
      if (wave_index_ >= def_.wave_count - 1) {
        enterLevelClear();
      } else {
        const std::int32_t cleared = wave_index_;
        wave_index_++;
        enterWaveClear(cleared);
      }
    }
    break;
  }
  case LevelManagerState::LevelClear:
  case LevelManagerState::Failed:
  default:
    break;
  }
}

} // namespace tower_swarm
