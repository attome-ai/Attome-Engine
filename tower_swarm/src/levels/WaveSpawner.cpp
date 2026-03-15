#include "levels/WaveSpawner.h"

#include "Constants.h"
#include "entities/EnemyContainer.h"
#include "levels/LevelScaler.h"

#include "ATMEngine.h"

#include <array>
#include <algorithm>
#include <cmath>

namespace tower_swarm {
namespace {

constexpr std::uint32_t kDefaultSeed = 0xC001D00Du;

std::uint32_t scramble_seed(std::uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352du;
  v ^= v >> 15;
  v *= 0x846ca68bu;
  v ^= v >> 16;
  return v;
}

} // namespace

void WaveSpawner::beginLevel(const LevelDefinition &def, std::uint32_t seed) {
  def_ = def;
  wave_index_ = 0;
  wave_spawn_boss_ = true;
  wave_is_elite_ = def.is_elite;
  wave_extra_hp_mult_ = 1.0f;
  wave_extra_speed_mult_ = 1.0f;
  spawning_ = false;
  spawn_timer_sec_ = 0.0f;
  inter_spawn_delay_sec_ =
      std::max(0.0f, level::kInterSpawnDelaySec);
  rng_state_ = scramble_seed(seed == 0 ? kDefaultSeed : seed);
  batches_.clear();
  batch_cursor_ = 0;
}

void WaveSpawner::setInterSpawnDelaySec(float sec) {
  inter_spawn_delay_sec_ = std::max(0.0f, sec);
}

bool WaveSpawner::isBossWave() const {
  return is_boss_wave(wave_index_, def_.wave_count);
}

void WaveSpawner::startWave(std::int32_t wave_index, bool spawn_boss,
                            bool is_elite, float extra_hp_multiplier,
                            float extra_speed_multiplier, float initial_delay_sec) {
  wave_index_ = std::max<std::int32_t>(0, wave_index);
  wave_spawn_boss_ = spawn_boss;
  wave_is_elite_ = is_elite;
  wave_extra_hp_mult_ = std::max(0.0f, extra_hp_multiplier);
  wave_extra_speed_mult_ = std::max(0.0f, extra_speed_multiplier);
  spawning_ = true;
  spawn_timer_sec_ = std::max(0.0f, initial_delay_sec);
  batches_.clear();
  batch_cursor_ = 0;
  buildWaveBatches();
}

void WaveSpawner::tick(float dt) {
  if (!spawning_ || !enemies_) {
    return;
  }

  const float step = std::max(0.0f, dt);
  spawn_timer_sec_ = std::max(0.0f, spawn_timer_sec_ - step);

  if (!hasPending()) {
    spawning_ = false;
    return;
  }

  while (spawn_timer_sec_ <= 0.0f && hasPending()) {
    spawnOne();
    spawn_timer_sec_ += inter_spawn_delay_sec_;
    if (inter_spawn_delay_sec_ <= 0.0f) {
      spawn_timer_sec_ = 0.0f;
      break;
    }
  }

  if (!hasPending()) {
    spawning_ = false;
  }
}

void WaveSpawner::buildWaveBatches() {
  if (def_.wave_count <= 0) {
    return;
  }

  if (wave_spawn_boss_ && is_boss_wave(wave_index_, def_.wave_count)) {
    SpawnBatch boss{};
    boss.type = def_.boss_type;
    boss.remaining = 1;
    boss.extra_hp_mult = wave_extra_hp_mult_;
    boss.extra_speed_mult = wave_extra_speed_mult_;
    batches_.push_back(boss);
    return;
  }

  const int level_number = std::max(1, static_cast<int>(def_.level_number));
  const int wave_total = std::max(
      1, static_cast<int>(wave_enemy_count(level_number, wave_index_)));

  struct WeightedType final {
    EnemyType type{EnemyType::Grub};
    float weight{1.0f};
  };

  auto weight_for = [&](EnemyType t) -> float {
    float w = 1.0f;
    switch (def_.biome) {
    case Biome::VerdantFields:
      if (t == EnemyType::Grub) {
        w = 3.0f;
      } else if (t == EnemyType::Hulk) {
        w = 2.0f;
      }
      break;
    case Biome::Ashlands:
      if (t == EnemyType::Scuttle) {
        w = 3.0f;
      } else if (t == EnemyType::Driftwing) {
        w = 2.0f;
      }
      break;
    case Biome::Frostmarsh:
      if (t == EnemyType::Divide) {
        w = 3.0f;
      } else if (t == EnemyType::Vanguard) {
        w = 2.0f;
      }
      break;
    case Biome::Deepcore:
      if (t == EnemyType::Mender) {
        w = 3.0f;
      }
      break;
    case Biome::TheVoid:
    case Biome::Count:
    default:
      break;
    }
    return std::max(0.0f, w);
  };

  auto unlocked = [&](EnemyType t) -> bool {
    switch (t) {
    case EnemyType::Grub:
      return level_number >= enemies::kIntroLevelGrub;
    case EnemyType::Hulk:
      return level_number >= enemies::kIntroLevelHulk;
    case EnemyType::Scuttle:
      return level_number >= enemies::kIntroLevelScuttle &&
             wave_total >= enemies::kScuttlePackMin;
    case EnemyType::Driftwing:
      return level_number >= enemies::kIntroLevelDriftwing;
    case EnemyType::Divide:
      return level_number >= enemies::kIntroLevelDivide;
    case EnemyType::Vanguard:
      return level_number >= enemies::kIntroLevelVanguard;
    case EnemyType::Mender:
      return level_number >= enemies::kIntroLevelMender;
    case EnemyType::SiegeLord:
    case EnemyType::Count:
      break;
    }
    return false;
  };

  std::vector<WeightedType> types;
  types.reserve(7);
  for (EnemyType t :
       {EnemyType::Grub, EnemyType::Hulk, EnemyType::Scuttle, EnemyType::Driftwing,
        EnemyType::Divide, EnemyType::Vanguard, EnemyType::Mender}) {
    if (!unlocked(t)) {
      continue;
    }
    const float w = weight_for(t);
    if (w <= 0.0f) {
      continue;
    }
    types.push_back(WeightedType{t, w});
  }

  if (types.empty()) {
    SpawnBatch batch{};
    batch.type = EnemyType::Grub;
    batch.remaining = wave_total;
    batch.extra_hp_mult = wave_extra_hp_mult_;
    batch.extra_speed_mult = wave_extra_speed_mult_;
    batches_.push_back(batch);
    return;
  }

  std::vector<int> counts(types.size(), 0);

  const auto scuttle_it = std::find_if(types.begin(), types.end(), [](const auto &wt) {
    return wt.type == EnemyType::Scuttle;
  });
  const int scuttle_idx =
      (scuttle_it == types.end()) ? -1 : static_cast<int>(scuttle_it - types.begin());

  int remaining = wave_total;

  if (scuttle_idx >= 0) {
    float sum_w = 0.0f;
    for (const auto &wt : types) {
      sum_w += wt.weight;
    }
    const float sw = types[static_cast<std::size_t>(scuttle_idx)].weight;
    const int desired =
        static_cast<int>(std::lround(static_cast<float>(wave_total) * (sw / std::max(0.001f, sum_w))));
    const int scuttle_count =
        std::clamp(desired, enemies::kScuttlePackMin, wave_total);
    counts[static_cast<std::size_t>(scuttle_idx)] = scuttle_count;
    remaining = std::max(0, remaining - scuttle_count);
  }

  float sum_w_others = 0.0f;
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (static_cast<int>(i) == scuttle_idx) {
      continue;
    }
    sum_w_others += types[i].weight;
  }

  if (remaining > 0 && sum_w_others > 0.0f) {
    int allocated = 0;
    for (std::size_t i = 0; i < types.size(); ++i) {
      if (static_cast<int>(i) == scuttle_idx) {
        continue;
      }
      const float frac = types[i].weight / sum_w_others;
      const int c = static_cast<int>(std::floor(static_cast<float>(remaining) * frac));
      counts[i] = std::max(0, c);
      allocated += counts[i];
    }

    int extra = std::max(0, remaining - allocated);
    while (extra-- > 0) {
      std::size_t best = 0;
      float best_w = -1.0f;
      for (std::size_t i = 0; i < types.size(); ++i) {
        if (static_cast<int>(i) == scuttle_idx) {
          continue;
        }
        if (types[i].weight > best_w) {
          best_w = types[i].weight;
          best = i;
        }
      }
      counts[best] += 1;
    }
  } else if (remaining > 0 && scuttle_idx >= 0) {
    counts[static_cast<std::size_t>(scuttle_idx)] += remaining;
  } else if (remaining > 0) {
    counts[0] += remaining;
  }

  EnemyType intro = EnemyType::Count;
  if (level_number == enemies::kIntroLevelHulk && wave_index_ == 0) {
    intro = EnemyType::Hulk;
  } else if (level_number == enemies::kIntroLevelScuttle) {
    int first = 0;
    const int last_normal_wave = std::max(0, def_.wave_count - 2);
    for (int w = 0; w <= last_normal_wave; ++w) {
      if (wave_enemy_count(level_number, w) >= enemies::kScuttlePackMin) {
        first = w;
        break;
      }
    }
    if (wave_index_ == first && wave_total >= enemies::kScuttlePackMin) {
      intro = EnemyType::Scuttle;
    }
  } else if (level_number == enemies::kIntroLevelDriftwing && wave_index_ == 0) {
    intro = EnemyType::Driftwing;
  } else if (level_number == enemies::kIntroLevelDivide && wave_index_ == 0) {
    intro = EnemyType::Divide;
  } else if (level_number == enemies::kIntroLevelVanguard && wave_index_ == 0) {
    intro = EnemyType::Vanguard;
  } else if (level_number == enemies::kIntroLevelMender && wave_index_ == 0) {
    intro = EnemyType::Mender;
  }

  if (intro != EnemyType::Count) {
    const auto intro_type_it = std::find_if(types.begin(), types.end(), [&](const auto &wt) {
      return wt.type == intro;
    });
    if (intro_type_it != types.end()) {
      const std::size_t intro_idx = static_cast<std::size_t>(intro_type_it - types.begin());
      if (intro_idx < counts.size() && counts[intro_idx] <= 0 && wave_total > 0) {
        std::size_t donor = 0;
        int donor_count = -1;
        for (std::size_t i = 0; i < counts.size(); ++i) {
          if (i == intro_idx) {
            continue;
          }
          if (counts[i] > donor_count) {
            donor = i;
            donor_count = counts[i];
          }
        }
        if (donor_count > 0) {
          counts[donor] = std::max(0, counts[donor] - 1);
          counts[intro_idx] = 1;
        }
      }
    }
  }

  std::vector<std::size_t> order;
  order.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (counts[i] > 0) {
      order.push_back(i);
    }
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return types[a].weight > types[b].weight;
  });

  if (intro != EnemyType::Count) {
    const auto it = std::find_if(order.begin(), order.end(), [&](std::size_t idx) {
      return types[idx].type == intro;
    });
    if (it != order.end() && it != order.begin()) {
      const std::size_t idx = *it;
      order.erase(it);
      order.insert(order.begin(), idx);
    }
  }

  for (const std::size_t idx : order) {
    SpawnBatch batch{};
    batch.type = types[idx].type;
    batch.remaining = counts[idx];
    batch.extra_hp_mult = wave_extra_hp_mult_;
    batch.extra_speed_mult = wave_extra_speed_mult_;
    batches_.push_back(batch);
  }
}

void WaveSpawner::nextRandom(std::uint32_t &state) const {
  // xorshift32
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
}

std::uint32_t WaveSpawner::randomU32(std::uint32_t &state) const {
  nextRandom(state);
  return state;
}

float WaveSpawner::random01(std::uint32_t &state) const {
  const std::uint32_t v = randomU32(state);
  return static_cast<float>(v & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

void WaveSpawner::pickSpawnPosition(float &out_x, float &out_y) {
  const float w = static_cast<float>(kWorldWidthPx);
  const float h = static_cast<float>(kWorldHeightPx);
  const float size = static_cast<float>(kEnemyBaseSizePx);
  const float max_x = std::max(0.0f, w - size);
  const float max_y = std::max(0.0f, h - size);

  std::array<int, 4> edges = {0, 1, 2, 3}; // top, bottom, left, right
  std::size_t edge_count = edges.size();

  switch (def_.map_variant % 5) {
  case 1:
    edges = {0, 1, 0, 1};
    edge_count = 2;
    break;
  case 2:
    edges = {2, 3, 2, 3};
    edge_count = 2;
    break;
  case 3:
    edges = {0, 2, 0, 2};
    edge_count = 2;
    break;
  case 4:
    edges = {1, 3, 1, 3};
    edge_count = 2;
    break;
  default:
    break;
  }

  const std::uint32_t rv = randomU32(rng_state_);
  const int edge = edges[static_cast<std::size_t>(rv % edge_count)];
  const float t = random01(rng_state_);

  switch (edge) {
  case 0: // top
    out_x = max_x * t;
    out_y = 0.0f;
    break;
  case 1: // bottom
    out_x = max_x * t;
    out_y = max_y;
    break;
  case 2: // left
    out_x = 0.0f;
    out_y = max_y * t;
    break;
  case 3: // right
  default:
    out_x = max_x;
    out_y = max_y * t;
    break;
  }
}

void WaveSpawner::spawnOne() {
  if (!enemies_ || batch_cursor_ >= batches_.size()) {
    return;
  }

  SpawnBatch &batch = batches_[batch_cursor_];
  if (batch.remaining <= 0) {
    batch_cursor_++;
    return;
  }

  float x = 0.0f;
  float y = 0.0f;
  pickSpawnPosition(x, y);

  const bool elite = wave_is_elite_;

  const int pack_min = enemies::kScuttlePackMin;
  const int pack_max = enemies::kScuttlePackMax;

  if (batch.type == EnemyType::Scuttle && batch.remaining >= pack_min) {
    const int remaining = std::max(0, static_cast<int>(batch.remaining));
    int pack = std::min(pack_max, remaining);
    if (remaining <= pack_max) {
      pack = remaining;
    } else {
      const int upper = std::min(pack_max, remaining - pack_min);
      if (upper >= pack_min) {
        const std::uint32_t rv = randomU32(rng_state_);
        pack = pack_min +
               static_cast<int>(rv % static_cast<std::uint32_t>(upper - pack_min + 1));
      }
    }

    int spawned_count = 0;
    for (int i = 0; i < pack; ++i) {
      const float ox = (random01(rng_state_) * 2.0f - 1.0f) * 18.0f;
      const float oy = (random01(rng_state_) * 2.0f - 1.0f) * 18.0f;
      const float px = std::clamp(x + ox, 0.0f,
                                  std::max(0.0f, static_cast<float>(kWorldWidthPx - kEnemyBaseSizePx)));
      const float py = std::clamp(y + oy, 0.0f,
                                  std::max(0.0f, static_cast<float>(kWorldHeightPx - kEnemyBaseSizePx)));
      const EntityHandle spawned =
          enemies_->spawnEnemy(batch.type, px, py, def_.level_number, wave_index_,
                               elite, batch.extra_hp_mult, batch.extra_speed_mult);
      if (spawned == INVALID_ID) {
        break;
      }
      spawned_count += 1;
    }

    if (spawned_count > 0) {
      batch.remaining = std::max<std::int32_t>(0, batch.remaining - spawned_count);
    } else {
      batch.remaining = 0;
    }
  } else {
    const EntityHandle spawned =
        enemies_->spawnEnemy(batch.type, x, y, def_.level_number, wave_index_,
                             elite, batch.extra_hp_mult, batch.extra_speed_mult);

    if (spawned != INVALID_ID) {
      batch.remaining = std::max<std::int32_t>(0, batch.remaining - 1);
    } else {
      batch.remaining = 0;
    }
  }

  while (batch_cursor_ < batches_.size() &&
         batches_[batch_cursor_].remaining <= 0) {
    batch_cursor_++;
  }
}

} // namespace tower_swarm
