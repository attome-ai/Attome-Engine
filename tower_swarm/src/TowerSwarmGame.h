#pragma once

#include "CameraController.h"
#include "Constants.h"
#include "characters/CharacterId.h"
#include "entities/EnemyType.h"
#include "levels/GameState.h"
#include "levels/LevelManager.h"
#include "levels/SaveState.h"
#include "screens/HUD.h"
#include "shop/WaveBuffShop.h"
#include "systems/PathGrid.h"

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

struct Engine;
typedef std::uint32_t EntityHandle;

namespace tower_swarm {

class InputManager;
class BaseEntity;
class CreatureContainer;
class EnemyContainer;
class ProjectileContainer;
class PickupContainer;
class TileContainer;

class TowerSwarmGame final {
public:
  explicit TowerSwarmGame(Engine *engine);
  ~TowerSwarmGame();

  bool initialize();
  void tick(float dt, const InputManager &input);
  void renderHUD(const InputManager &input);

private:
  bool show_main_menu_{true};

  enum class InterLevelTab : std::uint8_t {
    Bazaar = 0,
    Forge = 1,
    Relics = 2,
    Repair = 3,
  };

  enum class ArmoryTab : std::uint8_t {
    Characters = 0,
    Masteries = 1,
    Relics = 2,
    Cosmetics = 3,
  };

  enum class ArmoryConfirmKind : std::uint8_t {
    None = 0,
    UnlockCharacter = 1,
    BuyMasteryRank = 2,
    UnlockRelic = 3,
  };

  struct BazaarOffer final {
    CharacterId character{CharacterId::Brix};
    Rarity rarity{Rarity::Common};
    std::int32_t cost_essence{0};
    bool purchased{false};
  };

  struct MergeAnim final {
    bool active{false};
    float elapsed_sec{0.0f};
    EntityHandle a{0xFFFFFFFFu};
    EntityHandle b{0xFFFFFFFFu};
    CharacterId character{CharacterId::Brix};
    float a_cx{0.0f};
    float a_cy{0.0f};
    float b_cx{0.0f};
    float b_cy{0.0f};
    float target_cx{0.0f};
    float target_cy{0.0f};
    int keep_roster_index{-1};
    int remove_roster_index{-1};
    int new_tier{1};
    int new_kills{0};
    std::size_t cell_a_idx{0};
    std::size_t cell_b_idx{0};
    std::size_t target_cell_idx{0};
  };

  Engine *engine_{nullptr};

  CameraController camera_;
  bool show_debug_grid_{false};

  TileContainer *tiles_{nullptr};
  BaseEntity *base_{nullptr};
  EntityHandle base_id_{0xFFFFFFFFu};
  int base_texture_id_{-1};

  GameState game_state_{};
  LevelManager level_manager_{};
  PersistentSnapshot level_start_snapshot_{};
  bool have_level_start_snapshot_{false};
  LevelManagerState last_level_state_{LevelManagerState::Playing};

  CreatureContainer *creatures_{nullptr};
  EnemyContainer *enemies_{nullptr};
  ProjectileContainer *projectiles_{nullptr};
  PickupContainer *pickups_{nullptr};

  std::array<std::array<int, evolution::kVisualBandCount>,
             static_cast<std::size_t>(CharacterId::Count)>
      creature_texture_{};
  std::array<int, static_cast<std::size_t>(CharacterId::Count)>
      projectile_texture_{};
  std::array<int, static_cast<std::size_t>(EnemyType::Count)> enemy_texture_{};
  int projectile_texture_id_{-1};
  int pickup_texture_id_{-1};

  float debug_enemy_spawn_accum_{0.0f};
  std::int32_t selected_roster_index_{0};
  EntityHandle selected_creature_{0xFFFFFFFFu};
  bool show_sell_confirm_{false};
  EntityHandle pending_sell_creature_{0xFFFFFFFFu};
  std::int32_t pending_sell_roster_index_{-1};
  bool show_level_select_{false};
  std::int32_t level_select_level_{1};
  bool show_armory_{false};
  ArmoryTab armory_tab_{ArmoryTab::Characters};
  CharacterId armory_selected_character_{CharacterId::Brix};
  bool show_armory_character_detail_{false};
  bool show_armory_confirm_{false};
  ArmoryConfirmKind armory_confirm_kind_{ArmoryConfirmKind::None};
  CharacterId armory_confirm_character_{CharacterId::Brix};
  MasteryId armory_confirm_mastery_{MasteryId::EchoFoundation};
  RelicId armory_confirm_relic_{RelicId::None};
  bool show_inter_level_{false};
  InterLevelTab inter_level_tab_{InterLevelTab::Bazaar};
  float inter_level_elapsed_sec_{0.0f};
  std::uint32_t inter_level_rng_{0};
  std::array<BazaarOffer, inter_level_shop::kBazaarOfferCount> bazaar_offers_{};
  bool bazaar_rerolled_{false};
  bool show_bazaar_duplicate_confirm_{false};
  std::int32_t pending_bazaar_offer_index_{-1};
  CharacterId forge_selected_{CharacterId::Brix};
  bool repair_purchased_{false};
  RelicId relic_pick_{RelicId::None};
  bool merge_drag_active_{false};
  EntityHandle merge_drag_source_{0xFFFFFFFFu};
  EntityHandle drag_candidate_{0xFFFFFFFFu};
  float drag_candidate_start_wx_{0.0f};
  float drag_candidate_start_wy_{0.0f};
  float merge_cooldown_remaining_sec_{0.0f};
  float auto_merge_idle_sec_{0.0f};
  std::vector<std::pair<EntityHandle, EntityHandle>> merge_pairs_{};
  MergeAnim merge_anim_{};

  bool ghost_valid_{false};
  bool ghost_active_{false};
  float ghost_world_x_{0.0f};
  float ghost_world_y_{0.0f};

  int grid_cols_{0};
  int grid_rows_{0};
  PathGrid path_grid_{};
  std::vector<EntityHandle> cell_occupant_{};
  std::vector<EntityHandle> deployed_roster_{};
  std::array<int, static_cast<std::size_t>(Biome::Count)> biome_tile_texture_{};
  Biome active_biome_{Biome::VerdantFields};

  HUD hud_{};
  WaveBuffShop wave_buff_shop_{};

  int registerBiomeTileTexture(Biome biome);
  void startLevel(std::int32_t level_number);
  void openInterLevel();
  void rerollBazaar();
  void applyWaveBuff(WaveBuffId id);
};

} // namespace tower_swarm
