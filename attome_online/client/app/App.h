#pragma once

// The game client: window, renderer, audio, input, world streaming,
// networking, prediction, entities, HUD.

#include "net/NetClient.h"
#include "net/Prediction.h"
#include "fx/Particles.h"
#include "fx/Decor.h"
#include "ui/LookSettings.h"
#include "world/MapCache.h"

#include "shared/GameTypes.h"
#include "shared/Protocol.h"

#include "../../../engine/ATMAudio.h"
#include "../../../engine/ATMInput.h"
#include "../../../engine/model/Character.h"
#include "../../../engine/render/Renderer.h"
#include "../../../engine/voxel/BlockRegistry.h"
#include "../../../engine/voxel/VoxelWorld.h"

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct SDL_Window;

namespace ao::server {
class ZoneServer;
}

namespace ao::client {

class SfxPlayer;

struct ClientConfig {
  std::string name = "Player";
  std::string host = "127.0.0.1";
  uint16_t port = 27015;
  bool local = false;              // start an in-process server
  std::string serverConfigPath = "config/server.json";
  float mouseSensitivity = 0.0025f;
  bool invertY = false;
  bool profile = false;            // --profile: start with the profiler panel open
  int windowWidth = 1600, windowHeight = 900;
  bool fullscreen = false;
  atm::render::RendererConfig render;
  int viewRadiusChunks = 12;
  float masterVolume = 0.8f;
};

bool loadClientConfig(const std::string &path, ClientConfig &cfg, std::string *error);

// A remote entity (other players, monsters, dropped items, projectiles).
struct RemoteEntity {
  EntityId id = kNoEntity;
  EntityState last;                 // last full state (fills omitted delta fields)
  RemoteTrack track;
  atm::model::Animator animator;
  atm::model::Appearance appearance;
  std::string name;
  uint8_t lastActionSeq = 0;
  float hitFlash = 0.0f;            // seconds of red tint left
  float lastSeenTick = 0.0f;
  float stepDistance = 0.0f;        // footstep sounds
  glm::dvec3 trailFrom{1e30};       // projectiles: last trail spark position
};

struct FloatingText {
  glm::dvec3 world{0.0};
  std::string text;
  uint32_t color = 0xFFFFFFFF;
  float age = 0.0f, life = 1.2f;
};

class App : public NetHandler {
public:
  App();
  ~App() override;

  bool init(const ClientConfig &cfg, std::string *error);
  int run();
  void shutdown();

  // NetHandler
  void onConnected() override;
  void onDisconnected(atm::net2::DisconnectReason reason) override;
  void onWelcome(const proto::Welcome &m) override;
  void onChunkData(const proto::ChunkData &m) override;
  void onEditedChunks(const proto::EditedChunks &m) override;
  void onBlockChanged(const proto::BlockChanged &m) override;
  void onSnapshot(const proto::SnapshotMsg &m) override;
  void onAppearance(const proto::AppearanceMsg &m) override;
  void onDamage(const proto::DamageEvent &m) override;
  void onInventory(const proto::InventoryMsg &m) override;
  void onXpGain(const proto::XpGain &m) override;
  void onLoot(const proto::LootMsg &m) override;
  void onChat(const proto::ChatMsg &m) override;

private:
  // Frame stages
  void handleEvents();
  void fixedTick();                       // 30 Hz: input -> prediction -> send
  void updateWorld();
  void updateCamera(float dt);
  void updateInteraction(float dt);       // mining, placing, attacking
  void updateEntities(float dt);
  void render(float alpha, float dt);
  void drawHud(float dt);
  void drawLookPanel();                   // F10 graphics tuning (ui/GraphicsPanel.cpp)
  void drawMinimap();                     // ui/MapUi.cpp
  void drawWorldMap();                    // M (ui/MapUi.cpp)
  void drawProfilerPanel();               // F7 (ui/ProfilerPanel.cpp)

  // Helpers
  MoveInput buildInput();
  glm::dvec3 eyePosition(float alpha) const;
  glm::vec3 aimDirection() const;
  // Direction from `from` to what the crosshair points at (first block or
  // monster along the camera ray, else a far point). Corrects the
  // over-the-shoulder camera offset so shots land on the crosshair.
  glm::vec3 crosshairAim(const glm::dvec3 &from) const;
  bool worldToScreen(const glm::dvec3 &p, float &sx, float &sy) const;
  void uploadMaterials();
  void createModelMeshes();
  void addFloatingText(const glm::dvec3 &at, std::string text, uint32_t color);
  void addChatLine(std::string line);
  void startLocalServer();
  void stopLocalServer();
  int selectedSlot() const { return hotbar_; }
  ItemId equippedMainHand() const;

  // Character / model drawing (CharacterDraw.cpp)
  void drawCharacter(const atm::model::Appearance &appearance,
                     const atm::model::Animator &animator, const glm::dvec3 &feet,
                     float yaw, uint32_t tint);
  void drawMonster(uint8_t type, const atm::model::Animator &animator,
                   const glm::dvec3 &feet, float yaw, uint32_t tint);
  void drawItemEntity(uint16_t item, const glm::dvec3 &pos, float spin);
  void drawProjectile(const glm::dvec3 &pos, const glm::vec3 &vel);
  // creature = ModelLibrary creature index for monsters, -1 for humanoids.
  void selectLocomotion(atm::model::Animator &anim, const glm::vec3 &vel,
                        uint8_t flags, bool isLocal, int creature = -1);
  void playActionAnim(atm::model::Animator &anim, uint8_t action, WeaponType weapon,
                      int creature = -1);
  int creatureForMonster(uint8_t type);

  // Config and platform
  ClientConfig cfg_;
  SDL_Window *window_ = nullptr;
  bool running_ = false;
  bool mouseCaptured_ = false;
  bool sdlInit_ = false;           // what init() got through (shutdown() undoes only that)
  bool imguiContext_ = false;
  bool imguiSdlInit_ = false;

  // Engine systems
  atm::render::Renderer renderer_;
  atm::Audio audio_;
  std::unique_ptr<SfxPlayer> sfx_;
  atm::InputMap input_;
  atm::voxel::BlockRegistry blocks_;
  atm::model::ModelLibrary models_;
  atm::model::AnimLibrary anims_;
  std::vector<atm::render::ModelMeshId> partMeshes_;   // per ModelLibrary part
  std::vector<atm::render::ModelMeshId> blockItemMeshes_; // per block id (dropped items)
  atm::render::ModelMeshId arrowMesh_ = atm::render::kInvalidModelMesh;
  std::vector<int> monsterCreature_;                   // monster type -> creature (-2 = not looked up)
  std::unique_ptr<atm::voxel::VoxelWorld> world_;
  std::vector<atm::voxel::ChunkMeshResult> meshResults_;
  std::vector<atm::voxel::ChunkCoord> unloaded_;

  // Networking
  NetClient net_;
  std::unique_ptr<server::ZoneServer> localServer_;
  std::thread localServerThread_;
  std::atomic<bool> localServerStop_{false};
  bool welcomed_ = false;
  uint64_t worldSeed_ = 0;
  EntityId selfId_ = kNoEntity;

  // Local player
  Prediction prediction_;
  atm::model::Animator selfAnim_;
  atm::model::Appearance selfAppearance_;
  uint32_t nextInputSeq_ = 1;
  Tick clientTick_ = 0;
  uint32_t newestSnapshotTick_ = 0;
  float serverTickEstimate_ = 0.0f;     // for remote interpolation
  uint16_t hp_ = 100, maxHp_ = 100;
  uint16_t pendingButtons_ = 0;         // edge-triggered buttons collected between ticks
  float footstepDistance_ = 0.0f;
  bool wasOnGround_ = true;

  // Remote entities
  std::unordered_map<EntityId, RemoteEntity> remotes_;

  // Inventory, skills
  std::vector<ItemStack> inventory_ = std::vector<ItemStack>(kInventorySlots);
  std::vector<uint16_t> equipped_ = std::vector<uint16_t>(atm::model::kEquipSlotCount, 0);
  std::array<uint64_t, kSkillCount> skillXp_{};
  int hotbar_ = 0;

  // Camera
  float camYaw_ = 0.0f, camPitch_ = -0.25f, camDistance_ = 6.0f;
  glm::dvec3 camPos_{0.0};
  atm::render::Camera camera_;
  float aspect_ = 16.0f / 9.0f;

  // Interaction
  bool hasTarget_ = false;
  BlockPos targetBlock_{};
  atm::voxel::FaceDir targetFace_ = atm::voxel::FaceDir::PosY;
  float mineProgress_ = 0.0f;
  BlockPos miningBlock_{};
  BlockPos gatherWarnBlock_{};        // last "you need an axe" message (no spam)
  float gatherWarnCooldown_ = 0.0f;
  bool worldEditable_ = false;        // overworld: gather only; instances / plots later
  float attackCooldown_ = 0.0f;
  float placeCooldown_ = 0.0f;

  // HUD
  std::deque<std::string> chat_;
  std::vector<FloatingText> floating_;
  struct XpDrop { Skill skill; uint32_t amount; float age; };
  std::vector<XpDrop> xpDrops_;
  struct Banner { std::string text; float age; };
  std::vector<Banner> banners_;
  bool showInventory_ = false, showSkills_ = false, showDebug_ = false;
  bool showLook_ = false;
  bool showHitboxes_ = false;             // F8: collision / hit volumes
  uint64_t nextPickupMs_ = 0;             // E held: resend Pickup at most every 250 ms
  atm::render::ModelMeshId lootBeamMesh_ = atm::render::kInvalidModelMesh;
  // Decoration props placed in the world (home town now; homes / clan plots
  // later): voxel model parts at 16 voxels per block, drawn within range.
  struct PlacedProp {
    int part = -1;
    glm::dvec3 pos{0.0};
    float yaw = 0.0f;
  };
  std::vector<PlacedProp> worldProps_;
  void buildWorldProps();
  void drawWorldProps();
  MapCache mapCache_;                    // explored terrain colours (minimap, world map)
  bool showWorldMap_ = false;
  bool showProfiler_ = false;             // F7: frame profiler panel
  float worldMapZoom_ = 1.5f;            // pixels per block
  glm::dvec2 worldMapPan_{0.0};          // blocks from the player (x, z)
  const MapRegion *lastRegion_ = nullptr; // region banner on entry
  LookSettings look_;                    // graphics panel values (config/graphics.json)
  std::string lookStatus_;               // last save / load message
  bool chatOpen_ = false;
  float chatIdle_ = 0.0f;                // seconds since the last chat line (fade out)
  float hpTrail_ = 1.0f;                 // lagging health fraction (damage trail)
  float shake_ = 0.0f;                   // camera impact shake (0..1)
  int sentHotbar_ = -1;                  // hotbar slot last reported to the server
  Particles particles_;                  // voxel VFX (hits, slashes, debris, motes)
  float bodyYaw_ = 0.0f;                 // rendered facing of the local character
  Decor decor_;                          // grass tufts, flowers, pebbles
  uint32_t decorMaterialBase_ = 0;
  uint32_t microMaterialBase_ = 0;   // fine-voxel structure palette (ao::world::microPalette)
  // Fine-voxel structures (town buildings): 32^3-voxel tiles meshed a few per
  // frame after joining, drawn as models at 1/4 block per voxel.
  struct StructureTile {
    atm::render::ModelMeshId mesh = atm::render::kInvalidModelMesh;
    glm::dvec3 origin{0.0}, center{0.0};
    double radius = 0.0;
  };
  struct PendingTile {
    int model, tx, ty, tz;
  };
  std::vector<StructureTile> structureTiles_;
  std::vector<PendingTile> pendingTiles_;
  size_t pendingTileCursor_ = 0;
  void clearStructures();
  void buildStructureMeshes(float budgetMs);
  void drawStructures();
  char chatInput_[200] = {};
  std::string status_ = "Connecting...";

  // Timing
  double accumulator_ = 0.0;
  uint64_t lastCounter_ = 0;
  float fps_ = 0.0f;
};

} // namespace ao::client
