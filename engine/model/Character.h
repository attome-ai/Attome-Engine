#pragma once

// Modular voxel characters (docs/GAME_DESIGN.md §10.1):
//   - one Rig (skeleton of rigid parts) shared by every player,
//   - animation clips authored once against the rig and shared by everyone,
//   - equipment pieces that replace or attach to rig bones.
// Swapping gear = changing a piece id in an Appearance; nothing is rebuilt.

#include "../voxel/MeshTypes.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace atm::model {

// World blocks per model voxel (renderer ModelInstance::voxelScale default).
// The humanoid is 26 voxels tall (~2.2 blocks incl. its big Trove head).
inline constexpr float kModelVoxelScale = 1.0f / 12.0f;

// ---------------------------------------------------------------------------
// Voxel parts
// ---------------------------------------------------------------------------

// A small voxel grid (<= 32 per axis) with its own colour palette. Parts are
// meshed with the same ChunkMesher as the world into PackedFace data.
struct VoxelPart {
  std::string name;
  int sx = 0, sy = 0, sz = 0;             // size in voxels (1..32)
  std::vector<uint8_t> voxels;            // sx*sy*sz palette indices, 0 = empty; index = (y*sz + z)*sx + x
  std::vector<uint32_t> palette;          // RGBA8, palette[0] unused
  glm::vec3 pivot{0.0f};                  // attachment/rotation point, voxel units
  uint8_t emissiveFrom = 255;             // palette indices >= this glow
  // Optional recolour variants (skin tones, hair colours): extra palettes of
  // palette.size() entries each, stored after `palette` in the material table.
  // Variant v is drawn with paletteOffset = v * palette.size().
  std::vector<uint32_t> variantPalettes;
  int variantCount() const {
    return palette.empty() ? 1 : 1 + int(variantPalettes.size() / palette.size());
  }

  uint8_t at(int x, int y, int z) const { return voxels[size_t((y * sz + z) * sx + x)]; }
  uint8_t &at(int x, int y, int z) { return voxels[size_t((y * sz + z) * sx + x)]; }
};

// ---------------------------------------------------------------------------
// Rig and animation
// ---------------------------------------------------------------------------

enum class Bone : uint8_t {
  Root, Pelvis, Torso, Head,
  ArmUpperL, ArmLowerL, HandL, ArmUpperR, ArmLowerR, HandR,
  LegUpperL, LegLowerL, FootL, LegUpperR, LegLowerR, FootR,
  Count
};
inline constexpr int kBoneCount = int(Bone::Count);

enum class Socket : uint8_t { MainHand, OffHand, Back, Head, Count };

struct RigBone {
  const char *name;
  Bone parent;          // Root's parent is Root
  glm::vec3 offset;     // bind-pose position relative to parent pivot (voxels)
};

struct Rig {
  std::array<RigBone, kBoneCount> bones;
  std::array<Bone, size_t(Socket::Count)> socketBone;
  std::array<glm::vec3, size_t(Socket::Count)> socketOffset;
  static const Rig &humanoid(); // the shared player rig
  // Monster rigs reuse the Bone enum: for the quadruped the Arm* chains are
  // the front legs, Leg* the hind legs, Pelvis the rear body, Torso the chest.
  static const Rig &quadruped();
  static const Rig &slime();    // everything on Pelvis (a bouncing blob)
};

struct BoneKey {
  float time = 0.0f;
  glm::quat rotation{1, 0, 0, 0};
  glm::vec3 offset{0.0f};             // added to the bind offset
};

struct AnimClip {
  std::string name;                   // "run", "jump", "sword_swing", ...
  float length = 1.0f;                // seconds
  bool loop = true;
  // One track per bone (empty track = bind pose).
  std::array<std::vector<BoneKey>, kBoneCount> tracks;
  // Which bones this clip drives when played on an upper-body layer.
  uint32_t boneMask = 0xFFFFFFFFu;
  // Event times (e.g. the hit frame of an attack), seconds.
  std::vector<std::pair<float, uint8_t>> events;
};

// Event ids used in AnimClip::events (Animator::takeEvents bit = 1 << id).
namespace anim_event {
inline constexpr uint8_t Hit = 1;       // melee hit frame
inline constexpr uint8_t Release = 2;   // arrow / spell leaves
inline constexpr uint8_t Footstep = 3;  // a foot touches the ground
inline constexpr uint8_t Impact = 4;    // pickaxe / tool hits the block
} // namespace anim_event

// Library of shared clips (built in code for the demo; data files later).
class AnimLibrary {
public:
  void buildDefaults();                       // idle, walk, run, jump, fall, glide, swim, dash, mine, place, attack per weapon, hit, death, wave
  const AnimClip *find(std::string_view name) const;
  int indexOf(std::string_view name) const;   // -1 if missing
  const AnimClip &at(int index) const { return clips_[size_t(index)]; }
  size_t size() const { return clips_.size(); }
  int add(AnimClip clip);                     // returns the index (replaces a clip with the same name)

private:
  std::vector<AnimClip> clips_;
};

// Two layers: full-body locomotion + upper-body action (attack, mine), with
// short cross-fades. Evaluates bone transforms in model space.
class Animator {
public:
  void setLocomotion(int clip, float fadeSeconds = 0.15f);
  void playAction(int clip, float fadeSeconds = 0.08f); // one-shot upper body
  void update(const AnimLibrary &lib, float dt);
  // Model-space transform of each bone (rotation + translation in voxels).
  void evaluate(const AnimLibrary &lib, const Rig &rig,
                std::array<glm::mat4, kBoneCount> &out) const;
  bool actionPlaying() const { return action_.clip >= 0; }
  float actionTime() const { return action_.time; }
  int locomotionClip() const { return loco_.clip; }
  int actionClip() const { return action_.clip; }
  void stopAction() { action_.clip = -1; }
  // Events crossed since the last call (bit 1 << anim_event id), both layers.
  uint32_t takeEvents() { const uint32_t e = events_; events_ = 0; return e; }

private:
  struct Layer { int clip = -1; float time = 0; int prevClip = -1; float prevTime = 0; float fade = 0, fadeLen = 0; };
  Layer loco_, action_;
  uint32_t events_ = 0;
};

// ---------------------------------------------------------------------------
// Equipment and appearance
// ---------------------------------------------------------------------------

enum class EquipSlot : uint8_t { Head, Torso, Hands, Legs, Feet, Back, MainHand, OffHand, Count };
inline constexpr int kEquipSlotCount = int(EquipSlot::Count);

using PieceId = uint16_t;
inline constexpr PieceId kNoPiece = 0;

// A wearable piece: either replaces the body part on some bones (armour,
// clothes) or attaches to a socket (weapons, capes, gliders).
struct EquipPiece {
  std::string name;
  EquipSlot slot = EquipSlot::Head;
  // For replacement pieces: part index per bone it replaces (-1 = keep body).
  std::array<int16_t, kBoneCount> boneParts;
  // For socket pieces: the part and socket.
  int16_t socketPart = -1;
  Socket socket = Socket::MainHand;
  uint8_t weaponType = 0;              // selects attack animations (0 = none)
  EquipPiece() { boneParts.fill(-1); }
};

// ~24 bytes; replicated over the network only when it changes.
struct Appearance {
  std::array<PieceId, kEquipSlotCount> pieces{}; // kNoPiece = nothing
  uint8_t skinTone = 0, hairStyle = 0, hairColor = 0, bodyShape = 0;
  std::array<uint8_t, 4> dyes{};                 // palette variants
};

// A non-player model (monster/NPC): a rig plus one part per bone.
struct CreatureModel {
  std::string name;                    // matches MonsterDef::model
  const Rig *rig = nullptr;
  std::array<int16_t, kBoneCount> boneParts; // -1 = nothing on that bone
  float scale = 1.0f;                  // multiplies kModelVoxelScale
  std::string idleClip, moveClip, attackClip, hitClip, deathClip;
  CreatureModel() { boneParts.fill(-1); }
};

// Owns every part and piece; builds meshes once and hands them to the
// renderer (ModelMeshId per part).
class ModelLibrary {
public:
  ModelLibrary() {
    for (auto &shape : body_)
      shape.fill(-1); // no body part on any bone until set
  }
  void buildDefaults();   // procedural demo body + gear set (see Content.cpp)
  const std::vector<VoxelPart> &parts() const { return parts_; }
  const EquipPiece &piece(PieceId id) const { return pieces_[id < pieces_.size() ? id : 0]; }
  PieceId findPiece(std::string_view name) const;
  size_t pieceCount() const { return pieces_.size(); }
  // Body part per bone for a given body shape (before equipment).
  int bodyPart(Bone bone, uint8_t bodyShape) const;
  int addPart(VoxelPart part);
  PieceId addPiece(EquipPiece piece);
  void setBodyPart(Bone bone, uint8_t bodyShape, int part);

  int findPart(std::string_view name) const;          // -1 if missing
  // Hair styles attach to Socket::Head (hidden by any Head piece).
  int hairPart(uint8_t style) const;                  // -1 = bald / unknown
  size_t hairStyleCount() const { return hair_.size(); }
  static constexpr uint8_t kSkinToneCount = 4, kHairColorCount = 4, kBodyShapeCount = 2;

  int addCreature(CreatureModel model);
  int findCreature(std::string_view name) const;      // -1 if missing
  const CreatureModel &creature(int index) const { return creatures_[size_t(index)]; }
  size_t creatureCount() const { return creatures_.size(); }

  // Material table for every part: part i uses materials
  // [materialBase(i), materialBase(i) + palette.size() * variantCount()).
  // Entry materialBase(i) + k is palette[k] (k = 0 is unused/transparent).
  // Upload these after the world's block materials and pass
  // (blockMaterialCount + materialBase(i)) to meshPart.
  uint16_t materialBase(int part) const { return materialBase_[size_t(part)]; }
  uint16_t partMaterialBase(int part) const { return materialBase(part); } // alias
  uint16_t paletteOffset(int part, uint8_t variant) const;
  std::vector<uint32_t> materialColors() const;       // RGBA8
  std::vector<uint8_t> materialEmissive() const;      // 1 = glows
  uint32_t materialCount() const { return materialCount_; }

private:
  std::vector<VoxelPart> parts_;
  std::vector<EquipPiece> pieces_{EquipPiece{}}; // index 0 = kNoPiece
  std::array<std::array<int16_t, kBoneCount>, 4> body_{};
  std::vector<int16_t> hair_;
  std::vector<CreatureModel> creatures_;
  std::vector<uint16_t> materialBase_;
  uint32_t materialCount_ = 0;
};

// Meshes a part (<= 32^3) with the chunk mesher; palette indices become
// material ids offset by `materialBase`.
void meshPart(const VoxelPart &part, uint16_t materialBase, voxel::ChunkMeshData &out);

// Resolves which part to draw for each bone and socket this frame.
struct ResolvedModel {
  std::array<int16_t, kBoneCount> boneParts;       // -1 = nothing
  std::array<int16_t, size_t(Socket::Count)> socketParts;
  // ModelInstance::paletteOffset per bone / socket (skin tone, hair colour).
  std::array<uint16_t, kBoneCount> bonePaletteOffset{};
  std::array<uint16_t, size_t(Socket::Count)> socketPaletteOffset{};
};
ResolvedModel resolveAppearance(const ModelLibrary &lib, const Appearance &a);

} // namespace atm::model
