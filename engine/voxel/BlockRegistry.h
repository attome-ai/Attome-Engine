#pragma once

// Data-driven block types. The same table is used by the server (hardness,
// drops, collision), the mesher (opacity, translucency), light propagation
// (emission) and the renderer (colour, emissive glow).

#include "VoxelTypes.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace atm {
class Json;
}

namespace atm::voxel {

// Colour packing used by BlockDef colours: bytes in memory are R, G, B, A
// (VK_FORMAT_R8G8B8A8_UNORM / GLSL unpackUnorm4x8 order), i.e. the uint32
// value is 0xAABBGGRR.
inline constexpr uint32_t packRGBA(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
  return uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

enum class BlockRender : uint8_t {
  None,        // air: no faces
  Opaque,      // solid cube; hides neighbour faces
  Translucent, // water, glass: drawn in the translucent pass
  Cutout,      // leaves: drawn opaque but doesn't hide neighbour faces
};

struct BlockDef {
  std::string name;              // "stone", "oak_log", ...
  BlockRender render = BlockRender::Opaque;
  bool solid = true;             // collides with entities
  bool liquid = false;
  uint32_t colorTop = 0xFFFFFFFF;    // RGBA8 (Trove style: flat colours)
  uint32_t colorSide = 0xFFFFFFFF;
  uint32_t colorBottom = 0xFFFFFFFF;
  uint8_t emission = 0;          // block light 0..15 (lava, crystals, lamps)
  float hardness = 1.0f;         // seconds to break with bare hands
  uint8_t toolClass = 0;         // 0 none, 1 pickaxe, 2 axe, 3 shovel
  uint16_t dropItem = 0;         // item id given when broken (0 = none)
  uint8_t miningLevel = 1;       // Mining level required
  float miningXp = 0.0f;         // XP granted on break
};

class BlockRegistry {
public:
  // Starts with just air (id 0), so get() is always valid.
  BlockRegistry();
  // Registers the built-in demo blocks (air is always id 0).
  void registerDefaults();
  // Optional JSON overrides/additions ({"blocks": {"stone": {...}}}).
  bool loadJson(const Json &root, std::string *error = nullptr);

  BlockId add(BlockDef def);
  const BlockDef &get(BlockId id) const { return defs_[id < defs_.size() ? id : 0]; }
  BlockId find(std::string_view name) const; // kAir if unknown
  size_t size() const { return defs_.size(); }

  // Fast per-id flags (one byte each). Bounds-checked: ids beyond the table
  // (voxel-model palettes, corrupt network data) behave like opaque, solid,
  // non-translucent blocks — the same rule the mesher and lighting use.
  bool opaque(BlockId id) const { return id < opaque_.size() ? opaque_[id] != 0 : true; }
  bool meshed(BlockId id) const { return id < meshed_.size() ? meshed_[id] != 0 : true; }
  bool translucent(BlockId id) const { return id < translucent_.size() && translucent_[id] != 0; }
  bool solid(BlockId id) const { return id < solid_.size() ? solid_[id] != 0 : true; }
  // Additive helpers (bounds-checked; unknown ids behave like opaque
  // non-emissive blocks, so voxel-model palettes mesh without entries).
  bool known(BlockId id) const { return id < defs_.size(); }
  BlockRender renderMode(BlockId id) const {
    return id < defs_.size() ? defs_[id].render : BlockRender::Opaque;
  }
  uint8_t emission(BlockId id) const { return id < emission_.size() ? emission_[id] : 0; }
  // Light passes through (air, water, glass, leaves): anything not Opaque.
  bool transmitsLight(BlockId id) const {
    return id < opaque_.size() ? !opaque_[id] : false;
  }

private:
  void rebuildFlags();
  std::vector<BlockDef> defs_;
  std::vector<uint8_t> opaque_, meshed_, translucent_, solid_, emission_;
};

// Names of the built-in blocks, so gameplay code can refer to them without
// string lookups at runtime (ids assigned in registerDefaults order).
namespace blocks {
inline constexpr BlockId Air = 0, Stone = 1, Dirt = 2, Grass = 3, Sand = 4,
                         Water = 5, OakLog = 6, OakLeaves = 7, CopperOre = 8,
                         IronOre = 9, GoldOre = 10, Crystal = 11, Planks = 12,
                         Brick = 13, Glass = 14, Lamp = 15, Snow = 16,
                         Bedrock = 17,
                         // town / structures
                         Cobblestone = 18, StoneBricks = 19, RoofRed = 20, RoofBlue = 21,
                         Plaster = 22, DarkPlanks = 23, Path = 24,
                         // depleted resource nodes (regrown by the server's spawners)
                         Stump = 25, DepletedRock = 26,
                         // town building set (Trove-style hub)
                         WhiteStone = 27, WhiteStoneTrim = 28, RoofBlueDark = 29, GoldTrim = 30,
                         BannerBlue = 31, BannerGold = 32, AwningRed = 33, AwningWhite = 34,
                         PaperLantern = 35, GlowCrystal = 36, PlazaTile = 37, PlazaTileDark = 38,
                         HedgeLeaves = 39, Crate = 40, FlowersRed = 41, FlowersYellow = 42,
                         MossyStone = 43, LightPlanks = 44, RoofBlueLight = 45, WindowLit = 46,
                         TownLog = 47,
                         // invisible: collision of fine-voxel structures / light of their lamps
                         Barrier = 48, Light = 49,
                         // tree foliage variants (all sway; all fall with a felled tree)
                         OakLeavesDark = 50, OakLeavesLight = 51, PineLeaves = 52;
inline constexpr BlockId Count = 53;
// Any tree foliage block (felling, gathering rules).
inline constexpr bool isLeaves(BlockId id) {
  return id == OakLeaves || id == OakLeavesDark || id == OakLeavesLight || id == PineLeaves;
}
} // namespace blocks

} // namespace atm::voxel
