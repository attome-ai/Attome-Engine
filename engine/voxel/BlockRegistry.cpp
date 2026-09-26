#include "BlockRegistry.h"

#include "ATMJson.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace atm::voxel {

namespace {

struct DefaultBlock {
  const char *name;
  BlockRender render;
  bool solid, liquid;
  uint32_t top, side, bottom;
  uint8_t emission;
  float hardness;
  uint8_t toolClass; // 0 none, 1 pickaxe, 2 axe, 3 shovel
  uint8_t miningLevel;
  float miningXp;
};

constexpr uint32_t C(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) { return packRGBA(r, g, b, a); }

// Order MUST match namespace blocks:: in BlockRegistry.h.
const DefaultBlock kDefaults[blocks::Count] = {
    {"air", BlockRender::None, false, false, 0, 0, 0, 0, 0.0f, 0, 1, 0.0f},
    {"stone", BlockRender::Opaque, true, false, C(140, 146, 158), C(128, 134, 146), C(112, 118, 130), 0, 1.5f, 1, 1, 1.0f},
    {"dirt", BlockRender::Opaque, true, false, C(150, 100, 62), C(140, 92, 56), C(124, 80, 48), 0, 0.5f, 3, 1, 0.0f},
    {"grass", BlockRender::Opaque, true, false, C(106, 176, 64), C(140, 110, 66), C(140, 92, 56), 0, 0.6f, 3, 1, 0.0f},
    {"sand", BlockRender::Opaque, true, false, C(234, 200, 118), C(220, 184, 104), C(204, 168, 94), 0, 0.5f, 3, 1, 0.0f},
    {"water", BlockRender::Translucent, false, true, C(56, 140, 236, 160), C(48, 128, 224, 160), C(40, 112, 208, 160), 0, 100.0f, 0, 1, 0.0f},
    {"oak_log", BlockRender::Opaque, true, false, C(190, 150, 96), C(116, 82, 50), C(190, 150, 96), 0, 2.0f, 2, 1, 0.0f},
    {"oak_leaves", BlockRender::Cutout, true, false, C(72, 180, 64), C(64, 168, 58), C(56, 150, 52), 0, 0.2f, 0, 1, 0.0f},
    {"copper_ore", BlockRender::Opaque, true, false, C(214, 128, 76), C(200, 118, 70), C(186, 108, 64), 0, 2.5f, 1, 1, 5.0f},
    {"iron_ore", BlockRender::Opaque, true, false, C(206, 184, 168), C(194, 172, 156), C(180, 160, 144), 0, 3.0f, 1, 15, 10.0f},
    {"gold_ore", BlockRender::Opaque, true, false, C(255, 214, 64), C(246, 202, 56), C(232, 188, 48), 0, 3.5f, 1, 40, 20.0f},
    {"crystal", BlockRender::Opaque, true, false, C(180, 110, 255), C(164, 96, 246), C(150, 84, 232), 12, 4.0f, 1, 60, 40.0f},
    {"planks", BlockRender::Opaque, true, false, C(214, 170, 108), C(204, 160, 100), C(190, 148, 92), 0, 1.5f, 2, 1, 0.0f},
    {"brick", BlockRender::Opaque, true, false, C(196, 84, 70), C(184, 76, 64), C(170, 68, 58), 0, 2.0f, 1, 1, 0.0f},
    {"glass", BlockRender::Translucent, true, false, C(200, 236, 255, 90), C(200, 236, 255, 90), C(200, 236, 255, 90), 0, 0.3f, 0, 1, 0.0f},
    {"lamp", BlockRender::Opaque, true, false, C(255, 236, 160), C(255, 228, 140), C(240, 210, 120), 15, 0.3f, 0, 1, 0.0f},
    {"snow", BlockRender::Opaque, true, false, C(248, 252, 255), C(236, 242, 250), C(224, 232, 242), 0, 0.3f, 3, 1, 0.0f},
    {"bedrock", BlockRender::Opaque, true, false, C(52, 52, 60), C(48, 48, 56), C(44, 44, 52), 0, -1.0f, 0, 255, 0.0f},
    // Town / structures (not player-breakable in the overworld).
    {"cobblestone", BlockRender::Opaque, true, false, C(150, 150, 146), C(136, 136, 132), C(120, 120, 118), 0, 2.0f, 1, 1, 0.0f},
    {"stone_bricks", BlockRender::Opaque, true, false, C(176, 172, 164), C(160, 156, 148), C(140, 136, 130), 0, 2.0f, 1, 1, 0.0f},
    {"roof_red", BlockRender::Opaque, true, false, C(186, 70, 56), C(166, 58, 48), C(140, 50, 42), 0, 1.5f, 0, 1, 0.0f},
    {"roof_blue", BlockRender::Opaque, true, false, C(70, 110, 180), C(58, 94, 160), C(48, 78, 136), 0, 1.5f, 0, 1, 0.0f},
    {"plaster", BlockRender::Opaque, true, false, C(240, 232, 212), C(234, 224, 200), C(210, 200, 180), 0, 1.5f, 0, 1, 0.0f},
    {"dark_planks", BlockRender::Opaque, true, false, C(118, 80, 48), C(104, 70, 42), C(90, 60, 36), 0, 1.5f, 2, 1, 0.0f},
    {"path", BlockRender::Opaque, true, false, C(186, 160, 116), C(150, 112, 72), C(140, 100, 62), 0, 0.6f, 3, 1, 0.0f},
    // Depleted resource nodes: a felled tree's stump, a mined-out rock.
    {"stump", BlockRender::Opaque, true, false, C(160, 124, 80), C(110, 78, 46), C(110, 78, 46), 0, -1.0f, 0, 255, 0.0f},
    {"depleted_rock", BlockRender::Opaque, true, false, C(112, 112, 118), C(100, 100, 106), C(90, 90, 96), 0, -1.0f, 0, 255, 0.0f},
    // Town building set (Trove-style hub; placeable later in homes / clan plots).
    {"white_stone", BlockRender::Opaque, true, false, C(240, 234, 220), C(228, 221, 204), C(200, 194, 180), 0, 2.0f, 1, 1, 0.0f},
    {"white_stone_trim", BlockRender::Opaque, true, false, C(198, 194, 184), C(184, 180, 170), C(160, 156, 148), 0, 2.0f, 1, 1, 0.0f},
    {"roof_blue_dark", BlockRender::Opaque, true, false, C(46, 78, 160), C(38, 64, 138), C(30, 52, 116), 0, 1.5f, 0, 1, 0.0f},
    {"gold_trim", BlockRender::Opaque, true, false, C(252, 204, 72), C(236, 184, 56), C(210, 160, 46), 2, 2.0f, 1, 1, 0.0f},
    {"banner_blue", BlockRender::Opaque, true, false, C(44, 76, 190), C(40, 70, 178), C(34, 60, 156), 0, 0.5f, 0, 1, 0.0f},
    {"banner_gold", BlockRender::Opaque, true, false, C(250, 200, 60), C(244, 192, 54), C(220, 170, 46), 0, 0.5f, 0, 1, 0.0f},
    {"awning_red", BlockRender::Opaque, true, false, C(222, 60, 56), C(206, 52, 50), C(180, 44, 42), 0, 0.5f, 0, 1, 0.0f},
    {"awning_white", BlockRender::Opaque, true, false, C(246, 242, 232), C(236, 230, 218), C(212, 206, 194), 0, 0.5f, 0, 1, 0.0f},
    {"paper_lantern", BlockRender::Opaque, true, false, C(255, 214, 120), C(255, 200, 100), C(240, 180, 90), 14, 0.5f, 0, 1, 0.0f},
    {"glow_crystal", BlockRender::Opaque, true, false, C(120, 214, 255), C(90, 190, 255), C(70, 160, 240), 13, 2.0f, 1, 1, 0.0f},
    {"plaza_tile", BlockRender::Opaque, true, false, C(222, 216, 202), C(206, 200, 186), C(186, 180, 168), 0, 2.0f, 1, 1, 0.0f},
    {"plaza_tile_dark", BlockRender::Opaque, true, false, C(176, 170, 160), C(160, 154, 146), C(140, 136, 128), 0, 2.0f, 1, 1, 0.0f},
    {"hedge_leaves", BlockRender::Cutout, true, false, C(96, 200, 74), C(84, 184, 66), C(70, 160, 58), 0, 0.3f, 0, 1, 0.0f},
    {"crate", BlockRender::Opaque, true, false, C(196, 146, 84), C(170, 122, 68), C(150, 106, 58), 0, 1.0f, 2, 1, 0.0f},
    {"flowers_red", BlockRender::Cutout, true, false, C(236, 76, 96), C(96, 190, 72), C(80, 160, 60), 0, 0.2f, 0, 1, 0.0f},
    {"flowers_yellow", BlockRender::Cutout, true, false, C(252, 216, 72), C(96, 190, 72), C(80, 160, 60), 0, 0.2f, 0, 1, 0.0f},
    {"mossy_stone", BlockRender::Opaque, true, false, C(122, 158, 104), C(140, 146, 140), C(118, 122, 118), 0, 2.0f, 1, 1, 0.0f},
    {"light_planks", BlockRender::Opaque, true, false, C(232, 196, 136), C(222, 184, 124), C(204, 166, 110), 0, 1.5f, 2, 1, 0.0f},
    {"roof_blue_light", BlockRender::Opaque, true, false, C(92, 140, 220), C(80, 124, 204), C(66, 104, 180), 0, 1.5f, 0, 1, 0.0f},
    {"window_lit", BlockRender::Opaque, true, false, C(255, 222, 140), C(255, 214, 124), C(240, 196, 110), 9, 0.5f, 0, 1, 0.0f},
    {"town_log", BlockRender::Opaque, true, false, C(190, 150, 96), C(122, 88, 54), C(190, 150, 96), 0, 2.0f, 2, 1, 0.0f},
    // Invisible collision of a fine-voxel structure (drawn as a model), and an
    // invisible light source for its lamps and windows.
    {"barrier", BlockRender::None, true, false, C(214, 208, 196), C(214, 208, 196), C(214, 208, 196), 0, -1.0f, 0, 255, 0.0f},
    {"light", BlockRender::None, false, false, 0, 0, 0, 14, -1.0f, 0, 255, 0.0f},
    {"oak_leaves_dark", BlockRender::Cutout, true, false, C(52, 146, 52), C(46, 134, 48), C(40, 118, 42), 0, 0.2f, 0, 1, 0.0f},
    {"oak_leaves_light", BlockRender::Cutout, true, false, C(118, 206, 74), C(104, 192, 66), C(90, 172, 58), 0, 0.2f, 0, 1, 0.0f},
    {"pine_leaves", BlockRender::Cutout, true, false, C(40, 118, 76), C(34, 104, 68), C(28, 90, 60), 0, 0.2f, 0, 1, 0.0f},
};

bool parseColor(const Json &v, uint32_t &out) {
  if (v.isNumber()) {
    const double d = v.asNumber();
    if (d < 0.0 || d > 4294967295.0)
      return false;
    out = uint32_t(d);
    return true;
  }
  if (!v.isString())
    return false;
  // "#RRGGBB" or "#RRGGBBAA"
  const std::string &s = v.asString();
  size_t start = (!s.empty() && s[0] == '#') ? 1 : 0;
  const size_t n = s.size() - start;
  if (n != 6 && n != 8)
    return false;
  uint8_t bytes[4] = {0, 0, 0, 255};
  for (size_t i = 0; i < n / 2; ++i) {
    int value = 0;
    for (int k = 0; k < 2; ++k) {
      const char ch = s[start + i * 2 + size_t(k)];
      int d;
      if (ch >= '0' && ch <= '9')
        d = ch - '0';
      else if (ch >= 'a' && ch <= 'f')
        d = ch - 'a' + 10;
      else if (ch >= 'A' && ch <= 'F')
        d = ch - 'A' + 10;
      else
        return false;
      value = value * 16 + d;
    }
    bytes[i] = uint8_t(value);
  }
  out = packRGBA(bytes[0], bytes[1], bytes[2], bytes[3]);
  return true;
}

template <typename T> void readInt(const Json &obj, const char *key, T &out, double lo, double hi) {
  if (const Json *v = obj.find(key); v && v->isNumber()) {
    const double d = std::clamp(v->asNumber(), lo, hi);
    out = T(d);
  }
}

} // namespace

BlockRegistry::BlockRegistry() { rebuildFlags(); }

void BlockRegistry::registerDefaults() {
  defs_.clear();
  defs_.reserve(blocks::Count);
  for (const DefaultBlock &d : kDefaults) {
    BlockDef def;
    def.name = d.name;
    def.render = d.render;
    def.solid = d.solid;
    def.liquid = d.liquid;
    def.colorTop = d.top;
    def.colorSide = d.side;
    def.colorBottom = d.bottom;
    def.emission = d.emission;
    def.hardness = d.hardness;
    def.toolClass = d.toolClass;
    def.dropItem = 0; // gameplay maps drops
    def.miningLevel = d.miningLevel;
    def.miningXp = d.miningXp;
    defs_.push_back(std::move(def));
  }
  rebuildFlags();
}

bool BlockRegistry::loadJson(const Json &root, std::string *error) {
  const Json *list = root.find("blocks");
  if (!list || !list->isObject()) {
    if (error)
      *error = "missing \"blocks\" object";
    return false;
  }
  if (defs_.empty()) {
    BlockDef air;
    air.name = "air";
    air.render = BlockRender::None;
    air.solid = false;
    defs_.push_back(std::move(air));
  }
  for (const auto &[name, obj] : list->asObject()) {
    if (!obj.isObject()) {
      if (error)
        *error = "block \"" + name + "\" is not an object";
      return false;
    }
    BlockId id = find(name);
    const bool isNew = (id == kAir && name != defs_[0].name);
    if (isNew && defs_.size() >= 65535) {
      if (error)
        *error = "too many blocks";
      return false;
    }
    BlockDef def = isNew ? BlockDef{} : defs_[id];
    def.name = name;
    if (const Json *v = obj.find("render"); v && v->isString()) {
      const std::string &r = v->asString();
      if (r == "none")
        def.render = BlockRender::None;
      else if (r == "opaque")
        def.render = BlockRender::Opaque;
      else if (r == "translucent")
        def.render = BlockRender::Translucent;
      else if (r == "cutout")
        def.render = BlockRender::Cutout;
      else {
        if (error)
          *error = "block \"" + name + "\": unknown render mode \"" + r + "\"";
        return false;
      }
    }
    if (const Json *v = obj.find("solid"); v && v->isBool())
      def.solid = v->asBool();
    if (const Json *v = obj.find("liquid"); v && v->isBool())
      def.liquid = v->asBool();
    if (const Json *v = obj.find("color")) { // shorthand: all faces
      uint32_t c;
      if (parseColor(*v, c))
        def.colorTop = def.colorSide = def.colorBottom = c;
    }
    const std::pair<const char *, uint32_t *> colors[] = {
        {"colorTop", &def.colorTop}, {"colorSide", &def.colorSide}, {"colorBottom", &def.colorBottom}};
    for (const auto &[key, dst] : colors) {
      if (const Json *v = obj.find(key)) {
        if (!parseColor(*v, *dst)) {
          if (error)
            *error = "block \"" + name + "\": bad " + key;
          return false;
        }
      }
    }
    readInt(obj, "emission", def.emission, 0.0, 15.0);
    if (const Json *v = obj.find("hardness"); v && v->isNumber())
      def.hardness = float(v->asNumber());
    readInt(obj, "toolClass", def.toolClass, 0.0, 255.0);
    readInt(obj, "dropItem", def.dropItem, 0.0, 65535.0);
    readInt(obj, "miningLevel", def.miningLevel, 0.0, 255.0);
    if (const Json *v = obj.find("miningXp"); v && v->isNumber())
      def.miningXp = float(v->asNumber());
    if (isNew)
      defs_.push_back(std::move(def));
    else
      defs_[id] = std::move(def);
  }
  rebuildFlags();
  return true;
}

BlockId BlockRegistry::add(BlockDef def) {
  if (defs_.size() >= 65535)
    return kAir;
  defs_.push_back(std::move(def));
  rebuildFlags();
  return BlockId(defs_.size() - 1);
}

BlockId BlockRegistry::find(std::string_view name) const {
  for (size_t i = 0; i < defs_.size(); ++i)
    if (defs_[i].name == name)
      return BlockId(i);
  return kAir;
}

void BlockRegistry::rebuildFlags() {
  if (defs_.empty()) {
    BlockDef air;
    air.name = "air";
    air.render = BlockRender::None;
    air.solid = false;
    defs_.push_back(std::move(air));
  }
  const size_t n = defs_.size();
  opaque_.assign(n, 0);
  meshed_.assign(n, 0);
  translucent_.assign(n, 0);
  solid_.assign(n, 0);
  emission_.assign(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const BlockDef &d = defs_[i];
    opaque_[i] = d.render == BlockRender::Opaque;
    meshed_[i] = d.render != BlockRender::None;
    translucent_[i] = d.render == BlockRender::Translucent;
    solid_[i] = d.solid;
    emission_[i] = uint8_t(std::min<int>(d.emission, 15));
  }
}

} // namespace atm::voxel
