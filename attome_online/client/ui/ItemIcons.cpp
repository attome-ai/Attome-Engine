#include "ui/ItemIcons.h"

#include "../../../engine/ATMFrameProfiler.h"
#include "../../../engine/model/IconRender.h"

#include <imgui_internal.h>

#include <SDL3/SDL.h>

#include <chrono>
#include <cstring>
#include <string>

namespace ao::client::ui {

int itemModelPart(const atm::model::ModelLibrary &lib, ItemId item) {
  const ItemDef &d = itemDef(item);
  if (d.name.empty()) return -1;
  if (d.kind == ItemKind::Prop && d.prop) {
    const int p = lib.findPart(d.prop);
    if (p >= 0) return p;
  }
  if (const int p = lib.findPart("item_" + std::string(d.name)); p >= 0) return p;
  if (d.piece) {
    const atm::model::PieceId pid = lib.findPiece(d.piece);
    if (pid != atm::model::kNoPiece) {
      const atm::model::EquipPiece &piece = lib.piece(pid);
      if (piece.socketPart >= 0) return piece.socketPart;
      for (int16_t p : piece.boneParts)
        if (p >= 0) return p;
    }
  }
  return -1;
}

void ItemIcons::build(const atm::model::ModelLibrary &lib, const atm::voxel::BlockRegistry &blocks) {
  const auto t0 = std::chrono::steady_clock::now();
  const int W = kCell * kCols;
  if (!tex_) {
    tex_ = IM_NEW(ImTextureData)();
    tex_->Create(ImTextureFormat_RGBA32, W, W);
    tex_->UseColors = true;
  }
  have_.assign(itemCount(), 0);
  std::vector<uint32_t> px;
  const size_t n = std::min<size_t>(itemCount(), size_t(kCols * kCols));
  for (size_t id = 1; id < n; ++id) {
    const ItemDef &d = itemDef(ItemId(id));
    if (d.name.empty()) continue;
    const int part = itemModelPart(lib, ItemId(id));
    if (part >= 0) {
      atm::model::renderVoxelIcon(lib.parts()[size_t(part)], kCell, px);
    } else if (d.kind == ItemKind::Block) {
      // A small cube in the block's colours.
      const auto &b = blocks.get(d.placesBlock);
      atm::model::VoxelPart cube;
      cube.sx = cube.sy = cube.sz = 8;
      cube.palette = {0u, b.colorTop | 0xFF000000u, b.colorSide | 0xFF000000u};
      cube.voxels.assign(512, 2);
      for (int z = 0; z < 8; ++z)
        for (int x = 0; x < 8; ++x) cube.at(x, 7, z) = 1;
      if (b.emission > 0) cube.emissiveFrom = 1;
      atm::model::renderVoxelIcon(cube, kCell, px);
    } else {
      continue; // no model: the vector icon is used
    }
    const int cx = int(id % kCols) * kCell, cy = int(id / kCols) * kCell;
    for (int y = 0; y < kCell; ++y)
      std::memcpy(tex_->Pixels + (size_t(cy + y) * W + cx) * 4, &px[size_t(y) * kCell], size_t(kCell) * 4);
    have_[id] = 1;
  }
  if (tex_->RefCount == 0) ImGui::RegisterUserTexture(tex_);
  const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
  SDL_Log("[client] item icons rendered from models: %zu items in %.0f ms", n, double(ms));
}

bool ItemIcons::draw(ImDrawList *dl, ImVec2 c, float size, ItemId item) const {
  if (!tex_ || item >= have_.size() || !have_[item]) return false;
  const float W = float(kCell * kCols);
  const float u0 = float(int(item % kCols) * kCell) / W, v0 = float(int(item / kCols) * kCell) / W;
  const float du = float(kCell) / W;
  const float h = size * 0.5f;
  dl->AddImage(tex_->GetTexRef(), ImVec2(c.x - h, c.y - h), ImVec2(c.x + h, c.y + h), ImVec2(u0, v0),
               ImVec2(u0 + du, v0 + du));
  return true;
}

ItemIcons &itemIcons() {
  static ItemIcons icons;
  return icons;
}

} // namespace ao::client::ui
