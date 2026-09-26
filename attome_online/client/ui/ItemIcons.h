#pragma once

// Inventory icons rendered from the items' own voxel models at startup
// (engine/model/IconRender.h), packed into one ImGui texture. RuneScape-style:
// no icon images are shipped; every icon comes from the 3D model.

#include "shared/GameTypes.h"

#include "../../../engine/model/Character.h"
#include "../../../engine/voxel/BlockRegistry.h"

#include <imgui.h>

#include <vector>

struct ImTextureData;

namespace ao::client::ui {

// The voxel part that shows an item (prop, "item_<name>", equipment piece),
// or -1 (blocks are drawn as a cube of their colours).
int itemModelPart(const atm::model::ModelLibrary &lib, ItemId item);

class ItemIcons {
public:
  static constexpr int kCell = 64;  // icon size in the atlas (pixels)
  static constexpr int kCols = 16;  // atlas = 16 x 16 icons

  // Renders every item's icon and registers the atlas with ImGui. Call after
  // the ImGui backend is initialised.
  void build(const atm::model::ModelLibrary &lib, const atm::voxel::BlockRegistry &blocks);
  // Draws the icon centred at `c`, `size` pixels; false if there is none.
  bool draw(ImDrawList *dl, ImVec2 c, float size, ItemId item) const;

private:
  ImTextureData *tex_ = nullptr;
  std::vector<uint8_t> have_;
};

ItemIcons &itemIcons();

} // namespace ao::client::ui
