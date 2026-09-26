#pragma once

// Item icons rendered from voxel models on the CPU at startup (the way
// RuneScape draws inventory sprites from item models instead of shipping
// images): orthographic ray casting, 2x supersampling, face shading, a dark
// outline and a soft drop shadow. No image files, nothing on disk.

#include "Character.h"

#include <cstdint>
#include <vector>

namespace atm::model {

// Renders `part` into size x size RGBA8 pixels (R in the low byte), fitted
// and centred. Long thin items (swords, tools, bows) are laid diagonally.
void renderVoxelIcon(const VoxelPart &part, int size, std::vector<uint32_t> &out);

} // namespace atm::model
