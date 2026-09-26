#pragma once

// The home town (Brightwater), stamped into the procedural world at
// generation time (VoxelWorldConfig::postGenerate) identically on the server
// and on every client: a Trove-style hub with a castle on a terrace, a
// crystal fountain square, timber-framed houses with blue roofs, a market,
// banners, lamps, gardens and a town wall.
//
// The whole town is built once per seed into a block template (TownGrid),
// deterministically, then copied into chunks as they generate: no network
// traffic, nothing lost when chunks unload. Decorations smaller than a block
// (lanterns, barrels, benches, ...) are listed as props for the client.

#include "../../../engine/voxel/VoxelWorld.h"
#include "Micro.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ao::world {

// A decoration prop (voxel model part "prop_<name>", 16 voxels per block).
// Position in blocks relative to the town centre; y = 1 stands on the ground.
struct TownProp {
  const char *name;
  float x, y, z;
  float yaw; // radians, 0 = front faces -Z
};

struct TownGrid; // block template (Town.cpp)

struct TownLayout {
  int32_t centerX = 0, centerZ = 0; // the fountain
  int groundY = 70;                 // top of the flattened ground
  int flatRadius = 64;              // fully flat within this square radius
  int blendRadius = 80;             // blends back to natural terrain by here
  std::shared_ptr<const TownGrid> grid;
};

// Layout for a world seed (ground height sampled from the generator).
TownLayout townLayout(const atm::voxel::WorldGenerator &gen);

// Post-generation hook: rewrites the part of `chunk` covered by the town.
void stampTown(const TownLayout &town, const atm::voxel::WorldGenerator &gen, atm::voxel::ChunkCoord c,
               atm::voxel::Chunk &chunk);

// Height of the highest town block (or the terrain) at a column: map,
// spawning, AI.
int townSurfaceHeight(const TownLayout &town, const atm::voxel::WorldGenerator &gen, int32_t x, int32_t z);

// True inside the town (monster-free safe zone).
bool inTown(const TownLayout &town, double x, double z);

// Registers stampTown as the world's postGenerate hook (call on the server
// and client before constructing the VoxelWorld).
void installHomeTown(atm::voxel::VoxelWorldConfig &wc);

// Ground height of the finished world (natural terrain + town).
// Use this instead of WorldGenerator::surfaceHeight in gameplay code.
int groundHeight(const atm::voxel::WorldGenerator &gen, int32_t x, int32_t z);
// The home town of a world (built once, cached per seed).
const TownLayout &homeTown(const atm::voxel::WorldGenerator &gen);
// Decoration props of the town (positions relative to the town centre).
const std::vector<TownProp> &townProps(const TownLayout &town);
// Fine-voxel structures of the town (buildings, walls, fountain, square).
const std::vector<MicroModel> &townStructures(const TownLayout &town);
// World-map colour of a town column (top of the structures); false outside.
bool townMapColor(const TownLayout &town, int32_t x, int32_t z, uint32_t &rgba);

// Where players appear: on the square, south of the fountain.
void townSpawnPoint(const TownLayout &town, double &x, double &y, double &z);

} // namespace ao::world
