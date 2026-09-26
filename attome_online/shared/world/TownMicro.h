#pragma once

// Fine-voxel (Micro.h) building generators for the home town. Pure,
// deterministic functions of their parameters: the models are rebuilt from
// code on every machine, so they cost nothing on disk or on the network.

#include "Micro.h"

namespace ao::world::townmicro {

struct HouseSpec {
  int x0, z0, x1, z1; // wall footprint in town-relative blocks (inclusive)
  int floors;         // 1 or 2
  bool redRoof;
  uint32_t seed;
  int doorDX, doorDZ; // outward direction of the door (one of them +-1)
};

MicroModel house(const HouseSpec &h);
MicroModel plazaFloor();                      // square + ring street, 1 block thick at ground level
MicroModel fountain();
MicroModel castle(int terraceY);              // terrace, stairs, walls, keep and towers
MicroModel bannerPole(int bx, int bz, int dirX, int dirZ);
MicroModel marketStall(int x0, int side, uint32_t seed);
MicroModel wallSide(int side, int wall);      // 0 north, 1 south, 2 west, 3 east
MicroModel gatehouse(int cx, int cz, int wallHeight);
MicroModel cornerTower(int cx, int cz);

} // namespace ao::world::townmicro
