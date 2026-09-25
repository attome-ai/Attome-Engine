# =============================================================================
# AttomeVoxel — voxel world: palette chunks, block registry, binary greedy
# mesher, flood-fill lighting, procedural terrain, streamed VoxelWorld.
# =============================================================================
add_library(AttomeVoxel STATIC
  BlockRegistry.cpp
  Chunk.cpp
  ChunkMesher.cpp
  Lighting.cpp
  WorldGenerator.cpp
  VoxelWorld.cpp
)

# `#include "voxel/Chunk.h"` (engine/ on the path) and `#include "Chunk.h"`.
target_include_directories(AttomeVoxel PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(AttomeVoxel PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
find_package(glm CONFIG REQUIRED)
target_link_libraries(AttomeVoxel PUBLIC AttomeCore Threads::Threads glm::glm)

if(MSVC)
  # /fp:precise (default) never contracts a*b+c into FMA.
  target_compile_options(AttomeVoxel PRIVATE /W4 /permissive- /fp:precise)
else()
  # World generation must be bit-identical on client and server: no FMA
  # contraction (ARM and some x86 builds would otherwise fuse multiply-adds).
  target_compile_options(AttomeVoxel PRIVATE -Wall -Wextra -ffp-contract=off)
endif()
