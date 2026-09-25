#pragma once

// A streamed voxel world: chunk storage + background generation, lighting
// and meshing on worker threads. Used by the client (with meshing) and the
// server (without meshing).
//
// Threading model:
//   - All public methods are main-thread only unless marked otherwise.
//   - Workers never touch the chunk map directly: the main thread copies a
//     job's inputs (MeshInput) and receives results through a queue.
//   - Chunk data is immutable while a worker reads it (copy-on-write via
//     shared_ptr<const Chunk>); edits replace the pointer on the main thread.

#include "BlockRegistry.h"
#include "Chunk.h"
#include "MeshTypes.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace atm::voxel {

struct VoxelWorldConfig {
  uint64_t seed = 1;
  bool meshing = true;           // client: true, server: false
  int workerThreads = 0;         // 0 = hardware_concurrency - 2 (min 1)
  int viewRadiusChunks = 12;     // horizontal load radius around the focus
  int unloadMarginChunks = 2;    // hysteresis before unloading
  int verticalChunksBelow = 3;   // loaded below/above the focus chunk
  int verticalChunksAbove = 4;
  int maxJobsInFlight = 256;
  int maxResultsPerUpdate = 64;  // bounds main-thread work per frame
};

struct ChunkMeshResult {
  ChunkCoord coord;
  ChunkMeshData mesh;
  uint32_t revision = 0;         // chunk revision the mesh was built from
};

class VoxelWorld : public IBlockAccess {
public:
  VoxelWorld(const VoxelWorldConfig &config, const BlockRegistry &blocks);
  ~VoxelWorld() override;
  VoxelWorld(const VoxelWorld &) = delete;
  VoxelWorld &operator=(const VoxelWorld &) = delete;

  // Chunks within the view radius of `focus` (world blocks) are generated and
  // (client) meshed; chunks beyond radius + margin are unloaded. Multiple
  // foci (server: one per player) are supported by calling addFocus each
  // update before update().
  void clearFoci();
  void addFocus(double x, double y, double z);

  // Pumps jobs and results; call once per frame/tick.
  void update();

  // Completed meshes since the last call (client). Also reports chunks that
  // were unloaded (mesh empty, coord set) so the renderer can drop them.
  void takeMeshResults(std::vector<ChunkMeshResult> &out);
  void takeUnloaded(std::vector<ChunkCoord> &out);

  // IBlockAccess (unloaded chunks: blockAt returns kAir; physics checks isLoaded)
  BlockId blockAt(BlockPos p) const override;
  bool isLoaded(ChunkCoord c) const override;

  // Edits (main thread). Marks the chunk (and neighbours when on a border)
  // for re-light + re-mesh; player edits are prioritised over streaming.
  bool setBlock(BlockPos p, BlockId id);

  // Replaces a whole chunk (server-sent edited chunk on the client, or
  // loaded from disk on the server). Takes precedence over generation.
  void setChunkData(ChunkCoord c, std::shared_ptr<const Chunk> chunk);
  std::shared_ptr<const Chunk> chunk(ChunkCoord c) const;

  // Server: keys of chunks that differ from the generated world.
  const std::vector<uint64_t> &editedChunkKeys() const;
  bool isEdited(ChunkCoord c) const;

  // Client: chunks the server said are edited — don't generate them locally,
  // wait for setChunkData (they appear as not loaded until then).
  void setExpectedEdited(std::span<const uint64_t> keys);

  const BlockRegistry &blocks() const { return blocks_; }
  const WorldGenerator &generator() const;

  struct Stats {
    size_t loadedChunks = 0, pendingJobs = 0, meshedLastUpdate = 0;
    size_t memoryBytes = 0;
    double avgMeshMicros = 0, avgGenMicros = 0;
  };
  Stats stats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  const BlockRegistry &blocks_;
};

} // namespace atm::voxel
