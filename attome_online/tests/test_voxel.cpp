// Unit tests for the voxel world module (engine/voxel).

#include "atm_test.h"

#include "voxel/BlockRegistry.h"
#include "voxel/Chunk.h"
#include "voxel/Lighting.h"
#include "voxel/MeshTypes.h"
#include "voxel/VoxelWorld.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

using namespace atm::voxel;

namespace {

uint32_t lcg(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

BlockRegistry &registry() {
  static BlockRegistry r = [] {
    BlockRegistry b;
    b.registerDefaults();
    return b;
  }();
  return r;
}

bool sameChunk(const Chunk &a, const Chunk &b) {
  for (int y = 0; y < kChunkSize; ++y)
    for (int z = 0; z < kChunkSize; ++z)
      for (int x = 0; x < kChunkSize; ++x)
        if (a.get(x, y, z) != b.get(x, y, z))
          return false;
  return true;
}

void meshOf(const MeshInput &in, ChunkMeshData &out) {
  static std::unique_ptr<ChunkMesher::Scratch> scratch = ChunkMesher::makeScratch();
  ChunkMesher::mesh(in, registry(), *scratch, out);
}

void fillInput(MeshInput &in, BlockId id) { std::fill(in.blocks.begin(), in.blocks.end(), id); }

} // namespace

ATM_TEST(voxel_chunk_set_get_all_bit_widths) {
  Chunk c;
  ATM_CHECK(c.isUniform());
  ATM_CHECK_EQ(c.get(3, 4, 5), kAir);
  // Distinct ids force 0 -> 1 -> 2 -> 4 -> 8 -> 16 bits.
  const int counts[] = {2, 3, 5, 17, 257, 600};
  int written = 0;
  for (int n : counts) {
    for (; written < n - 1; ++written) {
      const int i = written * 37 % kChunkVolume;
      c.set(i & 31, i >> 10, (i >> 5) & 31, BlockId(written + 1));
    }
    for (int k = 0; k < written; ++k) {
      const int i = k * 37 % kChunkVolume;
      ATM_CHECK_EQ(c.get(i & 31, i >> 10, (i >> 5) & 31), BlockId(k + 1));
    }
    ATM_CHECK(!c.isUniform());
  }
  ATM_CHECK_EQ(c.get(31, 31, 31), kAir);
  ATM_CHECK(c.memoryBytes() > size_t(kChunkVolume) * 2 - 1);
  // Overwrite and read back.
  c.set(0, 0, 0, 12345);
  ATM_CHECK_EQ(c.get(0, 0, 0), BlockId(12345));
  // Out of range is ignored / returns air.
  c.set(32, 0, 0, 5);
  ATM_CHECK_EQ(c.get(-1, 0, 0), kAir);
}

ATM_TEST(voxel_chunk_revision_and_fill) {
  Chunk c(blocks::Stone);
  ATM_CHECK(c.isUniform());
  ATM_CHECK_EQ(c.uniformBlock(), blocks::Stone);
  const uint32_t r0 = c.revision();
  c.set(1, 1, 1, blocks::Stone); // no change
  ATM_CHECK_EQ(c.revision(), r0);
  c.set(1, 1, 1, blocks::Dirt);
  ATM_CHECK(c.revision() != r0);
  c.fill(blocks::Air);
  ATM_CHECK(c.isUniform());
  ATM_CHECK_EQ(c.get(1, 1, 1), kAir);
}

ATM_TEST(voxel_chunk_compact) {
  Chunk c;
  for (int i = 0; i < 40; ++i)
    c.set(i % 32, i / 32, 0, BlockId(100 + i)); // 41 entries -> 8 bits
  const size_t big = c.memoryBytes();
  for (int i = 0; i < 40; ++i)
    c.set(i % 32, i / 32, 0, blocks::Stone);
  c.set(5, 5, 5, blocks::Dirt);
  c.compact(); // air, stone, dirt -> 2 bits
  ATM_CHECK(c.memoryBytes() < big);
  ATM_CHECK_EQ(c.get(5, 5, 5), blocks::Dirt);
  ATM_CHECK_EQ(c.get(3, 0, 0), blocks::Stone);
  ATM_CHECK_EQ(c.get(9, 9, 9), kAir);
  // Everything the same -> uniform after compact.
  Chunk u;
  u.set(1, 2, 3, blocks::Sand);
  u.set(1, 2, 3, kAir);
  ATM_CHECK(!u.isUniform());
  u.compact();
  ATM_CHECK(u.isUniform());
  ATM_CHECK_EQ(u.uniformBlock(), kAir);
}

ATM_TEST(voxel_chunk_encode_decode_all) {
  std::vector<BlockId> in(kChunkVolume), out(kChunkVolume);
  uint32_t s = 7;
  for (int i = 0; i < kChunkVolume; ++i)
    in[size_t(i)] = BlockId(lcg(s) % 9);
  Chunk c;
  c.encodeAll(in.data());
  c.decodeAll(out.data());
  ATM_CHECK(in == out);
  ATM_CHECK_EQ(c.get(4, 2, 1), in[size_t(localIndex(4, 2, 1))]);
}

ATM_TEST(voxel_chunk_serialize_round_trip) {
  std::vector<uint8_t> bytes, bytes2;
  // Uniform chunk: tiny.
  {
    Chunk c(blocks::Stone), d;
    c.serialize(bytes);
    ATM_CHECK(bytes.size() <= 8);
    ATM_REQUIRE(d.deserialize(bytes));
    ATM_CHECK(d.isUniform());
    ATM_CHECK_EQ(d.uniformBlock(), blocks::Stone);
  }
  // Layered terrain-like chunk (RLE wins) and noisy chunks (raw wins).
  const int paletteSizes[] = {2, 4, 12, 200, 3000};
  for (int n : paletteSizes) {
    std::vector<BlockId> ids(kChunkVolume);
    uint32_t s = uint32_t(n) * 31u;
    for (int i = 0; i < kChunkVolume; ++i)
      ids[size_t(i)] = n == 4 ? BlockId((i >> 10) % 4) : BlockId(lcg(s) % uint32_t(n));
    Chunk c, d;
    c.encodeAll(ids.data());
    c.serialize(bytes);
    ATM_REQUIRE(d.deserialize(bytes));
    ATM_CHECK(sameChunk(c, d));
    d.serialize(bytes2);
    ATM_CHECK(bytes == bytes2);
    if (n == 4)
      ATM_CHECK(bytes.size() < 200); // 32 layers -> 32 runs
  }
}

ATM_TEST(voxel_chunk_deserialize_rejects_bad_input) {
  Chunk c;
  uint32_t s = 99;
  for (int i = 0; i < 2000; ++i)
    c.set(int(lcg(s) % 32), int(lcg(s) % 32), int(lcg(s) % 32), BlockId(1 + lcg(s) % 5));
  std::vector<uint8_t> good;
  c.serialize(good);

  Chunk target(blocks::Sand);
  ATM_CHECK(!target.deserialize({}));
  // Every truncation must fail and leave the chunk untouched.
  for (size_t len = 0; len < good.size(); ++len)
    ATM_CHECK(!target.deserialize(std::span<const uint8_t>(good.data(), len)));
  ATM_CHECK(target.isUniform());
  ATM_CHECK_EQ(target.uniformBlock(), blocks::Sand);
  // Trailing garbage.
  std::vector<uint8_t> longer = good;
  longer.push_back(0);
  ATM_CHECK(!target.deserialize(longer));
  // Bad version / bit width / palette size.
  std::vector<uint8_t> bad = good;
  bad[0] = 77;
  ATM_CHECK(!target.deserialize(bad));
  bad = good;
  bad[1] = 3;
  ATM_CHECK(!target.deserialize(bad));
  // Random garbage never crashes.
  std::vector<uint8_t> junk;
  for (int t = 0; t < 3000; ++t) {
    junk.resize(lcg(s) % 200);
    for (uint8_t &b : junk)
      b = uint8_t(lcg(s));
    if (!junk.empty())
      junk[0] = 1; // valid version so the parser goes deeper
    Chunk x;
    (void)x.deserialize(junk);
  }
  // Flipped bytes of a valid stream: parse or reject, never crash; if it
  // parses, every block must be readable.
  for (size_t i = 0; i < good.size(); ++i) {
    bad = good;
    bad[i] ^= 0x5A;
    Chunk x;
    if (x.deserialize(bad))
      (void)x.get(int(i % 32), 0, 0);
  }
  ATM_REQUIRE(target.deserialize(good));
  ATM_CHECK(sameChunk(target, c));
}

ATM_TEST(voxel_chunk_coord_key_round_trip) {
  const ChunkCoord cases[] = {{0, 0, 0},     {1, 2, 3},        {-1, -1, -1},
                              {-5, 7, 123},  {33554431, 2047, -33554432},
                              {-33554432, -2048, 33554431}, {1000000, 15, -1000000}};
  for (const ChunkCoord &c : cases) {
    ATM_CHECK(ChunkCoord::fromKey(c.key()) == c);
  }
  ATM_CHECK((ChunkCoord{1, 0, 0}.key() != ChunkCoord{0, 0, 1}.key()));
  ATM_CHECK((ChunkCoord{-1, 0, 0}.key() != ChunkCoord{1, 0, 0}.key()));
  // Block -> chunk/local for negative positions.
  const BlockPos p{-1, 5, -33};
  ATM_CHECK(chunkOf(p) == (ChunkCoord{-1, 0, -2}));
  ATM_CHECK_EQ(int(localOf(p).x), 31);
  ATM_CHECK_EQ(int(localOf(p).z), 31);
}

ATM_TEST(voxel_registry_defaults) {
  const BlockRegistry &r = registry();
  ATM_CHECK_EQ(r.size(), size_t(blocks::Count));
  ATM_CHECK_EQ(r.find("stone"), blocks::Stone);
  ATM_CHECK_EQ(r.find("bedrock"), blocks::Bedrock);
  ATM_CHECK_EQ(r.find("nope"), kAir);
  ATM_CHECK(r.translucent(blocks::Water));
  ATM_CHECK(!r.solid(blocks::Water));
  ATM_CHECK(r.get(blocks::Water).liquid);
  ATM_CHECK(r.opaque(blocks::Stone));
  ATM_CHECK(!r.opaque(blocks::OakLeaves));
  ATM_CHECK(r.meshed(blocks::OakLeaves));
  ATM_CHECK(!r.meshed(kAir));
  ATM_CHECK(r.get(blocks::Crystal).emission >= 12);
  ATM_CHECK(r.get(blocks::Lamp).emission >= 12);
  ATM_CHECK(r.get(blocks::GoldOre).miningLevel > r.get(blocks::IronOre).miningLevel);
}

ATM_TEST(voxel_mesher_single_block) {
  MeshInput in;
  in.at(5, 5, 5) = blocks::Stone;
  ChunkMeshData m;
  meshOf(in, m);
  ATM_CHECK_EQ(m.opaque.size(), size_t(6));
  ATM_CHECK(m.translucent.empty());
  for (int d = 0; d < kFaceDirCount; ++d)
    ATM_CHECK_EQ(m.opaqueDirOffset[size_t(d) + 1] - m.opaqueDirOffset[size_t(d)], 1u);
  for (int d = 0; d < kFaceDirCount; ++d) {
    const FaceFields f = unpackFace(m.opaque[m.opaqueDirOffset[size_t(d)]]);
    ATM_CHECK_EQ(int(f.dir), d);
    ATM_CHECK_EQ(f.w, 1u);
    ATM_CHECK_EQ(f.h, 1u);
    ATM_CHECK_EQ(f.material, blocks::Stone);
    ATM_CHECK_EQ(int(f.ao), 0xFF); // nothing around it
    ATM_CHECK_EQ(int(f.sky), 15);
    const bool positive = (d & 1) == 0;
    const uint32_t n = positive ? 6u : 5u;
    if (d < 2)
      ATM_CHECK(f.x == n && f.y == 5u && f.z == 5u);
    else if (d < 4)
      ATM_CHECK(f.y == n && f.x == 5u && f.z == 5u);
    else
      ATM_CHECK(f.z == n && f.x == 5u && f.y == 5u);
  }
  ATM_CHECK(m.minX == 5 && m.maxX == 6 && m.minY == 5 && m.maxY == 6 && m.minZ == 5 && m.maxZ == 6);
  ATM_CHECK_EQ(m.faceConnectivity, ~0ull);
}

ATM_TEST(voxel_mesher_merges_two_blocks) {
  MeshInput in;
  in.at(5, 5, 5) = blocks::Stone;
  in.at(6, 5, 5) = blocks::Stone;
  ChunkMeshData m;
  meshOf(in, m);
  ATM_REQUIRE(m.opaque.size() == 6);
  for (int d = 0; d < kFaceDirCount; ++d) {
    const FaceFields f = unpackFace(m.opaque[m.opaqueDirOffset[size_t(d)]]);
    if (d < 2) { // ±X: U = Z, V = Y
      ATM_CHECK_EQ(f.w, 1u);
      ATM_CHECK_EQ(f.h, 1u);
      ATM_CHECK_EQ(f.x, d == 0 ? 7u : 5u);
    } else if (d < 4) { // ±Y: U = X
      ATM_CHECK_EQ(f.w, 2u);
      ATM_CHECK_EQ(f.h, 1u);
      ATM_CHECK_EQ(f.x, 5u);
    } else { // ±Z: U = X
      ATM_CHECK_EQ(f.w, 2u);
      ATM_CHECK_EQ(f.h, 1u);
      ATM_CHECK_EQ(f.x, 5u);
    }
  }
}

ATM_TEST(voxel_mesher_ao_splits_quads) {
  // A floor with one block on top: the floor's top faces next to the block
  // get darker corners and must not merge with the unshaded ones.
  MeshInput in;
  for (int z = 0; z < 32; ++z)
    for (int x = 0; x < 32; ++x)
      in.at(x, 0, z) = blocks::Stone;
  ChunkMeshData flat;
  meshOf(in, flat);
  const uint32_t upFlat = flat.opaqueDirOffset[3] - flat.opaqueDirOffset[2];
  ATM_CHECK_EQ(upFlat, 1u); // one 32x32 quad
  in.at(10, 1, 10) = blocks::Stone;
  ChunkMeshData m;
  meshOf(in, m);
  const uint32_t up = m.opaqueDirOffset[3] - m.opaqueDirOffset[2];
  ATM_CHECK(up > 2u);
  bool sawDark = false;
  for (uint32_t i = m.opaqueDirOffset[2]; i < m.opaqueDirOffset[3]; ++i)
    if (unpackFace(m.opaque[i]).ao != 0xFF)
      sawDark = true;
  ATM_CHECK(sawDark);
}

ATM_TEST(voxel_mesher_buried_block) {
  MeshInput in;
  fillInput(in, blocks::Stone);
  ChunkMeshData m;
  meshOf(in, m);
  ATM_CHECK(m.empty());
  ATM_CHECK_EQ(m.faceConnectivity, 0ull);
}

ATM_TEST(voxel_mesher_water_culls_internal_faces) {
  MeshInput in;
  in.at(5, 5, 5) = blocks::Water;
  in.at(6, 5, 5) = blocks::Water;
  ChunkMeshData m;
  meshOf(in, m);
  ATM_CHECK(m.opaque.empty());
  ATM_CHECK_EQ(m.translucent.size(), size_t(6));
  // Water against glass keeps both faces; water on stone shows the stone top.
  MeshInput g;
  g.at(5, 5, 5) = blocks::Water;
  g.at(6, 5, 5) = blocks::Glass;
  g.at(5, 4, 5) = blocks::Stone;
  ChunkMeshData mg;
  meshOf(g, mg);
  ATM_CHECK_EQ(mg.translucent.size(), size_t(12) - 1); // stone hides water's bottom
  ATM_CHECK_EQ(mg.opaque.size(), size_t(6));           // stone top visible through water
}

ATM_TEST(voxel_mesher_leaves_do_not_hide) {
  MeshInput in;
  in.at(5, 5, 5) = blocks::OakLeaves;
  in.at(6, 5, 5) = blocks::Stone;
  ChunkMeshData m;
  meshOf(in, m);
  // Leaves: 5 faces (stone hides one). Stone: 6 (leaves don't hide).
  ATM_CHECK_EQ(m.opaque.size(), size_t(11));
  ATM_CHECK(m.translucent.empty());
}

ATM_TEST(voxel_mesher_connectivity) {
  MeshInput air;
  ChunkMeshData m;
  meshOf(air, m);
  ATM_CHECK(m.empty());
  ATM_CHECK_EQ(m.faceConnectivity, ~0ull);

  // A horizontal stone slab splits top from bottom.
  MeshInput slab;
  for (int z = 0; z < 32; ++z)
    for (int x = 0; x < 32; ++x)
      slab.at(x, 16, z) = blocks::Stone;
  meshOf(slab, m);
  ATM_CHECK(!m.connects(FaceDir::PosY, FaceDir::NegY));
  ATM_CHECK(m.connects(FaceDir::PosX, FaceDir::NegX));
  ATM_CHECK(m.connects(FaceDir::PosY, FaceDir::PosZ));
  ATM_CHECK(m.connects(FaceDir::NegY, FaceDir::NegZ));
  // Punch a hole: now connected.
  slab.at(3, 16, 3) = kAir;
  meshOf(slab, m);
  ATM_CHECK(m.connects(FaceDir::PosY, FaceDir::NegY));
}

ATM_TEST(voxel_lighting_sky_and_block_light) {
  // Stone roof over an open area at chunk y=0: under the roof it is darker
  // than in the open, a lamp lights its surroundings.
  Chunk c;
  for (int z = 0; z < 32; ++z)
    for (int x = 0; x < 16; ++x)
      c.set(x, 20, z, blocks::Stone);
  c.set(24, 5, 16, blocks::Lamp);
  auto scratch = ChunkLighting::makeScratch();
  ChunkNeighbourhood hood;
  hood.center = {0, 1, 0};
  hood.chunks[size_t(ChunkNeighbourhood::index(0, 0, 0))] = &c;
  MeshInput in;
  ChunkLighting::buildMeshInput(hood, registry(), *scratch, in);
  ATM_CHECK_EQ(in.at(24, 5, 16), blocks::Lamp);
  ATM_CHECK_EQ(in.lightAt(30, 10, 16) >> 4, 15); // open sky
  const int under = in.lightAt(8, 10, 16) >> 4;  // under the roof, 9 blocks from open sky
  ATM_CHECK(under < 15);
  ATM_CHECK_EQ(in.lightAt(25, 5, 16) & 15, 14); // next to the lamp (emission 15)
}

ATM_TEST(voxel_worldgen_deterministic) {
  WorldGenerator a(42), b(42), other(4242);
  std::vector<uint8_t> ba, bb, bo;
  bool anyDifferent = false;
  const ChunkCoord coords[] = {{0, 2, 0}, {-3, 1, 5}, {7, 0, -9}, {2, 3, 2}};
  for (const ChunkCoord &cc : coords) {
    Chunk ca, cb, co;
    a.generate(cc, ca);
    b.generate(cc, cb);
    other.generate(cc, co);
    ca.serialize(ba);
    cb.serialize(bb);
    co.serialize(bo);
    ATM_CHECK(ba == bb);
    if (ba != bo)
      anyDifferent = true;
  }
  ATM_CHECK(anyDifferent);
  // Bedrock floor and plausible heights.
  Chunk bottom;
  a.generate({0, 0, 0}, bottom);
  ATM_CHECK_EQ(bottom.get(4, 0, 4), blocks::Bedrock);
  for (int i = 0; i < 50; ++i) {
    const int h = a.surfaceHeight(i * 97 - 2000, i * 53 - 1000);
    ATM_CHECK(h >= 40 && h <= 160);
  }
  ATM_CHECK_EQ(a.surfaceHeight(123, -456), b.surfaceHeight(123, -456));
}

ATM_TEST(voxel_world_server_streaming_and_edits) {
  VoxelWorldConfig cfg;
  cfg.seed = 7;
  cfg.meshing = false;
  cfg.workerThreads = 2;
  cfg.viewRadiusChunks = 2;
  cfg.verticalChunksBelow = 1;
  cfg.verticalChunksAbove = 1;
  VoxelWorld world(cfg, registry());
  const size_t expected = 13 * 3; // 13 columns in radius 2, 3 chunks tall
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (std::chrono::steady_clock::now() < deadline) {
    world.clearFoci();
    world.addFocus(0.0, 80.0, 0.0);
    world.update();
    if (world.stats().loadedChunks >= expected)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ATM_REQUIRE(world.stats().loadedChunks == expected);
  ATM_CHECK(world.isLoaded({0, 2, 0}));
  ATM_CHECK(!world.isLoaded({10, 2, 0}));
  ATM_CHECK(world.stats().avgGenMicros > 0.0);

  // Generated data matches the generator.
  Chunk ref;
  world.generator().generate({0, 2, 0}, ref);
  ATM_CHECK(sameChunk(ref, *world.chunk({0, 2, 0})));

  const BlockPos p{3, 70, -4};
  const auto before = world.chunk(chunkOf(p));
  ATM_CHECK(world.setBlock(p, blocks::Lamp));
  ATM_CHECK_EQ(world.blockAt(p), blocks::Lamp);
  ATM_CHECK(world.isEdited(chunkOf(p)));
  ATM_CHECK_EQ(world.editedChunkKeys().size(), size_t(1));
  // Copy-on-write: the old snapshot is unchanged.
  const LocalPos l = localOf(p);
  ATM_CHECK(before->get(l.x, l.y, l.z) != blocks::Lamp || ref.get(l.x, l.y, l.z) == blocks::Lamp);
  ATM_CHECK(!world.setBlock({1000, 70, 0}, blocks::Stone)); // not loaded
  ATM_CHECK_EQ(world.blockAt({0, -5, 0}), kAir);

  // Moving away unloads; edits survive and come back.
  for (int i = 0; i < 5; ++i) {
    world.clearFoci();
    world.addFocus(3200.0, 80.0, 0.0);
    world.update();
  }
  ATM_CHECK(!world.isLoaded(chunkOf(p)));
  for (int i = 0; i < 5; ++i) {
    world.clearFoci();
    world.addFocus(0.0, 80.0, 0.0);
    world.update();
  }
  ATM_CHECK(world.isLoaded(chunkOf(p))); // restored from the edit store
  ATM_CHECK_EQ(world.blockAt(p), blocks::Lamp);
}

ATM_TEST(voxel_world_client_meshing) {
  VoxelWorldConfig cfg;
  cfg.seed = 3;
  cfg.meshing = true;
  cfg.workerThreads = 2;
  cfg.viewRadiusChunks = 2;
  cfg.verticalChunksBelow = 1;
  cfg.verticalChunksAbove = 1;
  VoxelWorld world(cfg, registry());
  // Focus on the surface so the focus chunk is never uniform air.
  const int surface = world.generator().surfaceHeight(0, 0);
  const int focusCy = surface >> kChunkShift;
  std::vector<ChunkMeshResult> meshes;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (std::chrono::steady_clock::now() < deadline && meshes.empty()) {
    world.clearFoci();
    world.addFocus(0.5, double(surface) + 0.5, 0.5);
    world.update();
    world.takeMeshResults(meshes);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ATM_REQUIRE(!meshes.empty());
  // Only the focus column has all 26 neighbours loaded: the focus chunk, plus
  // the chunk below it when that is the bottom layer (cy = 0 has no loadable
  // neighbours below, which count as ready).
  for (const ChunkMeshResult &r : meshes) {
    const bool bottomLayer = focusCy == 1 && r.coord.y == 0;
    ATM_CHECK(r.coord.x == 0 && r.coord.z == 0 && (r.coord.y == focusCy || bottomLayer));
    ATM_CHECK(!r.mesh.empty());
    ATM_CHECK(r.mesh.opaqueDirOffset[kFaceDirCount] == r.mesh.opaque.size());
  }
}
