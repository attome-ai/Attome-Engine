// Gameplay rules: XP curve, combat level, content tables, animation, models,
// movement and ray casts.

#include "atm_test.h"

#include "../shared/GameTypes.h"
#include "../shared/Movement.h"

#include "../../engine/model/Character.h"
#include "../../engine/voxel/BlockRegistry.h"
#include "../../engine/voxel/Chunk.h"

#include <cmath>
#include <initializer_list>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace ao;
using atm::voxel::BlockPos;
using atm::voxel::ChunkCoord;
using atm::voxel::FaceDir;

namespace {

// Flat stone floor with its top at y = floorTop, plus explicit overrides.
struct FakeWorld : atm::voxel::IBlockAccess {
  int floorTop = 11; // blocks y < floorTop are stone
  std::map<std::tuple<int, int, int>, BlockId> edits;

  BlockId blockAt(BlockPos p) const override {
    auto it = edits.find({p.x, p.y, p.z});
    if (it != edits.end())
      return it->second;
    return p.y < floorTop ? atm::voxel::blocks::Stone : atm::voxel::kAir;
  }
  bool isLoaded(ChunkCoord) const override { return true; }
  void set(int x, int y, int z, BlockId id) { edits[{x, y, z}] = id; }
};

const atm::voxel::BlockRegistry &registry() {
  static const atm::voxel::BlockRegistry r = [] {
    atm::voxel::BlockRegistry reg;
    reg.registerDefaults();
    return reg;
  }();
  return r;
}

MoveInput input(float moveX, float moveZ, uint16_t buttons, float yaw = 0.0f) {
  MoveInput in;
  in.moveX = moveX;
  in.moveZ = moveZ;
  in.buttons = buttons;
  in.yaw = yaw;
  return in;
}

void settle(MoveState &s, const FakeWorld &w, int ticks = 90) {
  for (int i = 0; i < ticks; ++i)
    stepMovement(s, input(0, 0, 0), w, registry(), kSimDt);
}

} // namespace

// ---------------------------------------------------------------------------
// Skills
// ---------------------------------------------------------------------------

ATM_TEST(gameplay_xp_curve) {
  ATM_CHECK_EQ(xpForLevel(1), uint64_t(0));
  ATM_CHECK_EQ(xpForLevel(2), uint64_t(83));
  ATM_CHECK_EQ(xpForLevel(10), uint64_t(1154));
  ATM_CHECK_EQ(xpForLevel(92), uint64_t(6517253));
  ATM_CHECK_EQ(xpForLevel(98), uint64_t(11805606));
  ATM_CHECK_EQ(xpForLevel(99), uint64_t(13034431));
  ATM_CHECK_EQ(xpForLevel(100), uint64_t(13034431 + 1228825));
  ATM_CHECK_EQ(xpForLevel(120), uint64_t(38839756));
  ATM_CHECK_EQ(xpForLevel(150), uint64_t(75704506));
  ATM_CHECK_EQ(xpForLevel(200), uint64_t(137145756));
}

ATM_TEST(gameplay_level_for_xp) {
  ATM_CHECK_EQ(levelForXp(0), 1u);
  ATM_CHECK_EQ(levelForXp(82), 1u);
  ATM_CHECK_EQ(levelForXp(83), 2u);
  for (uint32_t level = 2; level <= 400; ++level) {
    ATM_CHECK_EQ(levelForXp(xpForLevel(level)), level);
    ATM_CHECK_EQ(levelForXp(xpForLevel(level) - 1), level - 1);
  }
  // 10 billion XP: 99 + floor((1e10 - 13,034,431) / 1,228,825) = 8226
  const uint64_t big = 10'000'000'000ull;
  const uint32_t l = levelForXp(big);
  ATM_CHECK_EQ(l, 8226u);
  ATM_CHECK(xpForLevel(l) <= big);
  ATM_CHECK(xpForLevel(l + 1) > big);
  ATM_CHECK(levelForXp(~0ull) > 1000000u);
}

ATM_TEST(gameplay_combat_level) {
  std::array<uint32_t, kSkillCount> lv{};
  lv.fill(1);
  lv[size_t(Skill::Hitpoints)] = 10;
  ATM_CHECK_EQ(combatLevel(lv), 3u);
  lv.fill(99);
  ATM_CHECK_EQ(combatLevel(lv), 126u);
  lv.fill(500); // capped at 99
  ATM_CHECK_EQ(combatLevel(lv), 126u);
  lv.fill(1);
  lv[size_t(Skill::Hitpoints)] = 10;
  lv[size_t(Skill::Ranged)] = 99; // 0.25*11 + 0.325*148 = 50.85
  ATM_CHECK_EQ(combatLevel(lv), 50u);
  ATM_CHECK(skillName(Skill::Taming) == "Taming");
}

// ---------------------------------------------------------------------------
// Content tables
// ---------------------------------------------------------------------------

ATM_TEST(gameplay_items_and_monsters) {
  ATM_CHECK_EQ(itemCount(), items::Count);
  ATM_CHECK_EQ(findItem("iron_sword"), items::IronSword);
  ATM_CHECK_EQ(findItem("nope"), ItemId(0));
  for (ItemId i = 1; i < itemCount(); ++i) {
    ATM_CHECK(!itemDef(i).name.empty());
    ATM_CHECK_EQ(findItem(itemDef(i).name), i);
    if (itemDef(i).kind == ItemKind::Block)
      ATM_CHECK(itemDef(i).placesBlock != 0);
  }
  ATM_CHECK_EQ(blockDropItem(atm::voxel::blocks::Grass), items::Dirt);
  ATM_CHECK_EQ(blockDropItem(atm::voxel::blocks::IronOre), items::IronOre);
  ATM_CHECK_EQ(blockDropItem(atm::voxel::blocks::Bedrock), ItemId(0));
  ATM_CHECK(itemDef(items::CookedMeat).healAmount > 0);
  ATM_CHECK_EQ(monsterTypeCount(), monsters::Count);
  for (uint8_t m = 0; m < monsterTypeCount(); ++m) {
    const MonsterDef &d = monsterDef(m);
    ATM_CHECK(d.maxHp > 0 && d.speed > 0.0f && d.model != nullptr);
    ATM_CHECK(d.rare.item != 0 && d.rare.item < itemCount());
    for (const auto &drop : d.common)
      ATM_CHECK(drop.item < itemCount() && drop.min <= drop.max);
  }
}

// ---------------------------------------------------------------------------
// Models and animation
// ---------------------------------------------------------------------------

ATM_TEST(gameplay_rigs_parent_first) {
  using atm::model::Rig;
  for (const Rig *rig : {&Rig::humanoid(), &Rig::quadruped(), &Rig::slime()})
    for (int b = 1; b < atm::model::kBoneCount; ++b)
      ATM_CHECK(int(rig->bones[size_t(b)].parent) < b);
}

ATM_TEST(gameplay_model_library) {
  using namespace atm::model;
  ModelLibrary lib;
  lib.buildDefaults();
  const auto &parts = lib.parts();
  ATM_REQUIRE(!parts.empty());
  for (const VoxelPart &p : parts) {
    ATM_CHECK(p.sx >= 1 && p.sx <= 32 && p.sy >= 1 && p.sy <= 32 && p.sz >= 1 && p.sz <= 32);
    ATM_CHECK_EQ(p.voxels.size(), size_t(p.sx) * size_t(p.sy) * size_t(p.sz));
    for (uint8_t v : p.voxels)
      if (v >= p.palette.size()) {
        ATM_CHECK(v < p.palette.size());
        break;
      }
  }
  auto validPart = [&](int idx) { return idx == -1 || (idx >= 0 && size_t(idx) < parts.size()); };
  for (PieceId id = 1; id < lib.pieceCount(); ++id) {
    const EquipPiece &piece = lib.piece(id);
    for (int16_t bp : piece.boneParts)
      ATM_CHECK(validPart(bp));
    ATM_CHECK(validPart(piece.socketPart));
  }
  // Every equipment item resolves to a piece; every monster to a creature.
  for (ItemId i = 1; i < itemCount(); ++i)
    if (itemDef(i).piece)
      ATM_CHECK(lib.findPiece(itemDef(i).piece) != kNoPiece);
  for (uint8_t m = 0; m < monsterTypeCount(); ++m)
    ATM_CHECK(lib.findCreature(monsterDef(m).model) >= 0);
  for (uint8_t shape = 0; shape < ModelLibrary::kBodyShapeCount; ++shape)
    for (int b = 1; b < kBoneCount; ++b)
      ATM_CHECK(lib.bodyPart(Bone(b), shape) >= 0);

  ATM_CHECK_EQ(lib.materialColors().size(), size_t(lib.materialCount()));
  ATM_CHECK_EQ(lib.materialEmissive().size(), size_t(lib.materialCount()));

  Appearance a;
  a.skinTone = 2;
  a.hairStyle = 1;
  const ResolvedModel bare = resolveAppearance(lib, a);
  ATM_CHECK(bare.socketParts[size_t(Socket::Head)] >= 0); // hair
  ATM_CHECK(bare.bonePaletteOffset[size_t(Bone::Head)] > 0); // skin variant
  a.pieces[size_t(EquipSlot::Head)] = lib.findPiece("iron_helm");
  a.pieces[size_t(EquipSlot::Torso)] = lib.findPiece("iron_chestplate");
  a.pieces[size_t(EquipSlot::MainHand)] = lib.findPiece("crystal_sword");
  const ResolvedModel geared = resolveAppearance(lib, a);
  ATM_CHECK(geared.socketParts[size_t(Socket::Head)] == lib.piece(a.pieces[0]).socketPart);
  ATM_CHECK(geared.boneParts[size_t(Bone::Torso)] != bare.boneParts[size_t(Bone::Torso)]);
  ATM_CHECK(geared.socketParts[size_t(Socket::MainHand)] >= 0);
  for (int16_t p : geared.boneParts)
    ATM_CHECK(validPart(p));

  // Meshing a part: faces carry materials in [base, base + palette size).
  const int torso = lib.bodyPart(Bone::Torso, 0);
  atm::voxel::ChunkMeshData mesh;
  meshPart(parts[size_t(torso)], lib.materialBase(torso), mesh);
  ATM_CHECK(!mesh.opaque.empty());
  for (atm::voxel::PackedFace f : mesh.opaque) {
    const auto ff = atm::voxel::unpackFace(f);
    ATM_CHECK(ff.material > lib.materialBase(torso) &&
              ff.material < lib.materialBase(torso) + parts[size_t(torso)].palette.size());
  }
}

ATM_TEST(gameplay_animator) {
  using namespace atm::model;
  AnimLibrary lib;
  lib.buildDefaults();
  for (const char *name : {"idle", "walk", "run", "sprint", "jump", "fall", "glide", "swim", "dash",
                           "mine", "place", "sword_swing", "bow_draw", "bow_shoot", "staff_cast",
                           "hit_react", "death", "wave", "quad_idle", "quad_walk", "quad_attack",
                           "slime_hop"})
    ATM_CHECK(lib.indexOf(name) >= 0);

  Animator anim;
  anim.setLocomotion(lib.indexOf("idle"));
  anim.update(lib, 0.1f);
  anim.setLocomotion(lib.indexOf("run"));
  anim.playAction(lib.indexOf("sword_swing"));
  ATM_CHECK(anim.actionPlaying());
  uint32_t events = 0;
  const Rig &rig = Rig::humanoid();
  std::array<glm::mat4, kBoneCount> m;
  for (int i = 0; i < 40; ++i) {
    anim.update(lib, 1.0f / 60.0f);
    events |= anim.takeEvents();
    anim.evaluate(lib, rig, m);
    for (const glm::mat4 &x : m)
      for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
          ATM_CHECK(std::isfinite(x[c][r]));
  }
  ATM_CHECK(events & (1u << anim_event::Hit));
  ATM_CHECK(events & (1u << anim_event::Footstep));
  ATM_CHECK(!anim.actionPlaying()); // 0.5 s one-shot ended after ~0.67 s

  // Bind pose: bones sit at the sum of their parents' offsets.
  Animator bind;
  bind.evaluate(lib, rig, m);
  const glm::vec3 head(m[size_t(Bone::Head)][3]);
  ATM_CHECK_NEAR(head.y, 17.0f, 1e-4);
  const glm::vec3 handR(m[size_t(Bone::HandR)][3]);
  ATM_CHECK_NEAR(handR.x, 6.5f, 1e-4);
  ATM_CHECK_NEAR(handR.y, 10.0f, 1e-4);
}

// ---------------------------------------------------------------------------
// Movement
// ---------------------------------------------------------------------------

ATM_TEST(gameplay_move_falls_to_floor) {
  FakeWorld w;
  MoveState s;
  s.pos = {0.5, 16.0, 0.5};
  settle(s, w);
  ATM_CHECK(s.onGround);
  ATM_CHECK_NEAR(s.pos.y, 11.0, 1e-6);
  ATM_CHECK_NEAR(s.vel.y, 0.0, 1e-6);
  settle(s, w, 30); // stays put
  ATM_CHECK_NEAR(s.pos.y, 11.0, 1e-6);
}

ATM_TEST(gameplay_move_wall_blocks) {
  FakeWorld w;
  for (int z = -4; z <= 4; ++z)
    for (int y = 11; y <= 14; ++y)
      w.set(3, y, z, atm::voxel::blocks::Stone);
  MoveState s;
  s.pos = {0.5, 11.0, 0.5};
  settle(s, w, 5);
  for (int i = 0; i < 120; ++i) // strafe right = +X at yaw 0
    stepMovement(s, input(1, 0, 0), w, registry(), kSimDt);
  ATM_CHECK(s.pos.x + moveTuning().halfWidth <= 3.0 + 1e-6);
  ATM_CHECK(s.pos.x > 2.0);
  ATM_CHECK_NEAR(s.pos.y, 11.0, 1e-6);
}

ATM_TEST(gameplay_move_steps_up_ledge) {
  FakeWorld w;
  for (int x = 3; x <= 30; ++x)
    for (int z = -4; z <= 4; ++z)
      w.set(x, 11, z, atm::voxel::blocks::Stone);
  MoveState s;
  s.pos = {0.5, 11.0, 0.5};
  settle(s, w, 5);
  for (int i = 0; i < 30; ++i)
    stepMovement(s, input(1, 0, 0), w, registry(), kSimDt);
  ATM_CHECK(s.pos.x > 3.5);
  ATM_CHECK_NEAR(s.pos.y, 12.0, 1e-6);
}

ATM_TEST(gameplay_move_double_jump_once) {
  FakeWorld w;
  MoveState s;
  s.pos = {0.5, 11.0, 0.5};
  settle(s, w, 5);
  const MoveTuning &t = moveTuning();
  stepMovement(s, input(0, 0, button::Jump), w, registry(), kSimDt);
  ATM_CHECK(!s.onGround && s.vel.y > t.jumpSpeed * 0.8f);
  for (int i = 0; i < 12; ++i) // hold jump while rising
    stepMovement(s, input(0, 0, button::Jump), w, registry(), kSimDt);
  stepMovement(s, input(0, 0, 0), w, registry(), kSimDt);
  stepMovement(s, input(0, 0, button::Jump), w, registry(), kSimDt); // double jump
  ATM_CHECK(s.vel.y > t.jumpSpeed * 0.8f);
  ATM_CHECK_EQ(int(s.jumpsLeft), 0);
  stepMovement(s, input(0, 0, 0), w, registry(), kSimDt);
  const float before = s.vel.y;
  stepMovement(s, input(0, 0, button::Jump), w, registry(), kSimDt); // no third jump
  ATM_CHECK(s.vel.y < before);
  settle(s, w, 120);
  ATM_CHECK(s.onGround);
  ATM_CHECK_EQ(int(s.jumpsLeft), t.extraJumps);
}

ATM_TEST(gameplay_move_deterministic) {
  FakeWorld w;
  w.set(4, 11, 0, atm::voxel::blocks::Stone);
  auto run = [&]() {
    MoveState s;
    s.pos = {0.5, 14.0, 0.5};
    for (int i = 0; i < 200; ++i) {
      uint16_t b = 0;
      if (i % 37 == 5 || i % 37 == 12)
        b |= button::Jump;
      if (i % 50 == 20)
        b |= button::Dash;
      if (i > 100)
        b |= button::Sprint;
      stepMovement(s, input(std::sin(float(i) * 0.1f), 1.0f, b, float(i) * 0.02f), w, registry(), kSimDt);
    }
    return s;
  };
  const MoveState a = run(), b = run();
  ATM_CHECK(a.pos == b.pos);
  ATM_CHECK(a.vel == b.vel);
  ATM_CHECK_EQ(a.onGround, b.onGround);
  ATM_CHECK_EQ(a.dashCooldown, b.dashCooldown);
}

ATM_TEST(gameplay_raycast) {
  FakeWorld w;
  w.set(3, 11, 0, atm::voxel::blocks::Planks);
  BlockPos hit;
  FaceDir face = FaceDir::PosX;
  ATM_REQUIRE(raycastBlock(w, registry(), {0.5, 12.5, 0.5}, {0, -1, 0}, 8.0f, hit, face));
  ATM_CHECK(hit == (BlockPos{0, 10, 0}));
  ATM_CHECK(face == FaceDir::PosY);
  ATM_REQUIRE(raycastBlock(w, registry(), {0.5, 11.5, 0.5}, {1, 0, 0}, 8.0f, hit, face));
  ATM_CHECK(hit == (BlockPos{3, 11, 0}));
  ATM_CHECK(face == FaceDir::NegX);
  ATM_CHECK(!raycastBlock(w, registry(), {0.5, 12.5, 0.5}, {0, 1, 0}, 8.0f, hit, face));
  ATM_CHECK(!raycastBlock(w, registry(), {0.5, 13.5, 0.5}, {0, -1, 0}, 1.0f, hit, face));
}
