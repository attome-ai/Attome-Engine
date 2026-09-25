// ModelLibrary bookkeeping, part meshing and appearance resolution. The demo
// content itself (procedural parts and pieces) is in ModelContent.cpp.

#include "Character.h"

#include "../voxel/BlockRegistry.h"
#include "../voxel/Chunk.h"

#include <algorithm>
#include <memory>
#include <string>

namespace atm::model {

// ---------------------------------------------------------------------------
// ModelLibrary
// ---------------------------------------------------------------------------

PieceId ModelLibrary::findPiece(std::string_view name) const {
  for (size_t i = 1; i < pieces_.size(); ++i)
    if (pieces_[i].name == name)
      return PieceId(i);
  return kNoPiece;
}

int ModelLibrary::bodyPart(Bone bone, uint8_t bodyShape) const {
  if (bone >= Bone::Count)
    return -1;
  const int p = body_[bodyShape < body_.size() ? bodyShape : 0][size_t(bone)];
  return (p >= 0 && size_t(p) < parts_.size()) ? p : -1;
}

void ModelLibrary::setBodyPart(Bone bone, uint8_t bodyShape, int part) {
  if (bone >= Bone::Count || bodyShape >= body_.size())
    return;
  body_[bodyShape][size_t(bone)] = int16_t(part);
}

int ModelLibrary::addPart(VoxelPart part) {
  // Clamp to the mesher's limits and keep the voxel array consistent.
  part.sx = std::clamp(part.sx, 1, 32);
  part.sy = std::clamp(part.sy, 1, 32);
  part.sz = std::clamp(part.sz, 1, 32);
  part.voxels.resize(size_t(part.sx) * size_t(part.sy) * size_t(part.sz), 0);
  if (part.palette.empty())
    part.palette.push_back(0);
  const size_t palSize = part.palette.size();
  if (part.variantPalettes.size() % palSize != 0)
    part.variantPalettes.resize(part.variantPalettes.size() - part.variantPalettes.size() % palSize);
  const uint32_t count = uint32_t(palSize + part.variantPalettes.size());

  materialBase_.push_back(uint16_t(materialCount_));
  materialCount_ += count;
  parts_.push_back(std::move(part));
  return int(parts_.size() - 1);
}

PieceId ModelLibrary::addPiece(EquipPiece piece) {
  pieces_.push_back(std::move(piece));
  return PieceId(pieces_.size() - 1);
}

int ModelLibrary::findPart(std::string_view name) const {
  for (size_t i = 0; i < parts_.size(); ++i)
    if (parts_[i].name == name)
      return int(i);
  return -1;
}

int ModelLibrary::hairPart(uint8_t style) const {
  if (style >= hair_.size())
    return -1;
  return hair_[style];
}

int ModelLibrary::addCreature(CreatureModel model) {
  creatures_.push_back(std::move(model));
  return int(creatures_.size() - 1);
}

int ModelLibrary::findCreature(std::string_view name) const {
  for (size_t i = 0; i < creatures_.size(); ++i)
    if (creatures_[i].name == name)
      return int(i);
  return -1;
}

uint16_t ModelLibrary::paletteOffset(int part, uint8_t variant) const {
  if (part < 0 || size_t(part) >= parts_.size())
    return 0;
  const VoxelPart &p = parts_[size_t(part)];
  const int v = variant < p.variantCount() ? variant : 0;
  return uint16_t(size_t(v) * p.palette.size());
}

std::vector<uint32_t> ModelLibrary::materialColors() const {
  std::vector<uint32_t> out;
  out.reserve(materialCount_);
  for (const VoxelPart &p : parts_) {
    out.insert(out.end(), p.palette.begin(), p.palette.end());
    out.insert(out.end(), p.variantPalettes.begin(), p.variantPalettes.end());
  }
  return out;
}

std::vector<uint8_t> ModelLibrary::materialEmissive() const {
  std::vector<uint8_t> out;
  out.reserve(materialCount_);
  for (const VoxelPart &p : parts_) {
    const int variants = p.variantCount();
    for (int v = 0; v < variants; ++v)
      for (size_t k = 0; k < p.palette.size(); ++k)
        out.push_back(uint8_t(k > 0 && k >= p.emissiveFrom ? 1 : 0));
  }
  return out;
}

// ---------------------------------------------------------------------------
// meshPart
// ---------------------------------------------------------------------------

namespace {

// The chunk mesher looks block ids up in a BlockRegistry, so parts are meshed
// with a private registry of 255 plain opaque "palette" blocks and the face
// materials are remapped to materialBase + palette index afterwards.
struct PartRegistry {
  voxel::BlockRegistry reg;
  voxel::BlockId firstPalette = 1; // block id of palette index 1

  PartRegistry() {
    if (reg.size() == 0) {
      voxel::BlockDef air;
      air.name = "air";
      air.render = voxel::BlockRender::None;
      air.solid = false;
      reg.add(air);
    }
    for (int i = 1; i < 256; ++i) {
      voxel::BlockDef d;
      d.name = "palette_" + std::to_string(i);
      d.render = voxel::BlockRender::Opaque;
      d.solid = true;
      const voxel::BlockId id = reg.add(d);
      if (i == 1)
        firstPalette = id;
    }
  }
};

const PartRegistry &partRegistry() {
  static const PartRegistry r;
  return r;
}

} // namespace

void meshPart(const VoxelPart &part, uint16_t materialBase, voxel::ChunkMeshData &out) {
  out.clear();
  const PartRegistry &pr = partRegistry();
  voxel::MeshInput in; // all air, full sky light (0xF0) everywhere
  const int sx = std::min(part.sx, voxel::kChunkSize);
  const int sy = std::min(part.sy, voxel::kChunkSize);
  const int sz = std::min(part.sz, voxel::kChunkSize);
  if (part.voxels.size() < size_t(part.sx) * size_t(part.sy) * size_t(part.sz))
    return;
  for (int y = 0; y < sy; ++y)
    for (int z = 0; z < sz; ++z)
      for (int x = 0; x < sx; ++x) {
        const uint8_t v = part.at(x, y, z);
        if (v != 0)
          in.at(x, y, z) = voxel::BlockId(pr.firstPalette + v - 1);
      }

  thread_local std::unique_ptr<voxel::ChunkMesher::Scratch> scratch;
  if (!scratch)
    scratch = voxel::ChunkMesher::makeScratch();
  voxel::ChunkMesher::mesh(in, pr.reg, *scratch, out);

  auto remap = [&](std::vector<voxel::PackedFace> &faces) {
    for (voxel::PackedFace &f : faces) {
      voxel::FaceFields ff = voxel::unpackFace(f);
      const int paletteIndex = int(ff.material) - int(pr.firstPalette) + 1;
      ff.material = uint16_t(materialBase + std::max(paletteIndex, 0));
      f = voxel::packFace(ff);
    }
  };
  remap(out.opaque);
  remap(out.translucent);
}

// ---------------------------------------------------------------------------
// resolveAppearance
// ---------------------------------------------------------------------------

ResolvedModel resolveAppearance(const ModelLibrary &lib, const Appearance &a) {
  ResolvedModel r;
  r.boneParts.fill(-1);
  r.socketParts.fill(-1);
  r.bonePaletteOffset.fill(0);
  r.socketPaletteOffset.fill(0);

  const uint8_t shape = a.bodyShape < ModelLibrary::kBodyShapeCount ? a.bodyShape : uint8_t(0);
  for (int b = 0; b < kBoneCount; ++b) {
    const int part = lib.bodyPart(Bone(b), shape);
    r.boneParts[size_t(b)] = int16_t(part);
    r.bonePaletteOffset[size_t(b)] = lib.paletteOffset(part, a.skinTone);
  }

  // Hair on the head socket (hidden by any head piece below).
  const int hair = lib.hairPart(a.hairStyle);
  r.socketParts[size_t(Socket::Head)] = int16_t(hair);
  r.socketPaletteOffset[size_t(Socket::Head)] = lib.paletteOffset(hair, a.hairColor);

  const size_t partCount = lib.parts().size();
  for (int s = 0; s < kEquipSlotCount; ++s) {
    const PieceId id = a.pieces[size_t(s)];
    if (id == kNoPiece || id >= lib.pieceCount())
      continue;
    const EquipPiece &piece = lib.piece(id);
    for (int b = 0; b < kBoneCount; ++b) {
      const int16_t p = piece.boneParts[size_t(b)];
      if (p >= 0 && size_t(p) < partCount) {
        r.boneParts[size_t(b)] = p;
        r.bonePaletteOffset[size_t(b)] = 0; // armour isn't skin-toned
      }
    }
    const bool socketValid = piece.socketPart >= 0 && size_t(piece.socketPart) < partCount &&
                             piece.socket < Socket::Count;
    if (socketValid) {
      r.socketParts[size_t(piece.socket)] = piece.socketPart;
      r.socketPaletteOffset[size_t(piece.socket)] =
          lib.paletteOffset(piece.socketPart, 0); // TODO(demo): dye variants from a.dyes
    }
    // A head piece always hides hair, whatever form it takes (a Head-socket
    // part replaced it above; bone-replacement helmets or pieces without a
    // part still must not show hair poking through).
    if (EquipSlot(s) == EquipSlot::Head && !(socketValid && piece.socket == Socket::Head)) {
      r.socketParts[size_t(Socket::Head)] = -1;
      r.socketPaletteOffset[size_t(Socket::Head)] = 0;
    }
  }
  return r;
}

} // namespace atm::model
