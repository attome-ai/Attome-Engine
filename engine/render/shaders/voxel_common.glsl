// Shared declarations for the voxel shaders. Must match engine/render/GpuLayout.h
// (std140 uniform block, std430 storage buffers) and the PackedFace bit
// layout in engine/voxel/MeshTypes.h.
#ifndef VOXEL_COMMON_GLSL
#define VOXEL_COMMON_GLSL

#define GPU_LAYOUT_VERSION 2

layout(std140, set = 0, binding = 0) uniform FrameUBO {
  mat4 viewProj;      // camera-relative
  mat4 invViewProj;
  ivec4 camBlock;     // floor(camera)
  vec4 camFrac;       // xyz = camera - camBlock, w = time
  vec4 sunDir;        // xyz, w = ambient
  vec4 skyColor;      // rgb, w = timeOfDay
  vec4 fogColor;      // rgb, w = fog start
  vec4 fogParams;     // x = fog end, y = 1/(end-start), z = sun intensity
  vec4 sunColor;
  uvec4 counts;       // x = visible chunks, y = max draws
  mat4 lightViewProj; // camera-relative -> shadow map clip space
  vec4 shadowParams;  // x = texel size (blocks), y = strength (0 = off)
} frame;

struct ChunkGpu {
  int originX, originY, originZ;
  uint flags;
  uint faceBase;
  uint translucentCount;
  uint dirOffset[7];
  uint aabbMin;
  uint aabbMax;
  uint pad0;
};

struct MaterialGpu {
  uint top, side, bottom;
  float emissive;
  float alpha;
  uint flags;         // MATERIAL_WATER | MATERIAL_FOLIAGE
  uint pad1, pad2;
};

#ifndef NO_SCENE_BUFFERS
layout(std430, set = 0, binding = 1) readonly buffer FaceArena { uvec2 faces[]; };
layout(std430, set = 0, binding = 2) readonly buffer Chunks { ChunkGpu chunks[]; };
layout(std430, set = 0, binding = 3) readonly buffer Materials { MaterialGpu materials[]; };
#endif

// --- PackedFace (MeshTypes.h) ------------------------------------------------
//   lo: [0..5] x  [6..11] y  [12..17] z  [18..22] w-1  [23..27] h-1
//       [28..30] dir  [31] ao bit 0
//   hi: [0..6] ao bits 1..7  [7..22] material  [23..26] sky  [27..30] block
struct Face {
  uvec3 pos;       // quad origin corner (u0,v0) in chunk/part space, 0..32
  uint w, h;       // size along U, V (1..32)
  uint dir;        // FaceDir 0..5: +X -X +Y -Y +Z -Z
  uint ao;         // corner k AO in bits [2k, 2k+1]; 0 darkest .. 3 none
  uint material;
  uint sky, blockLight;
};

Face decodeFace(uvec2 f) {
  Face r;
  r.pos = uvec3(f.x & 63u, (f.x >> 6) & 63u, (f.x >> 12) & 63u);
  r.w = ((f.x >> 18) & 31u) + 1u;
  r.h = ((f.x >> 23) & 31u) + 1u;
  r.dir = (f.x >> 28) & 7u;
  r.ao = (f.x >> 31) | ((f.y & 0x7Fu) << 1);
  r.material = (f.y >> 7) & 0xFFFFu;
  r.sky = (f.y >> 23) & 15u;
  r.blockLight = (f.y >> 27) & 15u;
  return r;
}

const vec3 kFaceNormals[6] = vec3[6](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0),
                                     vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1));

uint aoAt(uint ao, uint corner) { return (ao >> (2u * corner)) & 3u; }

// Maps the shared index pattern (0,1,2, 0,2,3) vertex k to a quad corner:
//  - AO diagonal flip: the split diagonal joins the brighter corner pair
//    (0fps: flip when a00 + a11 < a01 + a10 would otherwise put the dark
//    corner on the diagonal). Flipping rotates corners by one, keeping winding.
//  - Winding: corners (u0,v0),(u1,v0),(u1,v1),(u0,v1) are counter-clockwise
//    seen from outside when U x V = normal: true for -X, -Y, +Z. For +X, +Y,
//    -Z the order is mirrored (c -> (4 - c) & 3), which keeps the diagonal.
uint quadCorner(uint k, uint dir, uint ao) {
  uint a0 = aoAt(ao, 0u), a1 = aoAt(ao, 1u), a2 = aoAt(ao, 2u), a3 = aoAt(ao, 3u);
  uint c = (a0 + a2 < a1 + a3) ? ((k + 1u) & 3u) : k;
  if (dir == 0u || dir == 2u || dir == 5u) c = (4u - c) & 3u;
  return c;
}

// Local position of corner c (0..3) of a face.
vec3 cornerPosition(Face f, uint c) {
  float du = (c == 1u || c == 2u) ? float(f.w) : 0.0;
  float dv = (c >= 2u) ? float(f.h) : 0.0;
  vec3 p = vec3(f.pos);
  uint axis = f.dir >> 1u;             // 0 = X, 1 = Y, 2 = Z
  if (axis == 0u) { p.z += du; p.y += dv; }        // U = Z, V = Y
  else if (axis == 1u) { p.x += du; p.z += dv; }   // U = X, V = Z
  else { p.x += du; p.y += dv; }                   // U = X, V = Y
  return p;
}

// Colours are packed as memory bytes R, G, B, A (uint value 0xAABBGGRR),
// the same as atm::voxel::packRGBA and GLSL unpackUnorm4x8.
vec4 unpackRGBA(uint c) { return unpackUnorm4x8(c); }

vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }

#define MATERIAL_WATER 1u
#define MATERIAL_FOLIAGE 2u

// Wind offset for foliage vertices. A function of the world position only,
// so vertices shared by neighbouring faces move together (no cracks).
vec3 foliageSway(vec3 worldPos, float time) {
  float ph = time * 1.6 + worldPos.x * 0.37 + worldPos.z * 0.29 + worldPos.y * 0.21;
  float gust = 0.75 + 0.25 * sin(time * 0.35 + worldPos.x * 0.02);
  return vec3(sin(ph) + 0.35 * sin(ph * 2.7 + 0.8), 0.25 * sin(ph * 1.3 + 2.1), cos(ph * 0.8 + 1.3)) *
         (0.06 * gust);
}

uint materialColor(MaterialGpu m, uint dir) {
  return dir == 2u ? m.top : (dir == 3u ? m.bottom : m.side);
}

// Minecraft-like light falloff: level 15 -> 1.0, each level ~0.8x.
float lightCurve(float level01) { return pow(0.8, 15.0 * (1.0 - level01)); }

#endif
