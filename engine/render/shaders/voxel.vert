#version 460
#extension GL_GOOGLE_include_directive : require
// Chunk faces by vertex pulling. Draw parameters (written by cull.comp or the
// CPU for translucent faces):
//   vertexOffset  = absolute first face in the arena * 4
//   firstInstance = chunk slot (gl_InstanceIndex includes firstInstance)
// Index buffer = shared quad pattern 4j + (0,1,2, 0,2,3).
#include "voxel_common.glsl"

layout(location = 0) flat out vec4 vColor;     // linear rgb, alpha
layout(location = 1) flat out vec4 vLight;     // sky, block, emissive, noise on
layout(location = 2) flat out vec3 vNormal;
layout(location = 3) out float vAo;
layout(location = 4) out vec3 vViewPos;        // camera-relative
layout(location = 5) out vec3 vLocal;          // chunk-local position
layout(location = 6) flat out ivec3 vOrigin;   // chunk origin (blocks)
layout(location = 7) flat out uint vFlags;     // MATERIAL_* bits
layout(location = 8) flat out vec3 vTopColor;  // linear top colour (grass overhang on sides)

void main() {
  uint faceIndex = uint(gl_VertexIndex) >> 2u;
  uint k = uint(gl_VertexIndex) & 3u;
  ChunkGpu ch = chunks[gl_InstanceIndex];
  Face f = decodeFace(faces[faceIndex]);

  uint c = quadCorner(k, f.dir, f.ao);
  vec3 local = cornerPosition(f, c);
  ivec3 origin = ivec3(ch.originX, ch.originY, ch.originZ);
  // Camera-relative: small integer difference first, then the fraction.
  vec3 rel = vec3(origin - frame.camBlock.xyz) - frame.camFrac.xyz + local;

  MaterialGpu m = materials[f.material];
  if ((m.flags & MATERIAL_FOLIAGE) != 0u) rel += foliageSway(vec3(origin) + local, frame.camFrac.w);
  vFlags = m.flags;
  vTopColor = srgbToLinear(unpackRGBA(m.top).rgb);
  vec4 col = unpackRGBA(materialColor(m, f.dir));
  vColor = vec4(srgbToLinear(col.rgb), col.a * m.alpha);
  vLight = vec4(float(f.sky) / 15.0, float(f.blockLight) / 15.0, m.emissive, 1.0);
  vNormal = kFaceNormals[f.dir];
  vAo = float(aoAt(f.ao, c));
  vViewPos = rel;
  vLocal = local;
  vOrigin = origin;
  gl_Position = frame.viewProj * vec4(rel, 1.0);
}
