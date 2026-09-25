#version 460
#extension GL_GOOGLE_include_directive : require
// Sun shadow map, chunk faces: same vertex pulling as voxel.vert, projected
// with the light matrix. Depth only (no fragment shader). Draws are CPU
// written: vertexOffset = first opaque face * 4, firstInstance = chunk slot.
#include "voxel_common.glsl"

void main() {
  uint faceIndex = uint(gl_VertexIndex) >> 2u;
  uint k = uint(gl_VertexIndex) & 3u;
  ChunkGpu ch = chunks[gl_InstanceIndex];
  Face f = decodeFace(faces[faceIndex]);
  vec3 local = cornerPosition(f, quadCorner(k, f.dir, f.ao));
  ivec3 origin = ivec3(ch.originX, ch.originY, ch.originZ);
  vec3 rel = vec3(origin - frame.camBlock.xyz) - frame.camFrac.xyz + local;
  gl_Position = frame.lightViewProj * vec4(rel, 1.0);
}
