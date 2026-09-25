#version 460
#extension GL_GOOGLE_include_directive : require
// Sun shadow map, model parts (characters, props): same as model.vert,
// projected with the light matrix. Depth only.
#include "voxel_common.glsl"

struct ModelInstanceGpu {
  mat4 model;
  uint tint;
  uint paletteOffset;
  uint flags;         // kInstanceNoRim = 1, kInstanceNoShadow = 2
  uint pad1;
};
layout(std430, set = 0, binding = 7) readonly buffer ModelFaces { uvec2 modelFaces[]; };
layout(std430, set = 0, binding = 8) readonly buffer Instances { ModelInstanceGpu instances[]; };

void main() {
  uint faceIndex = uint(gl_VertexIndex) >> 2u;
  uint k = uint(gl_VertexIndex) & 3u;
  ModelInstanceGpu inst = instances[gl_InstanceIndex];
  Face f = decodeFace(modelFaces[faceIndex]);
  vec3 local = cornerPosition(f, quadCorner(k, f.dir, f.ao));
  vec3 rel = (inst.model * vec4(local, 1.0)).xyz;
  gl_Position = frame.lightViewProj * vec4(rel, 1.0);
  if ((inst.flags & 2u) != 0u) gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // culled: outside the clip volume
}
