#version 460
#extension GL_GOOGLE_include_directive : require
// Voxel model parts (characters, props): same PackedFace format, faces in a
// separate model face buffer, one instance per placed part.
//   vertexOffset  = mesh first face * 4, firstInstance = first instance.
#include "voxel_common.glsl"

struct ModelInstanceGpu {
  mat4 model;          // part voxel space -> camera-relative blocks
  uint tint;
  uint paletteOffset;
  uint flags;         // kInstanceNoRim = 1, kInstanceNoShadow = 2
  uint pad1;
};
layout(std430, set = 0, binding = 7) readonly buffer ModelFaces { uvec2 modelFaces[]; };
layout(std430, set = 0, binding = 8) readonly buffer Instances { ModelInstanceGpu instances[]; };

layout(location = 0) flat out vec4 vColor;
layout(location = 1) flat out vec4 vLight;
layout(location = 2) flat out vec3 vNormal;
layout(location = 3) out float vAo;
layout(location = 4) out vec3 vViewPos;
layout(location = 5) out vec3 vLocal;
layout(location = 6) flat out ivec3 vOrigin;
layout(location = 7) flat out uint vFlags;
layout(location = 8) flat out vec3 vTopColor;

void main() {
  uint faceIndex = uint(gl_VertexIndex) >> 2u;
  uint k = uint(gl_VertexIndex) & 3u;
  ModelInstanceGpu inst = instances[gl_InstanceIndex];
  Face f = decodeFace(modelFaces[faceIndex]);

  uint c = quadCorner(k, f.dir, f.ao);
  vec3 local = cornerPosition(f, c);
  vec3 rel = (inst.model * vec4(local, 1.0)).xyz;

  uint matIndex = min(f.material + inst.paletteOffset, 65535u);
  MaterialGpu m = materials[matIndex];
  vec4 col = unpackRGBA(materialColor(m, f.dir)) * unpackRGBA(inst.tint);
  vColor = vec4(srgbToLinear(col.rgb), col.a * m.alpha);
  // Parts are meshed standalone: they get full sky light, plus their own
  // block light (glowing gear) if the mesher set any.
  vLight = vec4(1.0, float(f.blockLight) / 15.0, m.emissive, 0.0);
  vNormal = normalize(mat3(inst.model) * kFaceNormals[f.dir]);
  vAo = float(aoAt(f.ao, c));
  vViewPos = rel;
  vLocal = local;
  vOrigin = ivec3(0);
  vFlags = (inst.flags & 1u) != 0u ? MODEL_NO_RIM : 0u;
  vTopColor = vec3(0.0);
  gl_Position = frame.viewProj * vec4(rel, 1.0);
}
