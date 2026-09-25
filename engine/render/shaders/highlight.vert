#version 460
#extension GL_GOOGLE_include_directive : require
// Targeted-block outline: 12 box edges as a line list (24 vertices, no
// vertex buffer). Box corners come from push constants (camera-relative).
#define NO_SCENE_BUFFERS
#include "voxel_common.glsl"

layout(push_constant) uniform Push {
  vec4 minCorner;
  vec4 maxCorner;
  vec4 color;
} pc;

// Corner bits: 1 = x max, 2 = y max, 4 = z max.
const uint kEdges[24] = uint[24](0u, 1u, 1u, 3u, 3u, 2u, 2u, 0u,   // bottom (z min)
                                 4u, 5u, 5u, 7u, 7u, 6u, 6u, 4u,   // top (z max)
                                 0u, 4u, 1u, 5u, 2u, 6u, 3u, 7u);  // verticals

void main() {
  uint c = kEdges[gl_VertexIndex];
  vec3 p = vec3((c & 1u) != 0u ? pc.maxCorner.x : pc.minCorner.x,
                (c & 2u) != 0u ? pc.maxCorner.y : pc.minCorner.y,
                (c & 4u) != 0u ? pc.maxCorner.z : pc.minCorner.z);
  gl_Position = frame.viewProj * vec4(p, 1.0);
}
