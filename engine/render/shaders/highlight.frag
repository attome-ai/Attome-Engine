#version 460
layout(push_constant) uniform Push {
  vec4 minCorner;
  vec4 maxCorner;
  vec4 color;
} pc;
layout(location = 0) out vec4 outColor;
void main() { outColor = pc.color; }
