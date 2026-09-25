#version 460
// HDR -> display: exposure, bloom add, ACES fit (Narkowicz), gamma unless the
// swapchain format is already sRGB.
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;

layout(push_constant) uniform Push {
  float exposure;
  float bloomStrength;
  uint bloomEnabled;
  uint srgbOutput;
} pc;

vec3 aces(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 col = textureLod(hdrTex, vUv, 0.0).rgb;
  if (pc.bloomEnabled != 0u) col += textureLod(bloomTex, vUv, 0.0).rgb * pc.bloomStrength;
  col = aces(col * pc.exposure);
  if (pc.srgbOutput == 0u) col = pow(col, vec3(1.0 / 2.2));
  outColor = vec4(col, 1.0);
}
