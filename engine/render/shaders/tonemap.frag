#version 460
// HDR -> display: exposure, bloom add, saturation, ACES fit (Narkowicz),
// contrast + vignette grade, gamma unless the
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
  // Grade for a vivid, stylised look: ACES desaturates brights, so add
  // vibrance first (boosts muted colours more than already-saturated ones,
  // so grass does not go neon), then a gentle S-curve and a soft vignette.
  float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  float mx = max(col.r, max(col.g, col.b)), mn = min(col.r, min(col.g, col.b));
  float sat = (mx - mn) / max(mx, 1e-4);
  col = max(mix(vec3(l), col, 1.0 + 0.35 * (1.0 - sat)), vec3(0.0));
  col = aces(col * pc.exposure);
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.3);
  vec2 v = vUv - 0.5;
  col *= mix(0.8, 1.0, smoothstep(0.85, 0.3, length(v * vec2(1.1, 1.0))));
  if (pc.srgbOutput == 0u) col = pow(col, vec3(1.0 / 2.2));
  outColor = vec4(col, 1.0);
}
