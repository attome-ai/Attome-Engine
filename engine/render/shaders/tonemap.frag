#version 460
// HDR -> display: ambient occlusion (blurred half-res SSAO), sun shafts
// (radial march over the sky mask toward the sun), bloom add, vibrance, ACES
// fit (Narkowicz), contrast + vignette grade, gamma unless the swapchain
// format is already sRGB.
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrTex;
layout(set = 0, binding = 1) uniform sampler2D bloomTex;
layout(set = 0, binding = 2) uniform sampler2D depthTex; // reverse-Z, 0 = sky
layout(set = 0, binding = 3) uniform sampler2D aoTex;    // half res, 1 = open

layout(push_constant) uniform Push {
  float exposure;
  float bloomStrength;
  uint bloomEnabled;
  uint srgbOutput;
  vec2 sunUv;          // sun position on screen (may be off screen)
  float shaftStrength; // 0 = off (sun behind the camera / night)
  float aoStrength;
  vec4 sunColor;       // linear rgb
} pc;

vec3 aces(vec3 x) {
  const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float sunShafts(vec2 uv) {
  const int kSteps = 48;
  vec2 delta = (pc.sunUv - uv) / float(kSteps);
  vec2 p = uv;
  float decay = 1.0, sum = 0.0;
  float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  p += delta * jitter;
  for (int i = 0; i < kSteps; ++i) {
    p += delta;
    if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) break;
    sum += (textureLod(depthTex, p, 0.0).r <= 0.0 ? 1.0 : 0.0) * decay;
    decay *= 0.955;
  }
  float falloff = 1.0 - smoothstep(0.0, 0.85, length((uv - pc.sunUv) * vec2(1.6, 1.0)));
  return sum / float(kSteps) * falloff;
}

void main() {
  vec3 col = textureLod(hdrTex, vUv, 0.0).rgb;

  // SSAO: 4 bilinear taps around the pixel = a soft 4x4 blur of the half-res AO.
  if (pc.aoStrength > 0.0) {
    vec2 ts = 1.0 / vec2(textureSize(aoTex, 0));
    float ao = 0.25 * (textureLod(aoTex, vUv + ts * vec2(-0.75, -0.75), 0.0).r +
                       textureLod(aoTex, vUv + ts * vec2(0.75, -0.75), 0.0).r +
                       textureLod(aoTex, vUv + ts * vec2(-0.75, 0.75), 0.0).r +
                       textureLod(aoTex, vUv + ts * vec2(0.75, 0.75), 0.0).r);
    col *= mix(1.0, ao, pc.aoStrength);
  }

  if (pc.bloomEnabled != 0u) col += textureLod(bloomTex, vUv, 0.0).rgb * pc.bloomStrength;
  if (pc.shaftStrength > 0.0) col += pc.sunColor.rgb * (sunShafts(vUv) * pc.shaftStrength);

  // Grade for a vivid, stylised look: ACES desaturates brights, so add
  // vibrance first (boosts muted colours more than already-saturated ones,
  // so grass does not go neon), then a gentle S-curve and a soft vignette.
  float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
  float mx = max(col.r, max(col.g, col.b)), mn = min(col.r, min(col.g, col.b));
  float sat = (mx - mn) / max(mx, 1e-4);
  col = max(mix(vec3(l), col, 1.0 + 0.35 * (1.0 - sat)), vec3(0.0));
  col = aces(col * pc.exposure);
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.55); // punchier contrast
  vec2 v = vUv - 0.5;
  col *= mix(0.8, 1.0, smoothstep(0.85, 0.3, length(v * vec2(1.1, 1.0))));
  if (pc.srgbOutput == 0u) col = pow(col, vec3(1.0 / 2.2));
  outColor = vec4(col, 1.0);
}
