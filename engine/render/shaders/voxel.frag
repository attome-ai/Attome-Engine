#version 460
#extension GL_GOOGLE_include_directive : require
// Shared by chunks (opaque + translucent) and models: Trove-style stylised
// look. Warm sun + cool sky / warm ground hemisphere ambient, per-face-direction
// shading, per-corner AO, soft large-scale colour patches (not a per-block
// grid), rim light on models, emissive (feeds bloom), sun-tinted distance fog.
#define NO_SCENE_BUFFERS
#include "voxel_common.glsl"

layout(location = 0) flat in vec4 vColor;
layout(location = 1) flat in vec4 vLight;
layout(location = 2) flat in vec3 vNormal;
layout(location = 3) in float vAo;
layout(location = 4) in vec3 vViewPos;
layout(location = 5) in vec3 vLocal;
layout(location = 6) flat in ivec3 vOrigin;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 9) uniform sampler2DShadow shadowMap;

// Sun visibility 0..1: normal-offset lookup, 3x3 taps of hardware 2x2 PCF
// (soft edges), faded out towards the edge of the shadow map's area.
float sunShadow(vec3 p, vec3 n, float ndl) {
  if (frame.shadowParams.y <= 0.0 || ndl <= 0.0) return 1.0;
  float texel = frame.shadowParams.x;
  vec4 ls = frame.lightViewProj * vec4(p + n * (texel * 1.5), 1.0);
  vec2 uv = ls.xy * 0.5 + 0.5;
  float edge = max(abs(ls.x), abs(ls.y));
  if (edge >= 1.0 || ls.z >= 1.0) return 1.0;
  vec2 ts = vec2(1.0 / float(textureSize(shadowMap, 0).x));
  float sum = 0.0;
  for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
      sum += texture(shadowMap, vec3(uv + vec2(x, y) * ts * 1.5, ls.z));
  float vis = sum / 9.0;
  return mix(vis, 1.0, smoothstep(0.85, 1.0, edge));
}

uint hash3(uvec3 v) {
  uint h = v.x * 0x8da6b343u ^ v.y * 0xd8163841u ^ v.z * 0xcb1ab31fu;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  return h;
}

float hashF(ivec3 c) { return float(hash3(uvec3(c)) & 1023u) * (1.0 / 1023.0); }

// Smooth 3D value noise in [0, 1].
float valueNoise(vec3 p) {
  ivec3 i = ivec3(floor(p));
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = mix(mix(hashF(i), hashF(i + ivec3(1, 0, 0)), f.x),
                mix(hashF(i + ivec3(0, 1, 0)), hashF(i + ivec3(1, 1, 0)), f.x), f.y);
  float b = mix(mix(hashF(i + ivec3(0, 0, 1)), hashF(i + ivec3(1, 0, 1)), f.x),
                mix(hashF(i + ivec3(0, 1, 1)), hashF(i + ivec3(1, 1, 1)), f.x), f.y);
  return mix(a, b, f.z);
}

void main() {
  vec3 base = vColor.rgb;
  vec3 n = normalize(vNormal);
  bool isChunk = vLight.w > 0.5;

  if (isChunk) {
    // Painterly colour patches across the world (lighter / warmer and
    // darker / cooler areas) instead of a noisy per-block checkerboard.
    vec3 wp = vec3(vOrigin) + vLocal;
    float patchN = valueNoise(wp * (1.0 / 14.0)) * 0.65 + valueNoise(wp * (1.0 / 5.0)) * 0.35;
    float s = patchN - 0.5;
    base *= 1.0 + s * 0.28;
    base = mix(base, base * vec3(1.06, 1.03, 0.9), clamp(s * 1.5, 0.0, 1.0)); // sunny patches warmer
    // Faint per-block variation (+-2%) so flat areas are not perfectly uniform.
    ivec3 cell = vOrigin + ivec3(floor(vLocal - n * 0.5));
    base *= 1.0 + (float(hash3(uvec3(cell)) & 255u) * (1.0 / 255.0) - 0.5) * 0.04;
  }

  float ao = mix(0.3, 1.0, clamp(vAo / 3.0, 0.0, 1.0));
  ao = ao * ao * (3.0 - 2.0 * ao) * 0.5 + ao * 0.5; // soften the ramp

  vec3 L = frame.sunDir.xyz;
  float ndl = max(dot(n, L), 0.0);
  float sunI = frame.fogParams.z;
  float skyL = lightCurve(vLight.x);
  float blockL = lightCurve(vLight.y) * step(0.001, vLight.y);

  // Hemisphere ambient: cool light from the sky dome, warm bounce from below.
  vec3 skyAmb = mix(vec3(1.0), frame.skyColor.rgb, 0.5) * 0.95;
  vec3 groundAmb = vec3(0.6, 0.52, 0.42) * max(frame.skyColor.b, 0.1);
  vec3 ambient = frame.sunDir.w * mix(groundAmb, skyAmb, n.y * 0.5 + 0.5);
  // Stylised face shading: top 1.0, X sides 0.84, Z sides 0.78, bottom 0.62.
  float faceShade = 0.84 + 0.16 * max(n.y, 0.0) - 0.22 * max(-n.y, 0.0) - 0.06 * abs(n.z);

  float shadow = sunShadow(vViewPos, n, ndl);
  vec3 light = skyL * (ambient * faceShade + frame.sunColor.rgb * (sunI * ndl * 1.05 * shadow)) +
               blockL * vec3(1.0, 0.78, 0.52) * 1.2;
  light = max(light, vec3(0.025)) * ao;

  vec3 col = base * light;

  float dist = length(vViewPos);
  vec3 viewDir = vViewPos / max(dist, 1e-4);

  if (!isChunk) {
    // Rim light: characters and props pop against the world (Trove-like).
    float rim = pow(1.0 - max(dot(n, -viewDir), 0.0), 3.0);
    vec3 rimCol = mix(frame.skyColor.rgb, frame.sunColor.rgb, 0.5) * (0.35 + 0.65 * sunI);
    col += rimCol * rim * 0.55 * skyL;
  }

  col += base * (vLight.z * 3.0);

  float fog = clamp((dist - frame.fogColor.w) * frame.fogParams.y, 0.0, 1.0);
  fog = fog * fog * (3.0 - 2.0 * fog);
  // Aerial perspective: fog glows warm towards the sun (matches sky.frag).
  float sunGlow = pow(max(dot(viewDir, L), 0.0), 6.0);
  vec3 fogCol = frame.fogColor.rgb + frame.sunColor.rgb * (sunI * 0.25 * sunGlow);
  col = mix(col, fogCol, fog);

  outColor = vec4(col, vColor.a);
}
