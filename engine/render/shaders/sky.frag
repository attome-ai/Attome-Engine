#version 460
#extension GL_GOOGLE_include_directive : require
// Sky gradient: fog colour at the horizon (so distance fog blends into the
// sky seamlessly) -> deep sky colour at the zenith, soft drifting clouds, and
// a sun disc + glow.
#define NO_SCENE_BUFFERS
#include "voxel_common.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

float hash2(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float noise2(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2(1, 0)), f.x),
             mix(hash2(i + vec2(0, 1)), hash2(i + vec2(1, 1)), f.x), f.y);
}

float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; ++i) {
    v += a * noise2(p);
    p = p * 2.03 + vec2(17.1, 9.2);
    a *= 0.5;
  }
  return v;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  // z = 1 is the near plane in reverse-Z (finite w for the infinite projection).
  vec4 p = frame.invViewProj * vec4(ndc, 1.0, 1.0);
  vec3 dir = normalize(p.xyz / p.w);

  float s = max(dot(dir, frame.sunDir.xyz), 0.0);
  float sunI = frame.fogParams.z;

  // Horizon matches voxel.frag's fog colour, including its sun glow.
  vec3 horizon = frame.fogColor.rgb + frame.sunColor.rgb * (sunI * 0.25 * pow(s, 6.0));
  vec3 zenith = frame.skyColor.rgb * vec3(0.75, 0.85, 1.0);
  float up = clamp(dir.y, -1.0, 1.0);
  vec3 col = mix(horizon, zenith, pow(smoothstep(0.0, 0.7, up), 0.7));
  // smoothstep needs edge0 < edge1 (undefined otherwise): mirrored form.
  col = mix(col, horizon * 0.75, 1.0 - smoothstep(-0.4, 0.0, up)); // below the horizon

  col += frame.sunColor.rgb * sunI * (pow(s, 900.0) * 14.0 + pow(s, 10.0) * 0.22);

  // Clouds on a plane high above the camera, drifting with time.
  if (dir.y > 0.0) {
    vec2 cp = dir.xz / (dir.y + 0.08) * 0.9 + vec2(frame.camFrac.w * 0.004, frame.camFrac.w * 0.0015);
    float d = fbm(cp * 1.4);
    float cover = smoothstep(0.52, 0.72, d);
    // Fake self-shadowing: denser cloud cores are a little darker underneath.
    float thick = smoothstep(0.6, 0.95, d);
    vec3 lit = frame.sunColor.rgb * sunI * 0.9 + frame.skyColor.rgb * 0.45 + vec3(0.08);
    vec3 cloudCol = mix(lit, lit * vec3(0.78, 0.82, 0.92), thick);
    cloudCol += frame.sunColor.rgb * sunI * pow(s, 8.0) * 0.6; // silver lining near the sun
    float fade = smoothstep(0.02, 0.25, dir.y);
    col = mix(col, cloudCol, cover * fade * 0.92);
  }

  if (frame.fogParams.w > 0.5) col = vec3(0.03, 0.17, 0.27) * (0.6 + 0.4 * max(dir.y, 0.0)); // underwater
  outColor = vec4(col, 1.0);
}
