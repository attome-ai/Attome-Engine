#version 460
#extension GL_GOOGLE_include_directive : require
// Sky gradient: fog colour at the horizon (so distance fog blends into the
// sky seamlessly) -> sky colour at the zenith, plus a sun disc + glow.
#define NO_SCENE_BUFFERS
#include "voxel_common.glsl"

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  // z = 1 is the near plane in reverse-Z (finite w for the infinite projection).
  vec4 p = frame.invViewProj * vec4(ndc, 1.0, 1.0);
  vec3 dir = normalize(p.xyz / p.w);

  vec3 horizon = frame.fogColor.rgb;
  vec3 zenith = frame.skyColor.rgb;
  float up = clamp(dir.y, -1.0, 1.0);
  vec3 col = mix(horizon, zenith, smoothstep(0.0, 0.55, up));
  // smoothstep needs edge0 < edge1 (undefined otherwise): mirrored form.
  col = mix(col, horizon * 0.75, 1.0 - smoothstep(-0.4, 0.0, up)); // below the horizon

  float s = max(dot(dir, frame.sunDir.xyz), 0.0);
  float sunI = frame.fogParams.z;
  col += frame.sunColor.rgb * sunI * (pow(s, 900.0) * 12.0 + pow(s, 12.0) * 0.18);
  outColor = vec4(col, 1.0);
}
