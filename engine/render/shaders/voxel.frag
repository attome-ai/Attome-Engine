#version 460
#extension GL_GOOGLE_include_directive : require
// Shared by chunks (opaque + translucent) and models: flat Trove-style colour,
// sun N.L + ambient, sky/block light, per-corner AO, subtle per-block colour
// noise, emissive (feeds bloom), distance fog.
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

uint hash3(uvec3 v) {
  uint h = v.x * 0x8da6b343u ^ v.y * 0xd8163841u ^ v.z * 0xcb1ab31fu;
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  return h;
}

void main() {
  vec3 base = vColor.rgb;
  vec3 n = normalize(vNormal);

  if (vLight.w > 0.5) {
    // Per-block variation (+-4%), from the integer block position.
    ivec3 cell = vOrigin + ivec3(floor(vLocal - n * 0.5));
    uint h = hash3(uvec3(cell));
    base *= 1.0 + (float(h & 255u) * (1.0 / 255.0) - 0.5) * 0.08;
  }

  float ao = mix(0.45, 1.0, clamp(vAo / 3.0, 0.0, 1.0));
  ao = ao * ao * (3.0 - 2.0 * ao) * 0.35 + ao * 0.65; // soften the ramp

  vec3 L = frame.sunDir.xyz;
  float ndl = max(dot(n, L), 0.0);
  float sunI = frame.fogParams.z;
  float skyL = lightCurve(vLight.x);
  float blockL = lightCurve(vLight.y) * step(0.001, vLight.y);
  vec3 ambient = frame.sunDir.w * mix(vec3(0.75), frame.skyColor.rgb, 0.5);
  // Faces pointing down get a little less ambient (sky dome is above).
  ambient *= 0.8 + 0.2 * (n.y * 0.5 + 0.5);
  vec3 light = skyL * (ambient + frame.sunColor.rgb * (sunI * ndl)) +
               blockL * vec3(1.0, 0.8, 0.58) * 1.15;
  light = max(light, vec3(0.025)) * ao;

  vec3 col = base * light + base * (vLight.z * 3.0);

  float dist = length(vViewPos);
  float fog = clamp((dist - frame.fogColor.w) * frame.fogParams.y, 0.0, 1.0);
  fog = fog * fog * (3.0 - 2.0 * fog);
  col = mix(col, frame.fogColor.rgb, fog);

  outColor = vec4(col, vColor.a);
}
