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
layout(location = 7) flat in uint vFlags;
layout(location = 8) flat in vec3 vTopColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 9) uniform sampler2DShadow shadowMap;

// Sun visibility 0..1: normal-offset lookup, 16-tap Poisson disk of hardware
// 2x2 PCF (soft, round penumbrae like Trove, no grid pattern), faded out
// towards the edge of the shadow map's area.
const vec2 kPoisson[16] = vec2[16](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725), vec2(-0.09418410, -0.92938870),
    vec2(0.34495938, 0.29387760), vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379), vec2(0.44323325, -0.97511554),
    vec2(0.53742981, -0.47373420), vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590), vec2(0.19984126, 0.78641367),
    vec2(0.14383161, -0.14100790));

layout(set = 0, binding = 10) uniform sampler2D shadowDepth; // same map, raw depth

// Contact-hardening (PCSS-style): find the average blocker depth, then
// widen the filter with the distance between blocker and receiver. Shadows
// are sharp and dark where things touch (feet on the ground) and soften
// further out (the far end of a tree or character shadow).
float sunShadow(vec3 p, vec3 n, float ndl) {
  if (frame.shadowParams.y <= 0.0 || ndl <= 0.0) return 1.0;
  float texel = frame.shadowParams.x;
  float depthRange = frame.shadowParams.z;            // blocks per unit of shadow depth
  vec4 ls = frame.lightViewProj * vec4(p + n * (texel * 2.0), 1.0);
  // Only the top-left counts.z^2 of the map is used at lower quality settings.
  float mapSize = float(textureSize(shadowMap, 0).x);
  float uvScale = frame.counts.z > 0u ? float(frame.counts.z) / mapSize : 1.0;
  vec2 uv = (ls.xy * 0.5 + 0.5) * uvScale;
  float edge = max(abs(ls.x), abs(ls.y));
  if (edge >= 1.0 || ls.z >= 1.0) return 1.0;
  vec2 ts = vec2(1.0 / float(textureSize(shadowMap, 0).x));
  float softness = frame.style0.w;

  // Simple mode (default): one even soft filter everywhere.
  if (frame.shadowParams.w < 0.5) {
    float sum = 0.0;
    for (int i = 0; i < 16; ++i)
      sum += texture(shadowMap, vec3(uv + kPoisson[i] * ts * (3.5 * softness), ls.z));
    float vis = sum / 16.0;
    vis = vis * vis * (3.0 - 2.0 * vis);
    return mix(vis, 1.0, smoothstep(0.85, 1.0, edge));
  }

  // Contact-hardening mode.
  // 1. Blocker search.
  float blockSum = 0.0, blockN = 0.0;
  float bias = 0.12 / depthRange;                     // ~0.12 blocks
  for (int i = 0; i < 16; ++i) {
    float d = textureLod(shadowDepth, uv + kPoisson[i] * ts * 9.0, 0.0).r;
    if (d < ls.z - bias) { blockSum += d; blockN += 1.0; }
  }
  if (blockN < 0.5) return 1.0;                       // fully lit
  float blockerDist = (ls.z - blockSum / blockN) * depthRange; // blocks

  // 2. Penumbra grows with blocker distance (stylised: wider than a real sun).
  float penumbra = clamp((0.025 + blockerDist * 0.06) * softness, 0.02, 0.6); // blocks
  float radius = clamp(penumbra / texel, 1.0, 16.0);  // texels

  // 3. Filter.
  float sum = 0.0;
  for (int i = 0; i < 16; ++i)
    sum += texture(shadowMap, vec3(uv + kPoisson[i] * ts * radius, ls.z));
  float vis = sum / 16.0;
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

// Gradient of one travelling sine wave a*sin(dot(p, dir)*k + t*speed).
vec2 waveGrad(vec2 p, vec2 dir, float k, float speed, float a, float t) {
  return dir * (k * a * cos(dot(p, dir) * k + t * speed));
}

void main() {
  vec3 base = vColor.rgb;
  vec3 n = normalize(vNormal);
  bool isChunk = vLight.w > 0.5;

  float tileShade = 1.0;
  if (isChunk && (vFlags & MATERIAL_WATER) == 0u) {
    // Trove-style crafted blocks:
    //  1. broad painterly patches (sunny / shaded areas across the land),
    //  2. every block its own shade, so individual tiles read,
    //  3. fine sub-voxel grain on each face (4x4 "pixels" per block),
    //  4. soft bevel: block edges darken, the upper-left inner edge catches light.
    // Detail fades out once a block is only a few pixels on screen.
    vec3 wp = vec3(vOrigin) + vLocal;
    float patchN = valueNoise(wp * (1.0 / 16.0)) * 0.65 + valueNoise(wp * (1.0 / 6.0)) * 0.35;
    float s = patchN - 0.5;
    base *= 1.0 + s * 0.22 * frame.style2.x;
    base = mix(base, base * vec3(1.05, 1.02, 0.9), clamp(s * 1.5 * frame.style2.x, 0.0, 1.0));

    ivec3 cell = vOrigin + ivec3(floor(vLocal - n * 0.5));
    uint h = hash3(uvec3(cell));
    float blockVar = (float(h & 255u) / 255.0 - 0.5) * 0.07 * frame.style1.w;

    vec2 uv = abs(n.x) > 0.5 ? wp.zy : (abs(n.y) > 0.5 ? wp.xz : wp.xy);
    vec2 f = fract(uv);
    vec2 fw = fwidth(uv);
    float px = max(fw.x, fw.y);                        // blocks per pixel
    float detail = 1.0 - smoothstep(0.08, 0.3, px);    // fade in the distance

    // Grass overhang (Trove signature): on the sides of grass blocks the top
    // colour drips down over the dirt with a jagged fringe (4 columns per
    // block), and the fringe casts a thin dark line onto the dirt below.
    if ((vFlags & MATERIAL_GRASSTOP) != 0u && abs(n.y) < 0.5) {
      int colI = int(floor(f.x * 4.0));
      uint hc = hash3(uvec3(cell) * 3u + uvec3(uint(colI), 17u, uint(n.x + 2.0 * n.z + 3.0)));
      float drip = 0.30 + float(hc & 3u) * 0.09;       // 0.30 .. 0.57 of the block height
      float fy = f.y;
      if (fy > drip) {
        base = vTopColor * 0.92;                       // sides are a touch darker
      } else if (fy > drip - 0.07) {
        base *= mix(1.0, 0.72, detail);
      }
      // In the distance a 1-block step is a few pixels tall and its dirt
      // band turns hills into brown contour stripes: fade sides to grass.
      float farGrass = 1.0 - smoothstep(0.02, 0.09, px);
      base = mix(vTopColor * 0.86, base, farGrass);
    }

    ivec2 sub = ivec2(floor(f * 4.0));
    uint hs = hash3(uvec3(cell) * 7u + uvec3(uint(sub.x), uint(sub.y), uint(dot(abs(n), vec3(1, 2, 3)))));
    float grain = (float(hs & 255u) / 255.0 - 0.5) * 0.08 * frame.style1.z;

    float e = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
    float edge = 1.0 - smoothstep(0.0, 0.06 + px, e);  // 1 at the block edge
    // Light from the upper-left in face space: the top/left inner rim is lit.
    float lit = (f.x < 0.12 || f.y > 0.88) ? 1.0 : 0.0;
    float bevel = edge * (lit * 0.13 - 0.08) * frame.style1.y; // lit rim +5%, shadow rim -8%

    if ((vFlags & MATERIAL_FOLIAGE) != 0u) {
      // Leaves: no tile bevel (reads as a Minecraft grid); instead a mottled
      // two-tone canopy (2x2 clumps per face), lighter tops, darker undersides.
      ivec2 clump = ivec2(floor(f * 2.0));
      uint hl = hash3(uvec3(cell) * 5u + uvec3(uint(clump.x), uint(clump.y), 9u));
      grain = (float(hl & 255u) / 255.0 - 0.5) * 0.16;
      bevel = 0.0;
      blockVar *= 1.8;
      base *= 1.0 + 0.07 * n.y;
    }
    base *= 1.0 + blockVar;
    tileShade = 1.0 + (grain + bevel) * detail;

    // Wet shoreline: surfaces at sea level (64) are darker and less saturated
    // (wet sand / mud), fading out one block above the water.
    float wet = 1.0 - smoothstep(64.3, 65.4, wp.y);
    if (wet > 0.0) {
      float lw = dot(base, vec3(0.2126, 0.7152, 0.0722));
      base = mix(base, mix(vec3(lw), base, 0.8) * 0.68, wet);
    }
  }

  float ao = mix(1.0 - frame.style1.x, 1.0, clamp(vAo / 3.0, 0.0, 1.0)); // voxel corners
  ao = ao * ao * (3.0 - 2.0 * ao) * 0.5 + ao * 0.5; // soften the ramp

  vec3 L = frame.sunDir.xyz;
  float ndl = max(dot(n, L), 0.0);
  // Wrap lighting for models and leaves: their back sides get soft sun
  // instead of going black when the camera faces the sun.
  bool wrapLit = !isChunk || (vFlags & MATERIAL_FOLIAGE) != 0u;
  float ndlDiffuse = wrapLit ? clamp((dot(n, L) + 0.45) / 1.45, 0.0, 1.0) : ndl;
  float sunI = frame.fogParams.z;
  float skyL = lightCurve(vLight.x);
  float blockL = lightCurve(vLight.y) * step(0.001, vLight.y);

  // Hemisphere ambient: cool light from the sky dome, warm bounce from below.
  // Cool, slightly purple sky light (reads as shade vs the warm sun).
  vec3 skyAmb = mix(vec3(1.0), frame.skyColor.rgb, 0.72) * vec3(0.93, 0.92, 1.02);
  vec3 groundAmb = vec3(0.6, 0.52, 0.42) * max(frame.skyColor.b, 0.1);
  vec3 ambient = frame.sunDir.w * mix(groundAmb, skyAmb, n.y * 0.5 + 0.5);
  // Characters / props: a bit more sky fill so they sit in the scene instead
  // of looking pasted on (dark clothes in their own shadow otherwise go black).
  if (!isChunk) ambient *= 1.3;
  // Stylised face shading: top 1.0, X sides 0.84, Z sides 0.78, bottom 0.62.
  float faceShade = 0.84 + 0.16 * max(n.y, 0.0) - 0.22 * max(-n.y, 0.0) - 0.06 * abs(n.z);

  float shadow = sunShadow(vViewPos, n, ndlDiffuse);
  vec3 light = skyL * (ambient * faceShade + frame.sunColor.rgb * (sunI * ndlDiffuse * frame.style0.x * shadow)) +
               blockL * vec3(1.0, 0.78, 0.52) * 1.2;
  light = max(light, vec3(0.025)) * ao;

  vec3 col = base * light * tileShade;

  float dist = length(vViewPos);
  vec3 viewDir = vViewPos / max(dist, 1e-4);

  if (!isChunk && (vFlags & MODEL_NO_RIM) == 0u) {
    // Rim light: characters and props pop against the world (Trove-like).
    float rim = pow(1.0 - max(dot(n, -viewDir), 0.0), 3.0);
    vec3 rimCol = mix(frame.skyColor.rgb, frame.sunColor.rgb, 0.5) * (0.35 + 0.65 * sunI);
    col += rimCol * rim * frame.style2.w * skyL;
  }

  col += base * (vLight.z * 3.0);

  // Leaves glow when the sun shines through them toward the viewer.
  if ((vFlags & MATERIAL_FOLIAGE) != 0u) {
    float through = pow(max(dot(viewDir, L), 0.0), 4.0);
    col += base * frame.sunColor.rgb * (sunI * through * frame.style2.z * skyL);
  }

  float alpha = vColor.a;
  if ((vFlags & MATERIAL_WATER) != 0u && n.y > 0.5) {
    // Water surface: animated wave normals (sum of travelling sines), sky
    // reflection weighted by Fresnel, and a sharp sun glint that blooms.
    vec2 p = (vec3(vOrigin) + vLocal).xz;
    float t = frame.camFrac.w;
    vec2 g = vec2(0.0);
    g += waveGrad(p, normalize(vec2(0.8, 0.6)), 0.55, 1.1, 0.10, t);
    g += waveGrad(p, normalize(vec2(-0.5, 0.9)), 1.05, 1.7, 0.05, t);
    g += waveGrad(p, normalize(vec2(0.2, -1.0)), 2.3, 2.6, 0.022, t);
    g += waveGrad(p, normalize(vec2(-0.9, -0.3)), 4.1, 3.3, 0.010, t);
    vec3 wn = normalize(vec3(-g.x, 1.0, -g.y));
    float cosV = max(dot(wn, -viewDir), 0.0);
    float fresnel = 0.03 + 0.97 * pow(1.0 - cosV, 5.0);
    vec3 r = reflect(viewDir, wn);
    vec3 horizon = frame.fogColor.rgb + frame.sunColor.rgb * (sunI * 0.25 * pow(max(dot(r, L), 0.0), 6.0));
    vec3 zenith = frame.skyColor.rgb * vec3(0.75, 0.85, 1.0);
    vec3 refl = mix(horizon, zenith, pow(smoothstep(0.0, 0.7, max(r.y, 0.0)), 0.7)) * skyL * vec3(0.55, 0.75, 0.95);
    vec3 body = col * vec3(0.30, 0.68, 0.82);          // deep turquoise
    col = mix(body, refl, clamp(fresnel * frame.style2.y + 0.06, 0.0, 1.0));
    float spec = pow(max(dot(r, L), 0.0), 350.0) * 9.0 + pow(max(dot(r, L), 0.0), 40.0) * 0.25;
    col += frame.sunColor.rgb * (sunI * spec * shadow);
    alpha = mix(0.82, 0.97, fresnel);
  } else if ((vFlags & MATERIAL_WATER) != 0u) {
    // Water side / bottom faces: same deep tint, mostly opaque.
    col *= vec3(0.30, 0.68, 0.82);
    alpha = 0.85;
  }

  float fog = clamp((dist - frame.fogColor.w) * frame.fogParams.y, 0.0, 1.0);
  fog = fog * fog * (3.0 - 2.0 * fog);
  // Aerial perspective: fog glows warm towards the sun (matches sky.frag).
  float sunGlow = pow(max(dot(viewDir, L), 0.0), 6.0);
  vec3 fogCol = frame.fogColor.rgb + frame.sunColor.rgb * (sunI * 0.25 * sunGlow);
  // Gentle haze from near range on: distant hills lose contrast and turn
  // bluish (depth), and valleys below the camera hold a little mist.
  float worldY = float(frame.camBlock.y) + frame.camFrac.y + vViewPos.y;
  float valley = exp(-max(worldY - 60.0, 0.0) * 0.08);
  float haze = (1.0 - exp(-dist * frame.style0.z * (1.0 + valley))) * frame.style0.y;
  // Atmospheric perspective: distance first loses saturation, then turns
  // toward the (bluish) haze colour; the foreground stays crisp.
  col = mix(col, vec3(dot(col, vec3(0.2126, 0.7152, 0.0722))), haze * 0.4);
  col = mix(col, fogCol, haze);
  col = mix(col, fogCol, fog);

  // Underwater camera: absorb reds, thick blue-green fog within ~20 blocks.
  if (frame.fogParams.w > 0.5) {
    float uw = 1.0 - exp(-dist * 0.1);
    col = mix(col * vec3(0.55, 0.82, 0.95), vec3(0.03, 0.17, 0.27), uw);
  }

  outColor = vec4(col, alpha);
}
