// Tiny deterministic synthesiser for the demo sound effects: oscillators,
// seeded noise, envelopes, one-pole and state-variable filters, pitch sweeps
// and Karplus-Strong plucks. See Sfx.h.

#include "Sfx.h"

#include <algorithm>
#include <cmath>

namespace ao::audio {

namespace {

constexpr float kSr = float(kSfxSampleRate);
constexpr float kPi = 3.14159265358979f;
constexpr float kTau = 2.0f * kPi;

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
  uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float uni() { return float(next() & 0xFFFFFF) / float(0xFFFFFF); } // 0..1
  float noise() { return uni() * 2.0f - 1.0f; }                       // -1..1
};

int samplesFor(float seconds) { return std::max(1, int(seconds * kSr)); }

// Attack-then-exponential-decay envelope.
float envAD(float t, float attack, float decay) {
  if (t < 0.0f)
    return 0.0f;
  if (t < attack)
    return t / std::max(attack, 1e-5f);
  return std::exp(-(t - attack) / std::max(decay, 1e-5f));
}

// Smooth hump over [0, len] (sin^2).
float hump(float t, float len) {
  if (t <= 0.0f || t >= len)
    return 0.0f;
  const float s = std::sin(kPi * t / len);
  return s * s;
}

struct OnePoleLP {
  float y = 0.0f;
  float process(float x, float cutoffHz) {
    const float a = 1.0f - std::exp(-kTau * cutoffHz / kSr);
    y += a * (x - y);
    return y;
  }
};

struct OnePoleHP {
  OnePoleLP lp;
  float process(float x, float cutoffHz) { return x - lp.process(x, cutoffHz); }
};

// Chamberlin state-variable filter (bandpass / lowpass outputs).
struct Svf {
  float low = 0.0f, band = 0.0f;
  float bp(float x, float freq, float q) {
    const float f = 2.0f * std::sin(kPi * std::min(freq, kSr * 0.2f) / kSr);
    const float damp = 1.0f / std::max(q, 0.5f);
    low += f * band;
    const float high = x - low - damp * band;
    band += f * high;
    return band;
  }
};

struct Osc {
  float phase = 0.0f; // 0..1
  float step(float freq) {
    phase += freq / kSr;
    phase -= std::floor(phase);
    return phase;
  }
  float sine(float freq) { return std::sin(kTau * step(freq)); }
  float tri(float freq) {
    const float p = step(freq);
    return 4.0f * std::abs(p - 0.5f) - 1.0f;
  }
  float square(float freq) { return step(freq) < 0.5f ? 1.0f : -1.0f; }
  float saw(float freq) { return 2.0f * step(freq) - 1.0f; }
};

// Exponential pitch sweep from f0 to f1 over `len` seconds.
float sweep(float t, float len, float f0, float f1) {
  const float u = std::clamp(t / std::max(len, 1e-5f), 0.0f, 1.0f);
  return f0 * std::pow(f1 / f0, u);
}

void normalize(std::vector<float> &buf, float peak) {
  float m = 0.0f;
  for (float v : buf)
    m = std::max(m, std::abs(v));
  if (m > 1e-6f)
    for (float &v : buf)
      v *= peak / m;
}

// Short fade in/out to avoid clicks at the ends of one-shots.
void declick(std::vector<float> &buf) {
  const int n = std::min<int>(int(buf.size()) / 4, samplesFor(0.003f));
  for (int i = 0; i < n; ++i) {
    const float g = float(i) / float(n);
    buf[size_t(i)] *= g;
    buf[buf.size() - 1 - size_t(i)] *= g;
  }
}

// Makes a seamless loop of `len` samples from a buffer of len + fade samples
// by cross-fading the tail into the head (equal-power).
std::vector<float> loopify(const std::vector<float> &src, int len, int fade) {
  std::vector<float> out(src.begin(), src.begin() + len);
  for (int i = 0; i < fade; ++i) {
    const float u = float(i) / float(fade);
    const float a = std::cos(u * kPi * 0.5f), b = std::sin(u * kPi * 0.5f);
    out[size_t(i)] = src[size_t(len + i)] * a + src[size_t(i)] * b;
  }
  return out;
}

// Karplus-Strong plucked string.
std::vector<float> pluck(float freq, float seconds, float damping, uint32_t seed) {
  Rng rng(seed);
  const int n = samplesFor(seconds);
  const int period = std::max(2, int(kSr / freq));
  std::vector<float> ring(size_t(period));
  for (float &v : ring)
    v = rng.noise();
  std::vector<float> out(size_t(n));
  size_t idx = 0;
  for (int i = 0; i < n; ++i) {
    const size_t next = (idx + 1) % ring.size();
    const float v = ring[idx];
    ring[idx] = damping * 0.5f * (ring[idx] + ring[next]);
    out[size_t(i)] = v;
    idx = next;
  }
  return out;
}

// Noise grain burst through a bandpass (crunches, scrapes).
void addGrain(std::vector<float> &buf, Rng &rng, float start, float len, float freq, float q,
              float gain) {
  Svf f;
  const int s0 = samplesFor(start), n = samplesFor(len);
  for (int i = 0; i < n && size_t(s0 + i) < buf.size(); ++i) {
    const float t = float(i) / kSr;
    buf[size_t(s0 + i)] += f.bp(rng.noise(), freq, q) * envAD(t, 0.002f, len * 0.3f) * gain;
  }
}

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

std::vector<float> footstepGrass() {
  Rng rng(101);
  std::vector<float> b(size_t(samplesFor(0.14f)));
  OnePoleLP lp;
  OnePoleHP hp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    const float n = hp.process(lp.process(rng.noise(), 2200.0f), 300.0f);
    b[i] = n * (envAD(t, 0.004f, 0.03f) + 0.5f * envAD(t - 0.035f, 0.003f, 0.025f));
  }
  normalize(b, 0.55f);
  declick(b);
  return b;
}

std::vector<float> footstepStone() {
  Rng rng(102);
  std::vector<float> b(size_t(samplesFor(0.12f)));
  Svf bp;
  Osc thump;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = bp.bp(rng.noise(), 2600.0f, 3.0f) * envAD(t, 0.001f, 0.018f) +
           0.6f * thump.sine(sweep(t, 0.06f, 160.0f, 90.0f)) * envAD(t, 0.002f, 0.03f);
  }
  normalize(b, 0.55f);
  declick(b);
  return b;
}

std::vector<float> jump() {
  std::vector<float> b(size_t(samplesFor(0.2f)));
  Osc o, o2;
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    const float f = sweep(t, 0.14f, 280.0f, 720.0f);
    const float v = 0.6f * o.tri(f) + 0.25f * o2.square(f * 0.5f);
    b[i] = lp.process(v, 3000.0f) * envAD(t, 0.005f, 0.06f);
  }
  normalize(b, 0.5f);
  declick(b);
  return b;
}

std::vector<float> land() {
  Rng rng(104);
  std::vector<float> b(size_t(samplesFor(0.25f)));
  Osc o;
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = 0.9f * o.sine(sweep(t, 0.12f, 110.0f, 45.0f)) * envAD(t, 0.002f, 0.06f) +
           0.5f * lp.process(rng.noise(), 900.0f) * envAD(t, 0.001f, 0.04f);
  }
  normalize(b, 0.7f);
  declick(b);
  return b;
}

std::vector<float> glideWind() {
  // 3 s loop: bandpassed noise with a slow sweeping centre and swell; LFO
  // rates are whole cycles per loop so the modulation also loops.
  Rng rng(105);
  const float loopSec = 3.0f;
  const int len = samplesFor(loopSec), fade = samplesFor(0.25f);
  std::vector<float> src(size_t(len + fade));
  Svf f1, f2;
  for (size_t i = 0; i < src.size(); ++i) {
    const float t = float(i) / kSr;
    const float lfo = std::sin(kTau * t * (2.0f / loopSec));
    const float lfo2 = std::sin(kTau * t * (3.0f / loopSec) + 1.3f);
    const float n = rng.noise();
    const float v = f1.bp(n, 650.0f + 250.0f * lfo, 2.5f) + 0.5f * f2.bp(n, 1700.0f + 400.0f * lfo2, 4.0f);
    src[i] = v * (0.75f + 0.25f * lfo2);
  }
  std::vector<float> b = loopify(src, len, fade);
  normalize(b, 0.45f);
  return b;
}

std::vector<float> dash() {
  Rng rng(106);
  std::vector<float> b(size_t(samplesFor(0.32f)));
  Svf f;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    const float centre = t < 0.12f ? sweep(t, 0.12f, 400.0f, 2600.0f) : sweep(t - 0.12f, 0.2f, 2600.0f, 700.0f);
    b[i] = f.bp(rng.noise(), centre, 2.0f) * hump(t, 0.32f);
  }
  normalize(b, 0.6f);
  return b;
}

std::vector<float> swordSwing() {
  Rng rng(107);
  std::vector<float> b(size_t(samplesFor(0.26f)));
  Svf f;
  Osc ring;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = f.bp(rng.noise(), sweep(t, 0.2f, 700.0f, 3400.0f), 3.0f) * hump(t, 0.22f) +
           0.08f * ring.sine(1870.0f) * envAD(t, 0.05f, 0.08f);
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> bowTwang() {
  std::vector<float> s = pluck(98.0f, 0.4f, 0.996f, 108);
  std::vector<float> b(s.size());
  OnePoleLP lp;
  Osc thud;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = lp.process(s[i], 2500.0f) * envAD(t, 0.001f, 0.15f) +
           0.4f * thud.sine(sweep(t, 0.05f, 200.0f, 120.0f)) * envAD(t, 0.001f, 0.02f);
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> arrowHit() {
  Rng rng(109);
  std::vector<float> b(size_t(samplesFor(0.14f)));
  Osc o;
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = o.sine(sweep(t, 0.08f, 220.0f, 80.0f)) * envAD(t, 0.001f, 0.035f) +
           0.6f * lp.process(rng.noise(), 3000.0f) * envAD(t, 0.0005f, 0.008f);
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> monsterHit() {
  Rng rng(110);
  std::vector<float> b(size_t(samplesFor(0.22f)));
  Osc o;
  OnePoleLP lp, lp2;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    const float body = lp.process(o.saw(sweep(t, 0.15f, 220.0f, 80.0f)), 900.0f);
    b[i] = body * envAD(t, 0.002f, 0.07f) + 0.5f * lp2.process(rng.noise(), 1800.0f) * envAD(t, 0.001f, 0.02f);
  }
  normalize(b, 0.65f);
  declick(b);
  return b;
}

std::vector<float> playerHurt() {
  Rng rng(111);
  std::vector<float> b(size_t(samplesFor(0.32f)));
  Osc o, vib;
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    const float f = sweep(t, 0.25f, 340.0f, 200.0f) * (1.0f + 0.03f * vib.sine(28.0f));
    b[i] = lp.process(o.square(f), 1400.0f) * envAD(t, 0.008f, 0.1f) +
           0.2f * rng.noise() * envAD(t, 0.001f, 0.015f);
  }
  normalize(b, 0.55f);
  declick(b);
  return b;
}

std::vector<float> breakStone() {
  Rng rng(112);
  std::vector<float> b(size_t(samplesFor(0.38f)));
  addGrain(b, rng, 0.0f, 0.12f, 1300.0f, 1.8f, 1.0f);
  addGrain(b, rng, 0.04f, 0.1f, 2100.0f, 2.5f, 0.6f);
  addGrain(b, rng, 0.1f, 0.14f, 900.0f, 1.5f, 0.5f);
  addGrain(b, rng, 0.18f, 0.12f, 1600.0f, 2.0f, 0.3f);
  Osc o;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] += 0.35f * o.sine(sweep(t, 0.2f, 120.0f, 55.0f)) * envAD(t, 0.002f, 0.08f);
  }
  normalize(b, 0.7f);
  declick(b);
  return b;
}

std::vector<float> breakDirt() {
  Rng rng(113);
  std::vector<float> b(size_t(samplesFor(0.28f)));
  addGrain(b, rng, 0.0f, 0.12f, 500.0f, 0.8f, 1.0f);
  addGrain(b, rng, 0.05f, 0.12f, 750.0f, 0.9f, 0.6f);
  addGrain(b, rng, 0.12f, 0.12f, 400.0f, 0.8f, 0.4f);
  normalize(b, 0.65f);
  declick(b);
  return b;
}

std::vector<float> breakWood() {
  Rng rng(114);
  std::vector<float> knock = pluck(210.0f, 0.3f, 0.97f, 214);
  std::vector<float> b(knock.size());
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = lp.process(knock[i], 1600.0f) * envAD(t, 0.001f, 0.06f);
  }
  addGrain(b, rng, 0.0f, 0.08f, 950.0f, 3.0f, 0.5f);
  addGrain(b, rng, 0.07f, 0.1f, 700.0f, 2.0f, 0.35f);
  normalize(b, 0.7f);
  declick(b);
  return b;
}

std::vector<float> breakGlass() {
  Rng rng(115);
  std::vector<float> b(size_t(samplesFor(0.6f)));
  OnePoleHP hp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = 0.5f * hp.process(rng.noise(), 3000.0f) * envAD(t, 0.001f, 0.05f);
  }
  // Scattered high pings.
  for (int k = 0; k < 14; ++k) {
    const float start = rng.uni() * 0.25f;
    const float freq = 2200.0f + rng.uni() * 4200.0f;
    const float gain = 0.15f + rng.uni() * 0.25f;
    const float decay = 0.04f + 0.08f * rng.uni();
    Osc o;
    const int s0 = samplesFor(start);
    for (size_t i = size_t(s0); i < b.size(); ++i) {
      const float t = float(i - size_t(s0)) / kSr;
      b[i] += gain * o.sine(freq) * envAD(t, 0.0005f, decay);
    }
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> place() {
  Rng rng(116);
  std::vector<float> b(size_t(samplesFor(0.13f)));
  Osc o;
  OnePoleLP lp;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = o.sine(sweep(t, 0.08f, 170.0f, 105.0f)) * envAD(t, 0.001f, 0.035f) +
           0.4f * lp.process(rng.noise(), 2500.0f) * envAD(t, 0.0005f, 0.006f);
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> pickup() {
  std::vector<float> b(size_t(samplesFor(0.2f)));
  Osc a, c;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = a.sine(880.0f) * envAD(t, 0.002f, 0.04f) +
           c.sine(1320.0f) * envAD(t - 0.06f, 0.002f, 0.06f);
  }
  normalize(b, 0.45f);
  declick(b);
  return b;
}

std::vector<float> levelUp() {
  // C5 E5 G5 C6 arpeggio, then a held major chord with a sparkle on top.
  const float notes[4] = {523.25f, 659.26f, 783.99f, 1046.5f};
  std::vector<float> b(size_t(samplesFor(1.4f)));
  Rng rng(118);
  for (int k = 0; k < 4; ++k) {
    Osc o, o2;
    const float start = 0.09f * float(k);
    const float decay = k == 3 ? 0.5f : 0.18f;
    for (size_t i = size_t(samplesFor(start)); i < b.size(); ++i) {
      const float t = float(i) / kSr - start;
      b[i] += (0.7f * o.tri(notes[k]) + 0.3f * o2.sine(notes[k] * 2.0f)) * envAD(t, 0.004f, decay);
    }
  }
  const float chord[3] = {523.25f, 659.26f, 783.99f};
  for (float f : chord) {
    Osc o;
    const float start = 0.36f;
    for (size_t i = size_t(samplesFor(start)); i < b.size(); ++i) {
      const float t = float(i) / kSr - start;
      b[i] += 0.35f * o.sine(f) * envAD(t, 0.02f, 0.45f);
    }
  }
  for (int k = 0; k < 10; ++k) {
    Osc o;
    const float start = 0.36f + 0.06f * float(k);
    const float f = 2093.0f * (1.0f + 0.5f * rng.uni());
    for (size_t i = size_t(samplesFor(start)); i < b.size(); ++i) {
      const float t = float(i) / kSr - start;
      b[i] += 0.08f * o.sine(f) * envAD(t, 0.001f, 0.05f);
    }
  }
  normalize(b, 0.6f);
  declick(b);
  return b;
}

std::vector<float> uiClick() {
  std::vector<float> b(size_t(samplesFor(0.045f)));
  Osc o;
  for (size_t i = 0; i < b.size(); ++i) {
    const float t = float(i) / kSr;
    b[i] = o.sine(sweep(t, 0.02f, 1900.0f, 1300.0f)) * envAD(t, 0.0005f, 0.008f);
  }
  normalize(b, 0.4f);
  declick(b);
  return b;
}

std::vector<float> ambient() {
  // 24 s loop: Cmaj7 -> Am7 -> Fmaj7 -> G6, 6 s each, soft cross-fades.
  // Every frequency is snapped to a whole number of cycles per loop, so the
  // pad loops without a seam; the wind is noise cross-faded at the loop point.
  const float loopSec = 24.0f;
  const int len = samplesFor(loopSec), fade = samplesFor(1.0f);
  const float chords[4][4] = {
      {130.81f, 164.81f, 196.00f, 246.94f}, // C3 E3 G3 B3
      {110.00f, 164.81f, 196.00f, 261.63f}, // A2 E3 G3 C4
      {87.31f, 130.81f, 164.81f, 220.00f},  // F2 C3 E3 A3
      {98.00f, 146.83f, 196.00f, 246.94f},  // G2 D3 G3 B3
  };
  auto snap = [&](float f) { return std::round(f * loopSec) / loopSec; };

  std::vector<float> pad(size_t(len), 0.0f);
  const float seg = loopSec / 4.0f;
  for (int c = 0; c < 4; ++c) {
    const float centre = seg * (float(c) + 0.5f);
    for (int v = 0; v < 4; ++v) {
      const float f1 = snap(chords[c][v]);
      const float f2 = snap(chords[c][v] * 1.004f);  // gentle chorus
      const float f3 = snap(chords[c][v] * 2.0f);    // octave shimmer
      for (int i = 0; i < len; ++i) {
        const float t = float(i) / kSr;
        float d = std::abs(t - centre);
        d = std::min(d, loopSec - d);                // circular distance
        const float w = std::clamp(1.0f - (d - seg * 0.35f) / (seg * 0.65f), 0.0f, 1.0f);
        if (w <= 0.0f)
          continue;
        // Phase computed from absolute time (double) so it stays exact and
        // the snapped frequencies line up at the loop point.
        const double tt = double(i) / double(kSr);
        auto phase = [tt](float f) {
          const double p = double(f) * tt;
          return float(p - std::floor(p));
        };
        const float s = 0.6f * std::sin(kTau * phase(f1)) +
                        0.4f * (4.0f * std::abs(phase(f2) - 0.5f) - 1.0f) +
                        0.12f * std::sin(kTau * phase(f3));
        pad[size_t(i)] += s * w * w * (3.0f - 2.0f * w);
      }
    }
  }
  // Slow tremolo (whole cycles per loop).
  for (int i = 0; i < len; ++i) {
    const float t = float(i) / kSr;
    pad[size_t(i)] *= 0.85f + 0.15f * std::sin(kTau * t * (6.0f / loopSec));
  }

  Rng rng(120);
  std::vector<float> windSrc(size_t(len + fade));
  OnePoleLP lp1, lp2;
  for (size_t i = 0; i < windSrc.size(); ++i) {
    const float t = float(i) / kSr;
    const float swell = 0.5f + 0.5f * std::sin(kTau * t * (3.0f / loopSec) + 0.7f);
    windSrc[i] = lp2.process(lp1.process(rng.noise(), 500.0f + 300.0f * swell), 700.0f) * (0.3f + 0.7f * swell);
  }
  std::vector<float> wind = loopify(windSrc, len, fade);
  normalize(pad, 1.0f);
  normalize(wind, 1.0f);
  std::vector<float> out(size_t(len));
  for (int i = 0; i < len; ++i)
    out[size_t(i)] = std::tanh(0.55f * pad[size_t(i)] + 0.25f * wind[size_t(i)]);
  normalize(out, 0.5f);
  return out;
}

} // namespace

bool sfxIsLoop(Sfx which) { return which == Sfx::GlideWind || which == Sfx::Ambient; }

const char *sfxName(Sfx which) {
  static const char *kNames[kSfxCount] = {
      "footstep_grass", "footstep_stone", "jump",        "land",        "glide_wind",
      "dash",           "sword_swing",    "bow_twang",   "arrow_hit",   "monster_hit",
      "player_hurt",    "break_stone",    "break_dirt",  "break_wood",  "break_glass",
      "place",          "pickup",         "level_up",    "ui_click",    "ambient"};
  return size_t(which) < size_t(kSfxCount) ? kNames[size_t(which)] : "unknown";
}

std::vector<float> synthesize(Sfx which) {
  switch (which) {
  case Sfx::FootstepGrass: return footstepGrass();
  case Sfx::FootstepStone: return footstepStone();
  case Sfx::Jump: return jump();
  case Sfx::Land: return land();
  case Sfx::GlideWind: return glideWind();
  case Sfx::Dash: return dash();
  case Sfx::SwordSwing: return swordSwing();
  case Sfx::BowTwang: return bowTwang();
  case Sfx::ArrowHit: return arrowHit();
  case Sfx::MonsterHit: return monsterHit();
  case Sfx::PlayerHurt: return playerHurt();
  case Sfx::BreakStone: return breakStone();
  case Sfx::BreakDirt: return breakDirt();
  case Sfx::BreakWood: return breakWood();
  case Sfx::BreakGlass: return breakGlass();
  case Sfx::Place: return place();
  case Sfx::Pickup: return pickup();
  case Sfx::LevelUp: return levelUp();
  case Sfx::UiClick: return uiClick();
  case Sfx::Ambient: return ambient();
  case Sfx::Count: break;
  }
  return {};
}

bool SfxBank::build(atm::Audio &audio) {
  SDL_AudioSpec spec{};
  spec.format = SDL_AUDIO_S16;
  spec.channels = 1;
  spec.freq = kSfxSampleRate;

  bool ok = true;
  pcmBytes_ = 0;
  std::vector<int16_t> pcm;
  for (int i = 0; i < kSfxCount; ++i) {
    const std::vector<float> samples = synthesize(Sfx(i));
    pcm.resize(samples.size());
    for (size_t k = 0; k < samples.size(); ++k) {
      const float v = std::clamp(samples[k], -1.0f, 1.0f);
      pcm[k] = int16_t(std::lround(v * 32767.0f));
    }
    const uint32_t bytes = uint32_t(pcm.size() * sizeof(int16_t));
    ids_[size_t(i)] = audio.loadSoundFromMemory(spec, reinterpret_cast<const uint8_t *>(pcm.data()), bytes);
    pcmBytes_ += bytes;
    if (ids_[size_t(i)] == atm::kInvalidSound)
      ok = false;
  }
  return ok;
}

} // namespace ao::audio
