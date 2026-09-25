#pragma once

// Procedurally synthesised sound effects for the demo (no audio files).
// Everything is generated once at startup from deterministic seeds as 44.1 kHz
// mono S16 PCM (~3 MB total, the ambient loop being most of it) and handed to
// atm::Audio, which converts it to the device format.
//
//   ao::audio::SfxBank sfx;
//   sfx.build(audio);
//   audio.play(sfx.id(ao::audio::Sfx::Jump));
//   audio.play(sfx.id(ao::audio::Sfx::Ambient), 0.5f, /*loop=*/true);

#include "ATMAudio.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ao::audio {

enum class Sfx : uint8_t {
  FootstepGrass, FootstepStone, Jump, Land,
  GlideWind,          // seamless loop (~3 s)
  Dash, SwordSwing, BowTwang, ArrowHit, MonsterHit, PlayerHurt,
  BreakStone, BreakDirt, BreakWood, BreakGlass, Place, Pickup,
  LevelUp, UiClick,
  Ambient,            // seamless loop (24 s): pad chords + soft wind
  Count
};
inline constexpr int kSfxCount = int(Sfx::Count);
inline constexpr int kSfxSampleRate = 44100;

// Loops should be played with loop = true.
bool sfxIsLoop(Sfx which);
const char *sfxName(Sfx which);

// Deterministic synthesis of one effect as mono float samples in [-1, 1] at
// kSfxSampleRate (usable without an audio device, e.g. in tests).
std::vector<float> synthesize(Sfx which);

class SfxBank {
public:
  SfxBank() { ids_.fill(atm::kInvalidSound); }
  // Synthesises every effect and loads it into `audio`. Returns false if any
  // sound failed to load (the others stay usable). Call once, main thread.
  bool build(atm::Audio &audio);
  atm::SoundId id(Sfx which) const {
    return size_t(which) < ids_.size() ? ids_[size_t(which)] : atm::kInvalidSound;
  }
  size_t pcmBytes() const { return pcmBytes_; }

private:
  std::array<atm::SoundId, kSfxCount> ids_;
  size_t pcmBytes_ = 0;
};

} // namespace ao::audio
