#ifndef ATM_AUDIO_H
#define ATM_AUDIO_H

// Small sound system on top of SDL3 audio streams.
//
//   atm::Audio audio;
//   audio.init(engine->config);                  // or init(32, 1.0f)
//   atm::SoundId boom = audio.loadSound("assets/boom.wav");
//   audio.play(boom);                            // fire and forget
//   atm::VoiceId music = audio.play(song, 0.6f, /*loop=*/true);
//   audio.stop(music);
//
// Sounds are converted to the device format once at load time, and SDL mixes
// all playing voices on its own audio thread, so playback adds no work to the
// game loop. Looping voices are refilled from SDL's audio callback, so there's
// no per-frame update() to call either.

#include <SDL3/SDL.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct EngineConfig;

namespace atm {

using SoundId = int32_t;
inline constexpr SoundId kInvalidSound = -1;

// Generation-checked handle: stays safe to use after the voice was reused.
struct VoiceId {
  int32_t index = -1;
  uint32_t generation = 0;
  bool valid() const { return index >= 0; }
};

class Audio {
public:
  Audio() = default;
  ~Audio();

  Audio(const Audio &) = delete;
  Audio &operator=(const Audio &) = delete;

  // Uses audio_enabled / audio_max_voices / audio_master_volume.
  bool init(const EngineConfig &config);
  bool init(int max_voices = 32, float master_volume = 1.0f);
  void shutdown();
  bool isReady() const { return device_ != 0; }

  // WAV files (SDL built-in loader). Returns kInvalidSound on failure.
  SoundId loadSound(const std::string &path);
  // Raw PCM in any SDL format; it is converted to the device format.
  SoundId loadSoundFromMemory(const SDL_AudioSpec &spec, const uint8_t *data,
                              uint32_t size);
  // Stops any voice still playing it.
  void unloadSound(SoundId sound);

  VoiceId play(SoundId sound, float volume = 1.0f, bool loop = false);
  void stop(VoiceId voice);
  void stopAll();
  bool isPlaying(VoiceId voice) const;
  void setVoiceVolume(VoiceId voice, float volume);

  void setMasterVolume(float volume);
  float masterVolume() const { return master_volume_; }
  void pauseAll(bool paused);

  int voiceCount() const { return static_cast<int>(voices_.size()); }

private:
  struct Sound {
    std::vector<uint8_t> pcm; // device format
    bool loaded = false;
  };

  struct Voice {
    SDL_AudioStream *stream = nullptr;
    // Read on the audio thread inside the stream callback; written on the
    // game thread only while holding the stream lock.
    const std::vector<uint8_t> *loop_pcm = nullptr;
    SoundId sound = kInvalidSound;
    uint32_t generation = 1;
    uint64_t started_at = 0;
  };

  static void SDLCALL onStreamNeedsData(void *userdata, SDL_AudioStream *stream,
                                        int additional_amount, int total_amount);
  Voice *resolve(VoiceId voice);
  const Voice *resolve(VoiceId voice) const;
  int pickVoice() const;

  SDL_AudioDeviceID device_ = 0;
  SDL_AudioSpec device_spec_{};
  std::vector<Voice> voices_;
  // unique_ptr keeps each PCM buffer at a stable address: looping voices
  // point at it from the audio thread.
  std::vector<std::unique_ptr<Sound>> sounds_;
  std::vector<SoundId> free_sounds_;
  float master_volume_ = 1.0f;
  uint64_t play_counter_ = 0;
  bool owns_subsystem_ = false;
};

} // namespace atm

#endif // ATM_AUDIO_H
