#include "ATMAudio.h"

#include "ATMConfig.h"

#include <algorithm>

namespace atm {

Audio::~Audio() { shutdown(); }

bool Audio::init(const EngineConfig &config) {
  if (!config.audio_enabled) {
    return false;
  }
  return init(config.audio_max_voices, config.audio_master_volume);
}

bool Audio::init(int max_voices, float master_volume) {
  if (device_) {
    return true;
  }

  if (!SDL_WasInit(SDL_INIT_AUDIO)) {
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
      SDL_Log("[audio] SDL audio init failed: %s", SDL_GetError());
      return false;
    }
    owns_subsystem_ = true;
  }

  device_ = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
  if (!device_) {
    SDL_Log("[audio] cannot open playback device: %s", SDL_GetError());
    shutdown();
    return false;
  }
  if (!SDL_GetAudioDeviceFormat(device_, &device_spec_, nullptr)) {
    SDL_Log("[audio] cannot query device format: %s", SDL_GetError());
    shutdown();
    return false;
  }

  max_voices = std::clamp(max_voices, 1, 256);
  voices_.resize(static_cast<size_t>(max_voices));
  for (Voice &voice : voices_) {
    // Sounds are pre-converted, so source and destination specs match and
    // SDL only has to mix.
    voice.stream = SDL_CreateAudioStream(&device_spec_, &device_spec_);
    if (!voice.stream || !SDL_BindAudioStream(device_, voice.stream)) {
      SDL_Log("[audio] cannot create voice stream: %s", SDL_GetError());
      shutdown();
      return false;
    }
    SDL_SetAudioStreamGetCallback(voice.stream, &Audio::onStreamNeedsData,
                                  &voice);
  }

  setMasterVolume(master_volume);
  return true;
}

void Audio::shutdown() {
  for (Voice &voice : voices_) {
    if (voice.stream) {
      SDL_DestroyAudioStream(voice.stream);
    }
  }
  voices_.clear();
  if (device_) {
    SDL_CloseAudioDevice(device_);
    device_ = 0;
  }
  if (owns_subsystem_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    owns_subsystem_ = false;
  }
}

void SDLCALL Audio::onStreamNeedsData(void *userdata, SDL_AudioStream *stream,
                                      int additional_amount, int total_amount) {
  (void)additional_amount;
  (void)total_amount;
  // SDL holds the stream lock while calling us, so loop_pcm can't change
  // underneath.
  const Voice *voice = static_cast<const Voice *>(userdata);
  const std::vector<uint8_t> *pcm = voice->loop_pcm;
  if (pcm && !pcm->empty() &&
      SDL_GetAudioStreamQueued(stream) < static_cast<int>(pcm->size())) {
    SDL_PutAudioStreamData(stream, pcm->data(), static_cast<int>(pcm->size()));
  }
}

SoundId Audio::loadSoundFromMemory(const SDL_AudioSpec &spec,
                                   const uint8_t *data, uint32_t size) {
  if (!device_ || !data || size == 0) {
    return kInvalidSound;
  }

  Uint8 *converted = nullptr;
  int converted_size = 0;
  if (!SDL_ConvertAudioSamples(&spec, data, static_cast<int>(size),
                               &device_spec_, &converted, &converted_size)) {
    SDL_Log("[audio] cannot convert sound: %s", SDL_GetError());
    return kInvalidSound;
  }

  SoundId id;
  if (!free_sounds_.empty()) {
    id = free_sounds_.back();
    free_sounds_.pop_back();
  } else {
    id = static_cast<SoundId>(sounds_.size());
    sounds_.push_back(std::make_unique<Sound>());
  }
  sounds_[id]->pcm.assign(converted, converted + converted_size);
  sounds_[id]->loaded = true;
  SDL_free(converted);
  return id;
}

SoundId Audio::loadSound(const std::string &path) {
  if (!device_) {
    return kInvalidSound;
  }
  SDL_AudioSpec spec{};
  Uint8 *data = nullptr;
  Uint32 size = 0;
  if (!SDL_LoadWAV(path.c_str(), &spec, &data, &size)) {
    SDL_Log("[audio] cannot load '%s': %s", path.c_str(), SDL_GetError());
    return kInvalidSound;
  }
  const SoundId id = loadSoundFromMemory(spec, data, size);
  SDL_free(data);
  return id;
}

void Audio::unloadSound(SoundId sound) {
  if (sound < 0 || sound >= static_cast<SoundId>(sounds_.size()) ||
      !sounds_[sound]->loaded) {
    return;
  }
  for (size_t i = 0; i < voices_.size(); ++i) {
    if (voices_[i].sound == sound) {
      stop(VoiceId{static_cast<int32_t>(i), voices_[i].generation});
    }
  }
  sounds_[sound]->pcm.clear();
  sounds_[sound]->pcm.shrink_to_fit();
  sounds_[sound]->loaded = false;
  free_sounds_.push_back(sound);
}

int Audio::pickVoice() const {
  // Prefer an idle voice; otherwise steal the oldest one-shot. Looping voices
  // (music, ambience) are only stolen when everything is looping.
  int oldest_one_shot = -1;
  int oldest_any = -1;
  for (int i = 0; i < static_cast<int>(voices_.size()); ++i) {
    const Voice &v = voices_[i];
    if (v.sound == kInvalidSound || SDL_GetAudioStreamQueued(v.stream) <= 0) {
      if (!v.loop_pcm)
        return i;
    }
    if (!v.loop_pcm &&
        (oldest_one_shot < 0 || v.started_at < voices_[oldest_one_shot].started_at))
      oldest_one_shot = i;
    if (oldest_any < 0 || v.started_at < voices_[oldest_any].started_at)
      oldest_any = i;
  }
  return oldest_one_shot >= 0 ? oldest_one_shot : oldest_any;
}

VoiceId Audio::play(SoundId sound, float volume, bool loop) {
  if (!device_ || sound < 0 || sound >= static_cast<SoundId>(sounds_.size()) ||
      !sounds_[sound]->loaded || sounds_[sound]->pcm.empty()) {
    return {};
  }

  const int index = pickVoice();
  if (index < 0) {
    return {};
  }
  Voice &voice = voices_[index];
  const std::vector<uint8_t> &pcm = sounds_[sound]->pcm;

  SDL_LockAudioStream(voice.stream);
  SDL_ClearAudioStream(voice.stream);
  voice.loop_pcm = loop ? &pcm : nullptr;
  voice.sound = sound;
  voice.generation++;
  voice.started_at = ++play_counter_;
  SDL_SetAudioStreamGain(voice.stream, std::max(volume, 0.0f));
  SDL_PutAudioStreamData(voice.stream, pcm.data(), static_cast<int>(pcm.size()));
  SDL_UnlockAudioStream(voice.stream);

  return VoiceId{index, voice.generation};
}

Audio::Voice *Audio::resolve(VoiceId id) {
  if (id.index < 0 || id.index >= static_cast<int32_t>(voices_.size()))
    return nullptr;
  Voice &voice = voices_[id.index];
  return voice.generation == id.generation ? &voice : nullptr;
}

const Audio::Voice *Audio::resolve(VoiceId id) const {
  return const_cast<Audio *>(this)->resolve(id);
}

void Audio::stop(VoiceId id) {
  Voice *voice = resolve(id);
  if (!voice)
    return;
  SDL_LockAudioStream(voice->stream);
  voice->loop_pcm = nullptr;
  voice->sound = kInvalidSound;
  voice->generation++;
  SDL_ClearAudioStream(voice->stream);
  SDL_UnlockAudioStream(voice->stream);
}

void Audio::stopAll() {
  for (size_t i = 0; i < voices_.size(); ++i)
    stop(VoiceId{static_cast<int32_t>(i), voices_[i].generation});
}

bool Audio::isPlaying(VoiceId id) const {
  const Voice *voice = resolve(id);
  return voice && voice->sound != kInvalidSound &&
         (voice->loop_pcm || SDL_GetAudioStreamQueued(voice->stream) > 0);
}

void Audio::setVoiceVolume(VoiceId id, float volume) {
  if (Voice *voice = resolve(id))
    SDL_SetAudioStreamGain(voice->stream, std::max(volume, 0.0f));
}

void Audio::setMasterVolume(float volume) {
  master_volume_ = std::clamp(volume, 0.0f, 4.0f);
  if (device_)
    SDL_SetAudioDeviceGain(device_, master_volume_);
}

void Audio::pauseAll(bool paused) {
  if (!device_)
    return;
  if (paused)
    SDL_PauseAudioDevice(device_);
  else
    SDL_ResumeAudioDevice(device_);
}

} // namespace atm
