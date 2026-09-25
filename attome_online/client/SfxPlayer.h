#pragma once

// Game-event sounds for the client, on top of the procedural SfxBank
// (client/audio/Sfx.h) and atm::Audio.

#include "audio/Sfx.h"

#include "../../engine/voxel/VoxelTypes.h"

namespace ao::client {

enum class GameSound : uint8_t {
  Jump, Land, Dash, SwordSwing, BowShoot, ArrowHit, MonsterHit, PlayerHurt,
  Footstep, Pickup, LevelUp, UiClick,
};

class SfxPlayer {
public:
  bool init(atm::Audio &audio);
  void startAmbient();

  void play(GameSound s, float volume = 1.0f);
  // Distance attenuation: full volume within 4 blocks, silent beyond 32.
  void playAt(GameSound s, float distance);
  void playFootstep(atm::voxel::BlockId ground);
  void playBlockHit(atm::voxel::BlockId block);
  void playBlockBreak(atm::voxel::BlockId block, float distance);
  void playBlockPlace(atm::voxel::BlockId block, float distance);
  void setGlideWind(bool on);

private:
  audio::Sfx map(GameSound s) const;
  audio::Sfx breakSound(atm::voxel::BlockId block) const;
  static float attenuation(float distance);

  atm::Audio *audio_ = nullptr;
  audio::SfxBank bank_;
  atm::VoiceId ambient_{}, glide_{};
  bool gliding_ = false;
  uint32_t footstepAlt_ = 0;
};

} // namespace ao::client
