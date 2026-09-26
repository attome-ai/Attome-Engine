#include "SfxPlayer.h"

#include "../../../engine/voxel/BlockRegistry.h"

#include <algorithm>

namespace ao::client {

namespace blk = atm::voxel::blocks;
using audio::Sfx;

bool SfxPlayer::init(atm::Audio &audio) {
  audio_ = &audio;
  return bank_.build(audio);
}

void SfxPlayer::startAmbient() {
  if (!audio_ || ambient_.valid())
    return;
  ambient_ = audio_->play(bank_.id(Sfx::Ambient), 0.35f, /*loop=*/true);
}

Sfx SfxPlayer::map(GameSound s) const {
  switch (s) {
  case GameSound::Jump: return Sfx::Jump;
  case GameSound::Land: return Sfx::Land;
  case GameSound::Dash: return Sfx::Dash;
  case GameSound::SwordSwing: return Sfx::SwordSwing;
  case GameSound::BowShoot: return Sfx::BowTwang;
  case GameSound::ArrowHit: return Sfx::ArrowHit;
  case GameSound::MonsterHit: return Sfx::MonsterHit;
  case GameSound::PlayerHurt: return Sfx::PlayerHurt;
  case GameSound::Footstep: return Sfx::FootstepGrass;
  case GameSound::Pickup: return Sfx::Pickup;
  case GameSound::LevelUp: return Sfx::LevelUp;
  case GameSound::UiClick: return Sfx::UiClick;
  }
  return Sfx::UiClick;
}

float SfxPlayer::attenuation(float distance) {
  if (distance <= 4.0f)
    return 1.0f;
  if (distance >= 32.0f)
    return 0.0f;
  const float t = (distance - 4.0f) / 28.0f;
  return (1.0f - t) * (1.0f - t);
}

void SfxPlayer::play(GameSound s, float volume) {
  if (audio_ && volume > 0.01f)
    audio_->play(bank_.id(map(s)), volume);
}

void SfxPlayer::playAt(GameSound s, float distance) { play(s, attenuation(distance)); }

void SfxPlayer::playFootstep(atm::voxel::BlockId ground) {
  if (!audio_)
    return;
  const bool soft = ground == blk::Grass || ground == blk::Dirt || ground == blk::Sand ||
                    ground == blk::Snow || ground == blk::OakLeaves;
  // Slight volume variation so steps don't sound mechanical.
  const float vol = 0.45f + 0.1f * float((footstepAlt_++ * 7u) % 3u);
  audio_->play(bank_.id(soft ? Sfx::FootstepGrass : Sfx::FootstepStone), vol);
}

Sfx SfxPlayer::breakSound(atm::voxel::BlockId block) const {
  switch (block) {
  case blk::Dirt:
  case blk::Grass:
  case blk::Sand:
  case blk::Snow:
  case blk::OakLeaves:
    return Sfx::BreakDirt;
  case blk::OakLog:
  case blk::Planks:
    return Sfx::BreakWood;
  case blk::Glass:
  case blk::Crystal:
  case blk::Lamp:
    return Sfx::BreakGlass;
  default:
    return Sfx::BreakStone;
  }
}

void SfxPlayer::playBlockHit(atm::voxel::BlockId block) {
  if (audio_)
    audio_->play(bank_.id(breakSound(block)), 0.35f);
}

void SfxPlayer::playBlockBreak(atm::voxel::BlockId block, float distance) {
  const float v = attenuation(distance);
  if (audio_ && v > 0.01f)
    audio_->play(bank_.id(breakSound(block)), v);
}

void SfxPlayer::playBlockPlace(atm::voxel::BlockId, float distance) {
  const float v = attenuation(distance);
  if (audio_ && v > 0.01f)
    audio_->play(bank_.id(Sfx::Place), v);
}

void SfxPlayer::setGlideWind(bool on) {
  if (!audio_ || on == gliding_)
    return;
  gliding_ = on;
  if (on)
    glide_ = audio_->play(bank_.id(Sfx::GlideWind), 0.5f, /*loop=*/true);
  else if (glide_.valid())
    audio_->stop(glide_);
}

} // namespace ao::client
