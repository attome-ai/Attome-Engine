#include "ATMAssets.h"

#include "ATMConfig.h"
#include "ATMEngine.h"

#if defined(ATM_HAS_TEXT)
#include "ATMText.h"
#endif

// Private copy of stb_image: STB_IMAGE_STATIC keeps every symbol local to
// this file, so games that compile their own stb_image implementation don't
// get duplicate-symbol link errors.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "stb_image.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace atm {

SDL_Surface *load_image_surface(const std::string &path) {
  size_t size = 0;
  void *file = SDL_LoadFile(path.c_str(), &size);
  if (!file) {
    SDL_Log("[assets] cannot read '%s': %s", path.c_str(), SDL_GetError());
    return nullptr;
  }

  int w = 0;
  int h = 0;
  int channels = 0;
  stbi_uc *pixels =
      stbi_load_from_memory(static_cast<const stbi_uc *>(file),
                            static_cast<int>(size), &w, &h, &channels, 4);
  SDL_free(file);
  if (!pixels) {
    SDL_Log("[assets] cannot decode '%s': %s", path.c_str(),
            stbi_failure_reason());
    return nullptr;
  }

  SDL_Surface *surface = SDL_CreateSurface(w, h, SDL_PIXELFORMAT_RGBA32);
  if (surface) {
    for (int row = 0; row < h; ++row) {
      SDL_memcpy(static_cast<uint8_t *>(surface->pixels) + row * surface->pitch,
                 pixels + static_cast<size_t>(row) * w * 4,
                 static_cast<size_t>(w) * 4);
    }
  }
  stbi_image_free(pixels);
  return surface;
}

Assets::Assets(Engine *engine, Audio *audio) : engine_(engine), audio_(audio) {}

Assets::~Assets() { releaseAll(); }

int Assets::acquireTexture(const std::string &path) {
  auto it = textures_.find(path);
  if (it != textures_.end()) {
    it->second.refs++;
    return it->second.id;
  }
  if (!engine_) {
    return -1;
  }

  SDL_Surface *surface = load_image_surface(resolve_path(path));
  if (!surface) {
    return -1;
  }
  const int id = engine_register_texture(engine_, surface, 0, 0, 0, 0);
  SDL_DestroySurface(surface);
  if (id < 0) {
    return -1;
  }
  textures_[path] = TextureEntry{id, 1};
  return id;
}

void Assets::releaseTexture(const std::string &path) {
  auto it = textures_.find(path);
  if (it == textures_.end() || --it->second.refs > 0) {
    return;
  }
  if (engine_) {
    engine_->atlas.unregisterTexture(it->second.id);
  }
  textures_.erase(it);
}

SoundId Assets::acquireSound(const std::string &path) {
  auto it = sounds_.find(path);
  if (it != sounds_.end()) {
    it->second.refs++;
    return it->second.id;
  }
  if (!audio_ || !audio_->isReady()) {
    return kInvalidSound;
  }
  const SoundId id = audio_->loadSound(resolve_path(path));
  if (id == kInvalidSound) {
    return kInvalidSound;
  }
  sounds_[path] = SoundEntry{id, 1};
  return id;
}

void Assets::releaseSound(const std::string &path) {
  auto it = sounds_.find(path);
  if (it == sounds_.end() || --it->second.refs > 0) {
    return;
  }
  if (audio_) {
    audio_->unloadSound(it->second.id);
  }
  sounds_.erase(it);
}

#if defined(ATM_HAS_TEXT)
std::string Assets::fontKey(const std::string &path, float point_size) {
  return path + "@" + std::to_string(point_size);
}

Font *Assets::acquireFont(const std::string &path, float point_size) {
  const std::string key = fontKey(path, point_size);
  auto it = fonts_.find(key);
  if (it != fonts_.end()) {
    it->second.refs++;
    return it->second.font.get();
  }
  if (!engine_ || !engine_->renderer) {
    return nullptr;
  }
  auto font = std::make_unique<Font>();
  if (!font->load(engine_->renderer, resolve_path(path), point_size)) {
    return nullptr;
  }
  Font *raw = font.get();
  fonts_[key] = FontEntry{std::move(font), 1};
  return raw;
}

void Assets::releaseFont(const std::string &path, float point_size) {
  auto it = fonts_.find(fontKey(path, point_size));
  if (it == fonts_.end() || --it->second.refs > 0) {
    return;
  }
  fonts_.erase(it);
}
#endif

int Assets::refCount(const std::string &path) const {
  if (const auto it = textures_.find(path); it != textures_.end())
    return it->second.refs;
  if (const auto it = sounds_.find(path); it != sounds_.end())
    return it->second.refs;
#if defined(ATM_HAS_TEXT)
  int refs = 0;
  const std::string prefix = path + "@";
  for (const auto &[key, entry] : fonts_)
    if (key.compare(0, prefix.size(), prefix) == 0)
      refs += entry.refs;
  return refs;
#else
  return 0;
#endif
}

void Assets::releaseAll() {
  if (engine_) {
    for (const auto &[path, entry] : textures_)
      engine_->atlas.unregisterTexture(entry.id);
  }
  textures_.clear();
  if (audio_) {
    for (const auto &[path, entry] : sounds_)
      audio_->unloadSound(entry.id);
  }
  sounds_.clear();
#if defined(ATM_HAS_TEXT)
  fonts_.clear();
#endif
}

} // namespace atm
