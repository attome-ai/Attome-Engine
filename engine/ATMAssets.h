#ifndef ATM_ASSETS_H
#define ATM_ASSETS_H

// Reference-counted asset cache. Loading the same path twice returns the same
// asset; it is freed when the last user releases it.
//
//   atm::Assets assets(engine, &audio);
//   int ship = assets.acquireTexture("resource/ship1.png");  // atlas id
//   atm::SoundId boom = assets.acquireSound("sfx/boom.wav");
//   ...
//   assets.releaseTexture("resource/ship1.png");
//
// All work happens at load/release time; nothing here runs per frame.
// Paths go through atm::resolve_path(), so they work from the exe directory.

#include "ATMAudio.h"

#include <SDL3/SDL.h>

#include <memory>
#include <string>
#include <unordered_map>

struct Engine;

namespace atm {

#if defined(ATM_HAS_TEXT)
class Font;
#endif

// Decodes PNG/JPG/BMP/TGA/GIF(first frame) into an RGBA32 surface.
// Caller owns the surface (SDL_DestroySurface).
SDL_Surface *load_image_surface(const std::string &path);

class Assets {
public:
  explicit Assets(Engine *engine, Audio *audio = nullptr);
  ~Assets();

  Assets(const Assets &) = delete;
  Assets &operator=(const Assets &) = delete;

  // Texture ids are TextureAtlas ids (use them as entity texture_ids).
  int acquireTexture(const std::string &path);
  void releaseTexture(const std::string &path);

  SoundId acquireSound(const std::string &path);
  void releaseSound(const std::string &path);

#if defined(ATM_HAS_TEXT)
  // Fonts are cached per (path, size).
  Font *acquireFont(const std::string &path, float point_size);
  void releaseFont(const std::string &path, float point_size);
#endif

  // Current reference count (0 when not loaded).
  int refCount(const std::string &path) const;
  // Frees everything regardless of reference counts.
  void releaseAll();

private:
  struct TextureEntry {
    int id = -1;
    int refs = 0;
  };
  struct SoundEntry {
    SoundId id = kInvalidSound;
    int refs = 0;
  };
#if defined(ATM_HAS_TEXT)
  struct FontEntry {
    std::unique_ptr<Font> font;
    int refs = 0;
  };
  static std::string fontKey(const std::string &path, float point_size);
  std::unordered_map<std::string, FontEntry> fonts_;
#endif

  Engine *engine_;
  Audio *audio_;
  std::unordered_map<std::string, TextureEntry> textures_;
  std::unordered_map<std::string, SoundEntry> sounds_;
};

} // namespace atm

#endif // ATM_ASSETS_H
