#ifndef ATM_TEXT_H
#define ATM_TEXT_H

// Glyph-cached text rendering (requires SDL3_ttf; available when the engine
// was built with ATM_HAS_TEXT).
//
//   atm::Font font;
//   font.load(engine->renderer, "assets/fonts/AtomicMd.ttf", 24.0f);
//   font.draw(engine->renderer, "Score: 120", 16, 16, {1, 1, 1, 1});
//
// Every glyph is rendered with FreeType once, at load time, into one atlas
// texture. Drawing a string then builds a few quads and issues a single
// SDL_RenderGeometry call — no per-frame rasterisation and no texture churn.
// Glyphs outside the preloaded range are added to the atlas on first use.

#include <SDL3/SDL.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

typedef struct TTF_Font TTF_Font;

namespace atm {

class Font {
public:
  Font() = default;
  ~Font();

  Font(const Font &) = delete;
  Font &operator=(const Font &) = delete;
  Font(Font &&other) noexcept;
  Font &operator=(Font &&other) noexcept;

  // Preloads codepoints [first, last] (printable ASCII by default).
  bool load(SDL_Renderer *renderer, const std::string &path, float point_size,
            uint32_t first = 32, uint32_t last = 126);
  void unload();
  bool isLoaded() const { return font_ != nullptr; }

  // Draws UTF-8 text with its top-left at (x, y). '\n' starts a new line.
  void draw(SDL_Renderer *renderer, std::string_view text, float x, float y,
            SDL_FColor color = {1.0f, 1.0f, 1.0f, 1.0f}, float scale = 1.0f);

  // Size of the text block in pixels at the given scale.
  SDL_FPoint measure(std::string_view text, float scale = 1.0f);
  float lineHeight() const { return static_cast<float>(line_height_); }

private:
  struct Glyph {
    SDL_FRect uv{};   // atlas region in pixels
    float w = 0.0f;   // pixel size of the glyph cell
    float h = 0.0f;
    float advance = 0.0f;
    bool present = false;
  };

  const Glyph *glyph(uint32_t codepoint);
  bool addGlyph(uint32_t codepoint);
  bool uploadAtlas();

  SDL_Renderer *renderer_ = nullptr;
  TTF_Font *font_ = nullptr;
  SDL_Surface *atlas_surface_ = nullptr; // CPU copy, grown as glyphs are added
  SDL_Texture *atlas_texture_ = nullptr;
  bool atlas_dirty_ = false;
  bool holds_ttf_ = false; // this font counted in the shared TTF_Init refcount

  int pen_x_ = 0; // shelf packer state
  int pen_y_ = 0;
  int shelf_h_ = 0;
  int line_height_ = 0;

  std::unordered_map<uint32_t, Glyph> glyphs_;
  std::vector<SDL_Vertex> vertices_; // reused between draw() calls
  std::vector<int> indices_;
};

} // namespace atm

#endif // ATM_TEXT_H
