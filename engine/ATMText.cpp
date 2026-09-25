#include "ATMText.h"

#include <SDL3_ttf/SDL_ttf.h>

#include <algorithm>

namespace atm {

namespace {

constexpr int kAtlasWidth = 512;
constexpr int kGlyphPadding = 1;

// Decodes one UTF-8 codepoint and advances `i`. Invalid bytes become U+FFFD.
uint32_t next_codepoint(std::string_view s, size_t &i) {
  const unsigned char c = static_cast<unsigned char>(s[i++]);
  if (c < 0x80)
    return c;
  int extra = 0;
  uint32_t cp = 0;
  if ((c & 0xE0) == 0xC0) {
    extra = 1;
    cp = c & 0x1F;
  } else if ((c & 0xF0) == 0xE0) {
    extra = 2;
    cp = c & 0x0F;
  } else if ((c & 0xF8) == 0xF0) {
    extra = 3;
    cp = c & 0x07;
  } else {
    return 0xFFFD;
  }
  for (int k = 0; k < extra; ++k) {
    if (i >= s.size() || (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80)
      return 0xFFFD;
    cp = (cp << 6) | (static_cast<unsigned char>(s[i++]) & 0x3F);
  }
  return cp;
}

int g_ttf_users = 0;

} // namespace

Font::~Font() { unload(); }

Font::Font(Font &&other) noexcept { *this = std::move(other); }

Font &Font::operator=(Font &&other) noexcept {
  if (this != &other) {
    unload();
    renderer_ = other.renderer_;
    font_ = other.font_;
    atlas_surface_ = other.atlas_surface_;
    atlas_texture_ = other.atlas_texture_;
    atlas_dirty_ = other.atlas_dirty_;
    pen_x_ = other.pen_x_;
    pen_y_ = other.pen_y_;
    shelf_h_ = other.shelf_h_;
    line_height_ = other.line_height_;
    holds_ttf_ = other.holds_ttf_;
    glyphs_ = std::move(other.glyphs_);
    other.holds_ttf_ = false;
    other.font_ = nullptr;
    other.atlas_surface_ = nullptr;
    other.atlas_texture_ = nullptr;
    other.renderer_ = nullptr;
  }
  return *this;
}

bool Font::load(SDL_Renderer *renderer, const std::string &path,
                float point_size, uint32_t first, uint32_t last) {
  unload();
  if (!renderer) {
    return false;
  }
  if (g_ttf_users == 0 && !TTF_WasInit() && !TTF_Init()) {
    SDL_Log("[text] TTF_Init failed: %s", SDL_GetError());
    return false;
  }
  ++g_ttf_users;
  holds_ttf_ = true;

  font_ = TTF_OpenFont(path.c_str(), point_size);
  if (!font_) {
    SDL_Log("[text] cannot open font '%s': %s", path.c_str(), SDL_GetError());
    unload();
    return false;
  }
  renderer_ = renderer;
  line_height_ = TTF_GetFontHeight(font_);

  // Start with room for one shelf; the surface grows as glyphs are added.
  atlas_surface_ = SDL_CreateSurface(kAtlasWidth, std::max(line_height_, 1) + 2,
                                     SDL_PIXELFORMAT_RGBA32);
  if (!atlas_surface_) {
    unload();
    return false;
  }
  SDL_FillSurfaceRect(atlas_surface_, nullptr, 0);

  for (uint32_t cp = first; cp <= last; ++cp)
    addGlyph(cp);
  addGlyph('?'); // fallback for missing glyphs
  return uploadAtlas();
}

void Font::unload() {
  if (atlas_texture_) {
    SDL_DestroyTexture(atlas_texture_);
    atlas_texture_ = nullptr;
  }
  if (atlas_surface_) {
    SDL_DestroySurface(atlas_surface_);
    atlas_surface_ = nullptr;
  }
  if (font_) {
    TTF_CloseFont(font_);
    font_ = nullptr;
  }
  if (holds_ttf_) {
    // Balance the TTF_Init() done in load().
    holds_ttf_ = false;
    if (--g_ttf_users == 0)
      TTF_Quit();
  }
  renderer_ = nullptr;
  glyphs_.clear();
  pen_x_ = pen_y_ = shelf_h_ = 0;
  atlas_dirty_ = false;
}

bool Font::addGlyph(uint32_t codepoint) {
  Glyph &g = glyphs_[codepoint];
  if (!font_ || !TTF_FontHasGlyph(font_, codepoint)) {
    g.present = false;
    return false;
  }

  int advance = 0;
  TTF_GetGlyphMetrics(font_, codepoint, nullptr, nullptr, nullptr, nullptr,
                      &advance);
  g.advance = static_cast<float>(advance);

  SDL_Surface *rendered =
      TTF_RenderGlyph_Blended(font_, codepoint, SDL_Color{255, 255, 255, 255});
  if (!rendered) {
    // Whitespace has no bitmap but still advances the pen.
    g.present = true;
    g.w = g.h = 0.0f;
    return true;
  }

  const int w = rendered->w;
  const int h = rendered->h;
  if (pen_x_ + w + kGlyphPadding > kAtlasWidth) {
    pen_x_ = 0;
    pen_y_ += shelf_h_ + kGlyphPadding;
    shelf_h_ = 0;
  }
  if (pen_y_ + h + kGlyphPadding > atlas_surface_->h) {
    // Grow the atlas (copy the existing glyphs into a taller surface).
    const int new_h = std::max(atlas_surface_->h * 2, pen_y_ + h + kGlyphPadding);
    SDL_Surface *grown =
        SDL_CreateSurface(kAtlasWidth, new_h, SDL_PIXELFORMAT_RGBA32);
    if (!grown) {
      SDL_DestroySurface(rendered);
      g.present = false;
      return false;
    }
    SDL_FillSurfaceRect(grown, nullptr, 0);
    SDL_SetSurfaceBlendMode(atlas_surface_, SDL_BLENDMODE_NONE);
    SDL_BlitSurface(atlas_surface_, nullptr, grown, nullptr);
    SDL_DestroySurface(atlas_surface_);
    atlas_surface_ = grown;
    // Glyph rects are stored in atlas pixels, so nothing else to update.
  }

  SDL_Rect dst{pen_x_, pen_y_, w, h};
  SDL_SetSurfaceBlendMode(rendered, SDL_BLENDMODE_NONE);
  SDL_BlitSurface(rendered, nullptr, atlas_surface_, &dst);
  SDL_DestroySurface(rendered);

  // Store pixel rects for now; draw() normalises by the current atlas size.
  g.uv = SDL_FRect{static_cast<float>(pen_x_), static_cast<float>(pen_y_),
                   static_cast<float>(w), static_cast<float>(h)};
  g.w = static_cast<float>(w);
  g.h = static_cast<float>(h);
  g.present = true;

  pen_x_ += w + kGlyphPadding;
  shelf_h_ = std::max(shelf_h_, h);
  atlas_dirty_ = true;
  return true;
}

bool Font::uploadAtlas() {
  if (!atlas_dirty_ && atlas_texture_)
    return true;
  if (atlas_texture_)
    SDL_DestroyTexture(atlas_texture_);
  atlas_texture_ = SDL_CreateTextureFromSurface(renderer_, atlas_surface_);
  if (!atlas_texture_) {
    SDL_Log("[text] cannot create glyph atlas: %s", SDL_GetError());
    return false;
  }
  SDL_SetTextureBlendMode(atlas_texture_, SDL_BLENDMODE_BLEND);
  atlas_dirty_ = false;
  return true;
}

const Font::Glyph *Font::glyph(uint32_t codepoint) {
  auto it = glyphs_.find(codepoint);
  if (it == glyphs_.end()) {
    addGlyph(codepoint); // first use of a codepoint outside the preload range
    it = glyphs_.find(codepoint);
  }
  if (it != glyphs_.end() && it->second.present)
    return &it->second;
  const auto fallback = glyphs_.find('?');
  return fallback != glyphs_.end() && fallback->second.present
             ? &fallback->second
             : nullptr;
}

void Font::draw(SDL_Renderer *renderer, std::string_view text, float x,
                float y, SDL_FColor color, float scale) {
  if (!font_ || text.empty())
    return;

  vertices_.clear();
  indices_.clear();

  float pen_x = x;
  float pen_y = y;
  uint32_t prev = 0;
  size_t i = 0;
  while (i < text.size()) {
    const uint32_t cp = next_codepoint(text, i);
    if (cp == '\n') {
      pen_x = x;
      pen_y += static_cast<float>(line_height_) * scale;
      prev = 0;
      continue;
    }
    const Glyph *g = glyph(cp);
    if (!g)
      continue;
    if (prev) {
      int kern = 0;
      if (TTF_GetGlyphKerning(font_, prev, cp, &kern))
        pen_x += static_cast<float>(kern) * scale;
    }
    prev = cp;

    if (g->w > 0.0f && g->h > 0.0f) {
      const int base = static_cast<int>(vertices_.size());
      const float x0 = pen_x;
      const float y0 = pen_y;
      const float x1 = pen_x + g->w * scale;
      const float y1 = pen_y + g->h * scale;
      // uv holds atlas pixel rects; normalised below once the atlas size is
      // final (glyph() may have grown it mid-string).
      const SDL_FRect &r = g->uv;
      vertices_.push_back({{x0, y0}, color, {r.x, r.y}});
      vertices_.push_back({{x1, y0}, color, {r.x + r.w, r.y}});
      vertices_.push_back({{x1, y1}, color, {r.x + r.w, r.y + r.h}});
      vertices_.push_back({{x0, y1}, color, {r.x, r.y + r.h}});
      indices_.insert(indices_.end(),
                      {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    pen_x += g->advance * scale;
  }

  if (vertices_.empty() || !uploadAtlas())
    return;

  const float inv_w = 1.0f / static_cast<float>(atlas_surface_->w);
  const float inv_h = 1.0f / static_cast<float>(atlas_surface_->h);
  for (SDL_Vertex &v : vertices_) {
    v.tex_coord.x *= inv_w;
    v.tex_coord.y *= inv_h;
  }

  SDL_RenderGeometry(renderer ? renderer : renderer_, atlas_texture_,
                     vertices_.data(), static_cast<int>(vertices_.size()),
                     indices_.data(), static_cast<int>(indices_.size()));
}

SDL_FPoint Font::measure(std::string_view text, float scale) {
  if (!font_)
    return {0.0f, 0.0f};
  float line_w = 0.0f;
  float max_w = 0.0f;
  int lines = text.empty() ? 0 : 1;
  uint32_t prev = 0;
  size_t i = 0;
  while (i < text.size()) {
    const uint32_t cp = next_codepoint(text, i);
    if (cp == '\n') {
      max_w = std::max(max_w, line_w);
      line_w = 0.0f;
      ++lines;
      prev = 0;
      continue;
    }
    const Glyph *g = glyph(cp);
    if (!g)
      continue;
    if (prev) {
      int kern = 0;
      if (TTF_GetGlyphKerning(font_, prev, cp, &kern))
        line_w += static_cast<float>(kern);
    }
    prev = cp;
    line_w += g->advance;
  }
  max_w = std::max(max_w, line_w);
  return {max_w * scale, static_cast<float>(lines * line_height_) * scale};
}

} // namespace atm
