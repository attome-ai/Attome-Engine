#include "shop/WaveBuffShop.h"

#include "Constants.h"
#include "InputManager.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <array>

namespace tower_swarm {
namespace {

constexpr std::uint32_t kDefaultSeed = 0xBADC0DEu;

std::uint32_t xorshift32(std::uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

bool point_in_rect(float x, float y, const SDL_FRect &r) {
  return x >= r.x && y >= r.y && x <= (r.x + r.w) && y <= (r.y + r.h);
}

void set_color(SDL_Renderer *renderer, Rgba8 c) {
  SDL_SetRenderDrawColor(renderer, c.r, c.g, c.b, c.a);
}

constexpr std::array<WaveBuffCardDef, static_cast<std::size_t>(WaveBuffId::Count)>
    kDefs = {{
        {WaveBuffId::Surge, "Surge",
         "All creatures +25% attack speed for 4 waves",
         wave_shop::kSurgeDurationWaves},
        {WaveBuffId::Fortify, "Fortify", "Base HP +15 (this level)", 0},
        {WaveBuffId::FrenziedBlood, "Frenzied Blood",
         "Each kill grants +1 essence for 3 waves",
         wave_shop::kFrenziedBloodDurationWaves},
        {WaveBuffId::SlowTide, "Slow Tide", "Next wave enemies move 35% slower",
         wave_shop::kSlowTideDurationWaves},
        {WaveBuffId::Foresight, "Foresight",
         "Skip next wave's boss/elite modifier",
         wave_shop::kForesightDurationWaves},
        {WaveBuffId::Mend, "Mend", "All creatures restore 50% max HP", 0},
        {WaveBuffId::WildSeed, "Wild Seed",
         "Instantly place a random Tier-2 creature", 0},
        {WaveBuffId::EchoStrike, "Echo Strike",
         "20% of projectile damage repeats 0.3s later for 3 waves",
         wave_shop::kEchoStrikeDurationWaves},
        {WaveBuffId::EssenceCache, "Essence Cache",
         "Gain essence equal to 30% of current balance", 0},
        {WaveBuffId::IronSkin, "Iron Skin",
         "Creatures take 20% less damage for 2 waves",
         wave_shop::kIronSkinDurationWaves},
        {WaveBuffId::ApexHunter, "Apex Hunter",
         "Highest-kill creature deals +50% damage this wave",
         wave_shop::kApexHunterDurationWaves},
        {WaveBuffId::VoidPulse, "Void Pulse",
         "Every 10th kill explodes in an 80px blast this wave",
         wave_shop::kVoidPulseDurationWaves},
    }};

} // namespace

const WaveBuffCardDef &WaveBuffShop::def(WaveBuffId id) {
  const std::size_t idx = static_cast<std::size_t>(id);
  if (idx < kDefs.size()) {
    return kDefs[idx];
  }
  return kDefs[0];
}

Rgba8 WaveBuffShop::iconColor(WaveBuffId id) {
  switch (id) {
  case WaveBuffId::Surge:
    return Rgba8{80, 140, 255, 255};
  case WaveBuffId::Fortify:
    return Rgba8{90, 220, 150, 255};
  case WaveBuffId::FrenziedBlood:
    return Rgba8{240, 90, 90, 255};
  case WaveBuffId::SlowTide:
    return Rgba8{80, 220, 230, 255};
  case WaveBuffId::Foresight:
    return Rgba8{190, 120, 240, 255};
  case WaveBuffId::Mend:
    return Rgba8{120, 240, 190, 255};
  case WaveBuffId::WildSeed:
    return Rgba8{120, 200, 90, 255};
  case WaveBuffId::EchoStrike:
    return Rgba8{255, 224, 120, 255};
  case WaveBuffId::EssenceCache:
    return Rgba8{255, 200, 80, 255};
  case WaveBuffId::IronSkin:
    return Rgba8{200, 200, 210, 255};
  case WaveBuffId::ApexHunter:
    return Rgba8{255, 150, 80, 255};
  case WaveBuffId::VoidPulse:
    return Rgba8{230, 90, 200, 255};
  case WaveBuffId::Count:
    break;
  }
  return kHudTextColor;
}

const char *WaveBuffShop::iconGlyph(WaveBuffId id) {
  switch (id) {
  case WaveBuffId::Surge:
    return "S";
  case WaveBuffId::Fortify:
    return "HP";
  case WaveBuffId::FrenziedBlood:
    return "FB";
  case WaveBuffId::SlowTide:
    return "ST";
  case WaveBuffId::Foresight:
    return "FS";
  case WaveBuffId::Mend:
    return "M";
  case WaveBuffId::WildSeed:
    return "WS";
  case WaveBuffId::EchoStrike:
    return "E";
  case WaveBuffId::EssenceCache:
    return "EC";
  case WaveBuffId::IronSkin:
    return "IS";
  case WaveBuffId::ApexHunter:
    return "AH";
  case WaveBuffId::VoidPulse:
    return "VP";
  case WaveBuffId::Count:
    break;
  }
  return "?";
}

void WaveBuffShop::randomizeDraw(std::uint32_t seed) {
  std::uint32_t rng = seed == 0 ? kDefaultSeed : seed;
  std::array<WaveBuffId, static_cast<std::size_t>(WaveBuffId::Count)> pool{};
  for (std::size_t i = 0; i < pool.size(); ++i) {
    pool[i] = static_cast<WaveBuffId>(static_cast<std::uint8_t>(i));
  }

  for (std::size_t i = pool.size(); i > 1; --i) {
    const std::size_t j =
        static_cast<std::size_t>(xorshift32(rng) % static_cast<std::uint32_t>(i));
    std::swap(pool[i - 1], pool[j]);
  }

  for (std::size_t i = 0; i < draw_.size(); ++i) {
    draw_[i] = pool[i % pool.size()];
  }
}

void WaveBuffShop::open(std::uint32_t seed) {
  randomizeDraw(seed);
  open_ = true;
  hovered_index_ = -1;
  result_ready_ = false;
  skipped_ = false;
}

void WaveBuffShop::close() {
  open_ = false;
  hovered_index_ = -1;
  result_ready_ = false;
  skipped_ = false;
}

bool WaveBuffShop::consumeSelection(WaveBuffId &out_selected, bool &out_skipped) {
  if (!result_ready_) {
    return false;
  }
  out_selected = selected_;
  out_skipped = skipped_;
  result_ready_ = false;
  skipped_ = false;
  return true;
}

bool WaveBuffShop::tick(const InputManager &input, float screen_w,
                        float screen_h) {
  if (!open_) {
    return false;
  }

  const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
  const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
  const float gap = static_cast<float>(kWaveShopCardGapPx);

  const float card_w = static_cast<float>(kWaveShopCardWidthPx);
  const float card_h = static_cast<float>(kWaveShopCardHeightPx);
  const float total_w = card_w * 3.0f + gap * 2.0f;
  const float start_x = (screen_w - total_w) * 0.5f;
  const float card_y = screen_h - static_cast<float>(kWaveShopBottomMarginPx) -
                       card_h - bh - gap;

  const float skip_x = (screen_w - bw) * 0.5f;
  const float skip_y = card_y + card_h + gap;
  const SDL_FRect skip_rect = {skip_x, skip_y, bw, bh};

  const float mx = static_cast<float>(input.mouseX());
  const float my = static_cast<float>(input.mouseY());

  hovered_index_ = -1;
  for (int i = 0; i < 3; ++i) {
    const float x = start_x + static_cast<float>(i) * (card_w + gap);
    const SDL_FRect rect = {x, card_y, card_w, card_h};
    if (point_in_rect(mx, my, rect)) {
      hovered_index_ = i;
      break;
    }
  }

  if (!input.wasMousePressed(SDL_BUTTON_LEFT)) {
    return false;
  }

  if (point_in_rect(mx, my, skip_rect)) {
    selected_ = WaveBuffId::Surge;
    skipped_ = true;
    open_ = false;
    result_ready_ = true;
    return true;
  }

  if (hovered_index_ >= 0 && hovered_index_ < 3) {
    selected_ = draw_[static_cast<std::size_t>(hovered_index_)];
    skipped_ = false;
    open_ = false;
    result_ready_ = true;
    return true;
  }

  return false;
}

void WaveBuffShop::render(SDL_Renderer *r, const InputManager &input,
                          float screen_w, float screen_h,
                          float timer_sec) const {
  if (!open_ || !r) {
    return;
  }

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);

  SDL_FRect overlay = {0.0f, 0.0f, screen_w, screen_h};
  set_color(r, kModalOverlayColor);
  SDL_RenderFillRect(r, &overlay);

  const float bw = static_cast<float>(kConfirmDialogButtonWidthPx);
  const float bh = static_cast<float>(kConfirmDialogButtonHeightPx);
  const float gap = static_cast<float>(kWaveShopCardGapPx);

  const float card_w = static_cast<float>(kWaveShopCardWidthPx);
  const float card_h = static_cast<float>(kWaveShopCardHeightPx);
  const float total_w = card_w * 3.0f + gap * 2.0f;
  const float start_x = (screen_w - total_w) * 0.5f;
  const float card_y = screen_h - static_cast<float>(kWaveShopBottomMarginPx) -
                       card_h - bh - gap;

  set_color(r, kHudTextColor);
  SDL_RenderDebugTextFormat(r, static_cast<float>(kHudPaddingPx),
                            static_cast<float>(kHudTopBarHeightPx + kHudPaddingPx),
                            "WAVE BUFF SHOP  (%.1fs)", timer_sec);

  const float mx = static_cast<float>(input.mouseX());
  const float my = static_cast<float>(input.mouseY());

  for (int i = 0; i < 3; ++i) {
    const float x = start_x + static_cast<float>(i) * (card_w + gap);
    const SDL_FRect rect = {x, card_y, card_w, card_h};
    const bool hover = point_in_rect(mx, my, rect);

    set_color(r, hover ? kWaveShopCardHoverColor : kWaveShopCardColor);
    SDL_RenderFillRect(r, &rect);
    set_color(r, kWaveShopCardBorderColor);
    SDL_RenderRect(r, &rect);

    const WaveBuffCardDef &d = def(draw_[static_cast<std::size_t>(i)]);

    const float inset_x = static_cast<float>(kWaveShopCardTextInsetXPx);
    const float inset_y = static_cast<float>(kWaveShopCardTextInsetYPx);
    const float icon_size = 32.0f;
    const float icon_gap = 12.0f;

    SDL_FRect icon = {rect.x + inset_x, rect.y + inset_y, icon_size, icon_size};
    set_color(r, iconColor(d.id));
    SDL_RenderFillRect(r, &icon);
    set_color(r, kHudBorderColor);
    SDL_RenderRect(r, &icon);

    const char *glyph = iconGlyph(d.id);
    set_color(r, Rgba8{0, 0, 0, 200});
    SDL_RenderDebugTextFormat(r, icon.x + 7.0f, icon.y + 11.0f, "%s", glyph);
    set_color(r, kModalButtonTextColor);
    SDL_RenderDebugTextFormat(r, icon.x + 6.0f, icon.y + 10.0f, "%s", glyph);

    const float text_x = icon.x + icon.w + icon_gap;
    set_color(r, kHudTextColor);
    SDL_RenderDebugTextFormat(
        r, text_x, rect.y + inset_y, "%s", d.name);
    SDL_RenderDebugTextFormat(
        r, text_x,
        rect.y + static_cast<float>(kWaveShopCardTextInsetYPx + kModalPanelLineStepPx),
        "%s", d.description);
  }

  const float skip_x = (screen_w - bw) * 0.5f;
  const float skip_y = card_y + card_h + gap;
  const SDL_FRect skip_rect = {skip_x, skip_y, bw, bh};
  const bool skip_hover = point_in_rect(mx, my, skip_rect);

  set_color(r, skip_hover ? kModalButtonHoverColor : kModalButtonColor);
  SDL_RenderFillRect(r, &skip_rect);
  set_color(r, kHudBorderColor);
  SDL_RenderRect(r, &skip_rect);
  set_color(r, kModalButtonTextColor);
  SDL_RenderDebugTextFormat(
      r, skip_rect.x + static_cast<float>(kModalButtonTextInsetXPx),
      skip_rect.y + static_cast<float>(kModalButtonTextInsetYPx), "SKIP");

  SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

} // namespace tower_swarm
