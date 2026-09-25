#pragma once

// Attome Online UI look: fonts (Fira Sans body, Cinzel display), the ImGui
// theme and small draw-list helpers shared by the HUD and panels.

#include <imgui.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace ao::client::ui {

struct Fonts {
  ImFont *body = nullptr;     // Fira Sans Regular
  ImFont *semibold = nullptr; // Fira Sans SemiBold
  ImFont *bold = nullptr;     // Fira Sans Bold (numbers, damage)
  ImFont *display = nullptr;  // Cinzel (titles, banners)
};
const Fonts &fonts();

// Load the TTFs from assets/fonts (falls back to ImGui's default font) and
// apply the theme. Call once after ImGui::CreateContext().
void init();

// --- palette -------------------------------------------------------------------
namespace color {
inline constexpr ImU32 Gold = IM_COL32(236, 196, 104, 255);
inline constexpr ImU32 GoldDim = IM_COL32(236, 196, 104, 110);
inline constexpr ImU32 Text = IM_COL32(236, 232, 222, 255);
inline constexpr ImU32 TextDim = IM_COL32(170, 166, 158, 255);
inline constexpr ImU32 PanelTop = IM_COL32(26, 30, 42, 225);
inline constexpr ImU32 PanelBottom = IM_COL32(12, 14, 22, 235);
inline constexpr ImU32 Border = IM_COL32(236, 196, 104, 70);
inline constexpr ImU32 Shadow = IM_COL32(0, 0, 0, 140);
inline constexpr ImU32 Health = IM_COL32(214, 58, 62, 255);
inline constexpr ImU32 HealthHi = IM_COL32(255, 112, 100, 255);
inline constexpr ImU32 Xp = IM_COL32(92, 196, 255, 255);
} // namespace color

inline ImU32 withAlpha(ImU32 c, float a) {
  const uint32_t al = uint32_t(float((c >> IM_COL32_A_SHIFT) & 0xFF) * (a < 0 ? 0 : a > 1 ? 1 : a));
  return (c & ~IM_COL32_A_MASK) | (al << IM_COL32_A_SHIFT);
}

// --- drawing helpers ---------------------------------------------------------------
// Text with a soft drop shadow (readable over any background).
void text(ImDrawList *dl, ImFont *font, float size, ImVec2 p, ImU32 col, std::string_view s,
          float shadowAlpha = 0.85f);
// Same, horizontally centred on cx.
void textCentered(ImDrawList *dl, ImFont *font, float size, float cx, float y, ImU32 col,
                  std::string_view s, float shadowAlpha = 0.85f);
ImVec2 measure(ImFont *font, float size, std::string_view s);

// Dark glass panel: drop shadow, vertical gradient, hairline gold border.
void panel(ImDrawList *dl, ImVec2 a, ImVec2 b, float rounding = 8.0f, float alpha = 1.0f);
// Horizontal bar with a trail (damage taken recently) and a glossy highlight.
void bar(ImDrawList *dl, ImVec2 a, ImVec2 b, float frac, float trailFrac, ImU32 fill, ImU32 fillHi,
         float rounding = 4.0f, float alpha = 1.0f);

// "leather_tunic" -> "Leather Tunic".
std::string prettyName(std::string_view snake);

} // namespace ao::client::ui
