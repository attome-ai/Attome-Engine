#include "UiTheme.h"

#include "../../engine/ATMConfig.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace ao::client::ui {

namespace {

Fonts g_fonts;

ImFont *loadFont(const char *relative, float size) {
  const std::string path = atm::resolve_path(relative);
  if (FILE *f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    cfg.OversampleV = 2;
    return ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), size, &cfg);
  }
  SDL_Log("[ui] font not found: %s (using the default font)", path.c_str());
  return nullptr;
}

ImVec4 rgba(int r, int g, int b, float a = 1.0f) {
  return ImVec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f, a);
}

void applyTheme() {
  ImGuiStyle &s = ImGui::GetStyle();
  s.WindowPadding = ImVec2(16, 14);
  s.FramePadding = ImVec2(10, 6);
  s.ItemSpacing = ImVec2(8, 8);
  s.ItemInnerSpacing = ImVec2(6, 6);
  s.WindowRounding = 10.0f;
  s.ChildRounding = 8.0f;
  s.FrameRounding = 6.0f;
  s.PopupRounding = 8.0f;
  s.GrabRounding = 6.0f;
  s.TabRounding = 6.0f;
  s.ScrollbarRounding = 8.0f;
  s.ScrollbarSize = 10.0f;
  s.WindowBorderSize = 1.0f;
  s.FrameBorderSize = 0.0f;
  s.PopupBorderSize = 1.0f;
  s.WindowTitleAlign = ImVec2(0.5f, 0.5f);
  s.SeparatorTextBorderSize = 1.0f;
  s.SeparatorTextAlign = ImVec2(0.0f, 0.5f);

  ImVec4 *c = s.Colors;
  const ImVec4 gold = rgba(236, 196, 104);
  c[ImGuiCol_Text] = rgba(236, 232, 222);
  c[ImGuiCol_TextDisabled] = rgba(150, 146, 138);
  c[ImGuiCol_WindowBg] = rgba(16, 19, 28, 0.94f);
  c[ImGuiCol_ChildBg] = rgba(255, 255, 255, 0.02f);
  c[ImGuiCol_PopupBg] = rgba(14, 16, 24, 0.97f);
  c[ImGuiCol_Border] = rgba(236, 196, 104, 0.28f);
  c[ImGuiCol_BorderShadow] = rgba(0, 0, 0, 0.0f);
  c[ImGuiCol_FrameBg] = rgba(255, 255, 255, 0.05f);
  c[ImGuiCol_FrameBgHovered] = rgba(236, 196, 104, 0.12f);
  c[ImGuiCol_FrameBgActive] = rgba(236, 196, 104, 0.20f);
  c[ImGuiCol_TitleBg] = rgba(20, 23, 34, 1.0f);
  c[ImGuiCol_TitleBgActive] = rgba(28, 32, 46, 1.0f);
  c[ImGuiCol_TitleBgCollapsed] = rgba(20, 23, 34, 0.8f);
  c[ImGuiCol_ScrollbarBg] = rgba(0, 0, 0, 0.0f);
  c[ImGuiCol_ScrollbarGrab] = rgba(236, 196, 104, 0.25f);
  c[ImGuiCol_ScrollbarGrabHovered] = rgba(236, 196, 104, 0.4f);
  c[ImGuiCol_ScrollbarGrabActive] = rgba(236, 196, 104, 0.55f);
  c[ImGuiCol_CheckMark] = gold;
  c[ImGuiCol_SliderGrab] = gold;
  c[ImGuiCol_SliderGrabActive] = rgba(255, 220, 140);
  c[ImGuiCol_Button] = rgba(255, 255, 255, 0.06f);
  c[ImGuiCol_ButtonHovered] = rgba(236, 196, 104, 0.22f);
  c[ImGuiCol_ButtonActive] = rgba(236, 196, 104, 0.35f);
  c[ImGuiCol_Header] = rgba(236, 196, 104, 0.14f);
  c[ImGuiCol_HeaderHovered] = rgba(236, 196, 104, 0.22f);
  c[ImGuiCol_HeaderActive] = rgba(236, 196, 104, 0.30f);
  c[ImGuiCol_Separator] = rgba(236, 196, 104, 0.22f);
  c[ImGuiCol_SeparatorHovered] = rgba(236, 196, 104, 0.4f);
  c[ImGuiCol_SeparatorActive] = gold;
  c[ImGuiCol_ResizeGrip] = rgba(236, 196, 104, 0.12f);
  c[ImGuiCol_ResizeGripHovered] = rgba(236, 196, 104, 0.3f);
  c[ImGuiCol_ResizeGripActive] = rgba(236, 196, 104, 0.5f);
  c[ImGuiCol_PlotHistogram] = rgba(92, 196, 255);
  c[ImGuiCol_TableBorderLight] = rgba(236, 196, 104, 0.12f);
  c[ImGuiCol_TableBorderStrong] = rgba(236, 196, 104, 0.2f);
  c[ImGuiCol_TextSelectedBg] = rgba(236, 196, 104, 0.3f);
  c[ImGuiCol_NavCursor] = gold;
}

} // namespace

const Fonts &fonts() { return g_fonts; }

void init() {
  g_fonts.body = loadFont("assets/fonts/FiraSans-Regular.ttf", 17.0f);
  g_fonts.semibold = loadFont("assets/fonts/FiraSans-SemiBold.ttf", 17.0f);
  g_fonts.bold = loadFont("assets/fonts/FiraSans-Bold.ttf", 17.0f);
  g_fonts.display = loadFont("assets/fonts/Cinzel.ttf", 22.0f);
  ImFont *fallback = g_fonts.body ? g_fonts.body : ImGui::GetIO().Fonts->AddFontDefault();
  if (!g_fonts.body) g_fonts.body = fallback;
  if (!g_fonts.semibold) g_fonts.semibold = fallback;
  if (!g_fonts.bold) g_fonts.bold = g_fonts.semibold;
  if (!g_fonts.display) g_fonts.display = g_fonts.bold;
  ImGui::GetIO().FontDefault = g_fonts.body;
  applyTheme();
}

ImVec2 measure(ImFont *font, float size, std::string_view s) {
  return font->CalcTextSizeA(size, FLT_MAX, 0.0f, s.data(), s.data() + s.size());
}

void text(ImDrawList *dl, ImFont *font, float size, ImVec2 p, ImU32 col, std::string_view s,
          float shadowAlpha) {
  const float a = float((col >> IM_COL32_A_SHIFT) & 0xFF) / 255.0f;
  const ImU32 sh = IM_COL32(0, 0, 0, int(200.0f * shadowAlpha * a));
  const float o = std::max(1.0f, size * 0.07f);
  const char *b = s.data(), *e = s.data() + s.size();
  dl->AddText(font, size, ImVec2(p.x + o, p.y + o), sh, b, e);
  dl->AddText(font, size, ImVec2(p.x, p.y + o * 0.5f), IM_COL32(0, 0, 0, int(90.0f * shadowAlpha * a)), b, e);
  dl->AddText(font, size, p, col, b, e);
}

void textCentered(ImDrawList *dl, ImFont *font, float size, float cx, float y, ImU32 col,
                  std::string_view s, float shadowAlpha) {
  const ImVec2 m = measure(font, size, s);
  text(dl, font, size, ImVec2(cx - m.x * 0.5f, y), col, s, shadowAlpha);
}

void panel(ImDrawList *dl, ImVec2 a, ImVec2 b, float rounding, float alpha) {
  // Soft drop shadow (3 expanding layers).
  for (int i = 3; i >= 1; --i) {
    const float g = float(i) * 2.5f;
    dl->AddRectFilled(ImVec2(a.x - g, a.y - g + 3), ImVec2(b.x + g, b.y + g + 3),
                      IM_COL32(0, 0, 0, int(28.0f * alpha)), rounding + g);
  }
  // Gradient body: rounded base in the bottom colour, then a gradient inset.
  dl->AddRectFilled(a, b, withAlpha(color::PanelBottom, alpha), rounding);
  const float inset = rounding * 0.3f;
  dl->AddRectFilledMultiColor(ImVec2(a.x + inset, a.y + inset), ImVec2(b.x - inset, b.y - inset),
                              withAlpha(color::PanelTop, alpha), withAlpha(color::PanelTop, alpha),
                              withAlpha(color::PanelBottom, alpha),
                              withAlpha(color::PanelBottom, alpha));
  // Top sheen + gold hairline border.
  dl->AddLine(ImVec2(a.x + rounding, a.y + 1), ImVec2(b.x - rounding, a.y + 1),
              IM_COL32(255, 255, 255, int(28.0f * alpha)));
  dl->AddRect(a, b, withAlpha(color::Border, alpha), rounding, 0, 1.0f);
}

void bar(ImDrawList *dl, ImVec2 a, ImVec2 b, float frac, float trailFrac, ImU32 fill, ImU32 fillHi,
         float rounding, float alpha) {
  frac = std::clamp(frac, 0.0f, 1.0f);
  trailFrac = std::clamp(trailFrac, 0.0f, 1.0f);
  const float w = b.x - a.x, h = b.y - a.y;
  dl->AddRectFilled(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2),
                    IM_COL32(0, 0, 0, int(170.0f * alpha)), rounding + 2);
  dl->AddRectFilled(a, b, IM_COL32(40, 20, 24, int(220.0f * alpha)), rounding);
  if (trailFrac > frac)
    dl->AddRectFilled(ImVec2(a.x + w * frac, a.y), ImVec2(a.x + w * trailFrac, b.y),
                      IM_COL32(255, 226, 150, int(210.0f * alpha)), rounding);
  if (frac > 0.0f) {
    const ImVec2 e(a.x + w * frac, b.y);
    dl->AddRectFilledMultiColor(a, ImVec2(e.x, a.y + h * 0.5f), withAlpha(fillHi, alpha),
                                withAlpha(fillHi, alpha), withAlpha(fill, alpha), withAlpha(fill, alpha));
    dl->AddRectFilled(ImVec2(a.x, a.y + h * 0.5f), e, withAlpha(fill, alpha));
    // Gloss line near the top.
    dl->AddLine(ImVec2(a.x + 2, a.y + 2), ImVec2(std::max(a.x + 2, e.x - 2), a.y + 2),
                IM_COL32(255, 255, 255, int(70.0f * alpha)));
  }
  dl->AddRect(ImVec2(a.x - 1, a.y - 1), ImVec2(b.x + 1, b.y + 1), withAlpha(color::Border, alpha),
              rounding + 1);
}

std::string prettyName(std::string_view snake) {
  std::string out;
  out.reserve(snake.size());
  bool up = true;
  for (char ch : snake) {
    if (ch == '_') {
      out.push_back(' ');
      up = true;
    } else {
      out.push_back(up ? char(std::toupper(static_cast<unsigned char>(ch))) : ch);
      up = false;
    }
  }
  return out;
}

} // namespace ao::client::ui
