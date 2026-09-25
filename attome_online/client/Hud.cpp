// HUD and UI panels (ImGui, drawn with ui:: theme helpers): player frame,
// compass with nearby-threat markers, hotbar with item icons, crosshair +
// mining progress, monster nameplates with health bars, damage numbers, XP
// drops, level-up banners, low-health vignette, chat, inventory with
// equipment, RuneScape-style skills panel, debug overlay.

#include "App.h"
#include "UiTheme.h"

#include <SDL3/SDL.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

namespace ao::client {

namespace {

namespace color = ui::color;

// Game colours are packed bytes R,G,B,A: the same layout as IM_COL32.
ImU32 col(uint32_t c) { return ImU32(c); }

ImU32 shade(ImU32 c, float k) {
  auto ch = [&](int shift) { return std::min(255, int(float((c >> shift) & 0xFF) * k)); };
  return IM_COL32(ch(0), ch(8), ch(16), (c >> 24) & 0xFF);
}

ImU32 mixCol(ImU32 a, ImU32 b, float t) {
  auto ch = [&](int shift) {
    return int(float((a >> shift) & 0xFF) * (1.0f - t) + float((b >> shift) & 0xFF) * t);
  };
  return IM_COL32(ch(0), ch(8), ch(16), ch(24));
}

float uiScale(const ImVec2 &screen) { return std::clamp(screen.y / 900.0f, 0.8f, 2.0f); }

// Accent per item category (slot rim, tooltip title).
ImU32 kindAccent(ItemKind k) {
  switch (k) {
  case ItemKind::Weapon: return IM_COL32(236, 128, 84, 255);
  case ItemKind::Tool: return IM_COL32(120, 184, 236, 255);
  case ItemKind::Armour: return IM_COL32(176, 132, 236, 255);
  case ItemKind::Food: return IM_COL32(236, 196, 104, 255);
  case ItemKind::Resource: return IM_COL32(120, 214, 150, 255);
  case ItemKind::Block: return IM_COL32(200, 190, 170, 255);
  default: return IM_COL32(200, 200, 200, 255);
  }
}

const char *kindLabel(ItemKind k) {
  switch (k) {
  case ItemKind::Weapon: return "Weapon";
  case ItemKind::Tool: return "Tool";
  case ItemKind::Armour: return "Armour";
  case ItemKind::Food: return "Food";
  case ItemKind::Resource: return "Material";
  case ItemKind::Block: return "Block";
  default: return "Item";
  }
}

// Material tint guessed from the item name (leather, iron, ...).
ImU32 materialTint(std::string_view name, ImU32 fallback) {
  auto has = [&](std::string_view s) { return name.find(s) != std::string_view::npos; };
  if (has("leather")) return IM_COL32(158, 104, 62, 255);
  if (has("iron")) return IM_COL32(190, 196, 206, 255);
  if (has("steel")) return IM_COL32(150, 170, 196, 255);
  if (has("gold")) return IM_COL32(236, 196, 90, 255);
  if (has("red")) return IM_COL32(200, 56, 56, 255);
  if (has("wood")) return IM_COL32(170, 122, 72, 255);
  if (has("slime")) return IM_COL32(110, 214, 110, 255);
  if (has("bone")) return IM_COL32(232, 226, 204, 255);
  if (has("coal")) return IM_COL32(64, 64, 72, 255);
  if (has("crystal")) return IM_COL32(140, 220, 255, 255);
  return fallback;
}

// Vector icons: blocks as isometric cubes in their real colours, weapons and
// tools as silhouettes, armour as a chest piece, food, materials as gems.
void itemIcon(ImDrawList *dl, ImVec2 c, float s, const ItemDef &d,
              const atm::voxel::BlockRegistry &blocks) {
  const float r = s * 0.5f;
  const ImU32 outline = IM_COL32(0, 0, 0, 170);
  switch (d.kind) {
  case ItemKind::Block: {
    const auto &b = blocks.get(d.placesBlock);
    const ImU32 top = shade(col(b.colorTop) | IM_COL32_A_MASK, 1.08f);
    const ImU32 side = col(b.colorSide) | IM_COL32_A_MASK;
    const float w = r * 0.86f, h = r * 0.5f;
    const ImVec2 t(c.x, c.y - r * 0.92f), l(c.x - w, c.y - h * 0.84f), rr(c.x + w, c.y - h * 0.84f),
        m(c.x, c.y), bl(c.x - w, c.y + r * 0.5f), br(c.x + w, c.y + r * 0.5f), bm(c.x, c.y + r * 0.95f);
    dl->AddQuadFilled(t, rr, m, l, top);
    dl->AddQuadFilled(l, m, bm, bl, shade(side, 0.82f));
    dl->AddQuadFilled(m, rr, br, bm, shade(side, 0.62f));
    dl->AddQuad(t, rr, m, l, IM_COL32(255, 255, 255, 60), 1.0f);
    dl->AddLine(m, bm, IM_COL32(0, 0, 0, 60), 1.0f);
    break;
  }
  case ItemKind::Weapon:
  case ItemKind::Tool: {
    const ImU32 metal = materialTint(d.name, IM_COL32(206, 212, 222, 255));
    const ImU32 wood = IM_COL32(140, 94, 52, 255);
    if (d.weapon == WeaponType::Bow) {
      dl->PathArcTo(ImVec2(c.x - r * 0.45f, c.y), r * 0.95f, -1.15f, 1.15f, 20);
      dl->PathStroke(outline, 0, s * 0.16f);
      dl->PathArcTo(ImVec2(c.x - r * 0.45f, c.y), r * 0.95f, -1.15f, 1.15f, 20);
      dl->PathStroke(wood, 0, s * 0.09f);
      const float ex = c.x - r * 0.45f + std::cos(1.15f) * r * 0.95f;
      const float ey = std::sin(1.15f) * r * 0.95f;
      dl->AddLine(ImVec2(ex, c.y - ey), ImVec2(ex, c.y + ey), IM_COL32(236, 232, 222, 220), 1.5f);
    } else if (d.weapon == WeaponType::Staff) {
      dl->AddLine(ImVec2(c.x - r * 0.7f, c.y + r * 0.8f), ImVec2(c.x + r * 0.45f, c.y - r * 0.4f), outline, s * 0.16f);
      dl->AddLine(ImVec2(c.x - r * 0.7f, c.y + r * 0.8f), ImVec2(c.x + r * 0.45f, c.y - r * 0.4f), wood, s * 0.09f);
      dl->AddCircleFilled(ImVec2(c.x + r * 0.5f, c.y - r * 0.5f), r * 0.32f, IM_COL32(120, 200, 255, 255));
      dl->AddCircleFilled(ImVec2(c.x + r * 0.42f, c.y - r * 0.58f), r * 0.1f, IM_COL32(255, 255, 255, 200));
    } else if (d.weapon == WeaponType::Pickaxe || d.kind == ItemKind::Tool) {
      dl->AddLine(ImVec2(c.x - r * 0.7f, c.y + r * 0.8f), ImVec2(c.x + r * 0.35f, c.y - r * 0.3f), outline, s * 0.15f);
      dl->AddLine(ImVec2(c.x - r * 0.7f, c.y + r * 0.8f), ImVec2(c.x + r * 0.35f, c.y - r * 0.3f), wood, s * 0.085f);
      dl->PathArcTo(ImVec2(c.x + r * 0.05f, c.y + r * 0.05f), r * 0.85f, -2.55f, -0.55f, 16);
      dl->PathStroke(outline, 0, s * 0.19f);
      dl->PathArcTo(ImVec2(c.x + r * 0.05f, c.y + r * 0.05f), r * 0.85f, -2.55f, -0.55f, 16);
      dl->PathStroke(metal, 0, s * 0.11f);
    } else { // sword
      const ImVec2 tip(c.x + r * 0.72f, c.y - r * 0.72f), guard(c.x - r * 0.25f, c.y + r * 0.25f);
      dl->AddLine(guard, tip, outline, s * 0.2f);
      dl->AddLine(guard, tip, metal, s * 0.12f);
      dl->AddLine(ImVec2(guard.x + r * 0.1f, guard.y + r * 0.1f), ImVec2(tip.x - r * 0.1f, tip.y + r * 0.1f),
                  IM_COL32(255, 255, 255, 110), 1.2f);
      dl->AddLine(ImVec2(c.x - r * 0.55f, c.y - r * 0.05f), ImVec2(c.x - r * 0.05f, c.y + r * 0.45f),
                  IM_COL32(236, 196, 104, 255), s * 0.1f);
      dl->AddLine(ImVec2(c.x - r * 0.3f, c.y + r * 0.3f), ImVec2(c.x - r * 0.72f, c.y + r * 0.72f), wood, s * 0.1f);
    }
    break;
  }
  case ItemKind::Armour: {
    const ImU32 m = materialTint(d.name, IM_COL32(150, 130, 200, 255));
    const ImVec2 pts[] = {{c.x - r * 0.75f, c.y - r * 0.6f}, {c.x - r * 0.3f, c.y - r * 0.78f},
                          {c.x, c.y - r * 0.55f},             {c.x + r * 0.3f, c.y - r * 0.78f},
                          {c.x + r * 0.75f, c.y - r * 0.6f},  {c.x + r * 0.6f, c.y - r * 0.1f},
                          {c.x + r * 0.5f, c.y + r * 0.8f},   {c.x - r * 0.5f, c.y + r * 0.8f},
                          {c.x - r * 0.6f, c.y - r * 0.1f}};
    dl->AddConvexPolyFilled(pts, 9, m);
    dl->AddPolyline(pts, 9, outline, ImDrawFlags_Closed, 1.5f);
    dl->AddLine(ImVec2(c.x, c.y - r * 0.5f), ImVec2(c.x, c.y + r * 0.75f), shade(m, 0.7f), 1.5f);
    dl->AddLine(ImVec2(c.x - r * 0.5f, c.y + r * 0.2f), ImVec2(c.x + r * 0.5f, c.y + r * 0.2f),
                IM_COL32(236, 196, 104, 200), 2.0f);
    break;
  }
  case ItemKind::Food: {
    dl->AddLine(ImVec2(c.x + r * 0.1f, c.y + r * 0.1f), ImVec2(c.x + r * 0.7f, c.y + r * 0.7f),
                IM_COL32(236, 228, 210, 255), s * 0.12f);
    dl->AddCircleFilled(ImVec2(c.x + r * 0.75f, c.y + r * 0.62f), r * 0.14f, IM_COL32(236, 228, 210, 255));
    dl->AddCircleFilled(ImVec2(c.x - r * 0.12f, c.y - r * 0.12f), r * 0.58f, IM_COL32(150, 70, 40, 255));
    dl->AddCircleFilled(ImVec2(c.x - r * 0.24f, c.y - r * 0.26f), r * 0.28f, IM_COL32(196, 110, 64, 255));
    dl->AddCircle(ImVec2(c.x - r * 0.12f, c.y - r * 0.12f), r * 0.58f, outline, 20, 1.5f);
    break;
  }
  default: { // resource: faceted gem / nugget
    const ImU32 m = materialTint(d.name, IM_COL32(120, 214, 150, 255));
    const ImVec2 top(c.x, c.y - r * 0.8f), lft(c.x - r * 0.7f, c.y - r * 0.15f), rgt(c.x + r * 0.7f, c.y - r * 0.15f),
        bot(c.x, c.y + r * 0.85f), mid(c.x, c.y - r * 0.15f);
    dl->AddTriangleFilled(top, rgt, mid, shade(m, 1.15f));
    dl->AddTriangleFilled(top, mid, lft, shade(m, 0.95f));
    dl->AddTriangleFilled(lft, mid, bot, shade(m, 0.78f));
    dl->AddTriangleFilled(mid, rgt, bot, shade(m, 0.62f));
    const ImVec2 q[] = {top, rgt, bot, lft};
    dl->AddPolyline(q, 4, outline, ImDrawFlags_Closed, 1.5f);
    break;
  }
  }
}

// One inventory / hotbar slot. Returns true when clicked (if interactive).
void drawSlot(ImDrawList *dl, ImVec2 a, float size, const ItemStack &s, bool selected, bool hovered,
              const atm::voxel::BlockRegistry &blocks, const char *keyLabel, float sc) {
  const ImVec2 b(a.x + size, a.y + size);
  const float rnd = 7.0f * sc;
  dl->AddRectFilled(ImVec2(a.x - 1, a.y + 2), ImVec2(b.x + 1, b.y + 3), IM_COL32(0, 0, 0, 90), rnd);
  dl->AddRectFilledMultiColor(a, b, IM_COL32(40, 44, 58, 235), IM_COL32(40, 44, 58, 235),
                              IM_COL32(18, 20, 30, 240), IM_COL32(18, 20, 30, 240));
  if (s.item) {
    const ItemDef &d = itemDef(s.item);
    const ImU32 acc = kindAccent(d.kind);
    // Soft category glow at the bottom of the slot.
    dl->AddRectFilledMultiColor(ImVec2(a.x, a.y + size * 0.55f), b, ui::withAlpha(acc, 0.0f),
                                ui::withAlpha(acc, 0.0f), ui::withAlpha(acc, 0.22f), ui::withAlpha(acc, 0.22f));
    itemIcon(dl, ImVec2(a.x + size * 0.5f, a.y + size * 0.48f), size * 0.56f, d, blocks);
    if (s.count > 1) {
      char cnt[12];
      std::snprintf(cnt, sizeof(cnt), "%u", unsigned(s.count));
      const float fs = 14.0f * sc;
      const ImVec2 m = ui::measure(ui::fonts().bold, fs, cnt);
      ui::text(dl, ui::fonts().bold, fs, ImVec2(b.x - m.x - 4 * sc, b.y - m.y - 2 * sc), color::Text, cnt);
    }
  }
  if (keyLabel)
    ui::text(dl, ui::fonts().semibold, 12.0f * sc, ImVec2(a.x + 4 * sc, a.y + 2 * sc),
             IM_COL32(236, 232, 222, 150), keyLabel, 0.6f);
  if (selected) {
    dl->AddRect(ImVec2(a.x - 3, a.y - 3), ImVec2(b.x + 3, b.y + 3), IM_COL32(236, 196, 104, 70), rnd + 3, 0, 4.0f);
    dl->AddRect(a, b, color::Gold, rnd, 0, 2.0f);
  } else {
    dl->AddRect(a, b, hovered ? IM_COL32(236, 196, 104, 150) : IM_COL32(255, 255, 255, 34), rnd, 0, 1.0f);
  }
}

void itemTooltip(const ItemStack &s) {
  const ItemDef &d = itemDef(s.item);
  const ui::Fonts &f = ui::fonts();
  ImGui::BeginTooltip();
  ImGui::PushFont(f.bold, 18.0f);
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(kindAccent(d.kind)));
  ImGui::TextUnformatted(ui::prettyName(d.name).c_str());
  ImGui::PopStyleColor();
  ImGui::PopFont();
  ImGui::TextDisabled("%s", kindLabel(d.kind));
  if (d.damage) ImGui::Text("Damage  %u", unsigned(d.damage));
  if (d.armour) ImGui::Text("Armour  %u", unsigned(d.armour));
  if (d.healAmount) ImGui::Text("Heals  %u HP", unsigned(d.healAmount));
  if ((d.kind == ItemKind::Weapon || d.kind == ItemKind::Armour || d.kind == ItemKind::Tool) && d.levelReq > 1)
    ImGui::TextColored(ImVec4(0.93f, 0.77f, 0.41f, 1.0f), "Requires %s %u",
                       std::string(skillName(d.skill)).c_str(), unsigned(d.levelReq));
  if (s.count > 1) ImGui::TextDisabled("Quantity %u", unsigned(s.count));
  if (d.kind == ItemKind::Weapon || d.kind == ItemKind::Armour || d.kind == ItemKind::Tool)
    ImGui::TextDisabled("Click to equip");
  ImGui::EndTooltip();
}

// Plain-colour skill accents for the skills panel.
ImU32 skillAccent(int i) {
  static const ImU32 k[] = {IM_COL32(236, 128, 84, 255), IM_COL32(120, 184, 236, 255),
                            IM_COL32(120, 214, 150, 255), IM_COL32(176, 132, 236, 255),
                            IM_COL32(236, 196, 104, 255), IM_COL32(236, 110, 150, 255)};
  return k[i % 6];
}

} // namespace

void App::drawHud(float dt) {
  const ImGuiIO &io = ImGui::GetIO();
  const ImVec2 screen = io.DisplaySize;
  const float sc = uiScale(screen);
  const ui::Fonts &F = ui::fonts();
  ImDrawList *bg = ImGui::GetBackgroundDrawList(); // world-anchored (under windows)
  ImDrawList *fg = ImGui::GetForegroundDrawList();

  // Connection status (before the world exists).
  if (!status_.empty()) {
    ui::textCentered(fg, F.display, 34.0f * sc, screen.x * 0.5f, screen.y * 0.44f, color::Gold, status_);
  }
  if (!welcomed_)
    return;

  // --- low health vignette -----------------------------------------------------------
  const float hpFrac = maxHp_ ? std::clamp(float(hp_) / float(maxHp_), 0.0f, 1.0f) : 0.0f;
  hpTrail_ = hpTrail_ > hpFrac ? std::max(hpFrac, hpTrail_ - dt * 0.35f) : hpFrac;
  if (hpFrac < 0.4f) {
    const float pulse = 0.75f + 0.25f * std::sin(float(SDL_GetTicks()) * 0.006f);
    const int a = int((0.4f - hpFrac) / 0.4f * 150.0f * pulse);
    const float e = screen.y * 0.22f;
    const ImU32 red = IM_COL32(150, 0, 0, a), none = IM_COL32(150, 0, 0, 0);
    bg->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(screen.x, e), red, red, none, none);
    bg->AddRectFilledMultiColor(ImVec2(0, screen.y - e), screen, none, none, red, red);
    bg->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(e, screen.y), red, none, none, red);
    bg->AddRectFilledMultiColor(ImVec2(screen.x - e, 0), screen, none, red, red, none);
  }

  // --- monster nameplates and player names -------------------------------------------
  const float renderTick = serverTickEstimate_ - 3.0f;
  const glm::dvec3 me = prediction_.current().pos;
  for (auto &[id, r] : remotes_) {
    if (r.track.empty())
      continue;
    const bool monster = r.last.kind == EntityKind::Monster;
    const bool player = r.last.kind == EntityKind::Player;
    if (!monster && !(player && !r.name.empty()))
      continue;
    if (r.last.flags & 4u) // dead
      continue;
    const RemoteSample s = r.track.sample(renderTick);
    const float dist = float(glm::length(s.pos - me));
    if (dist > 32.0f)
      continue;
    float sx, sy;
    if (!worldToScreen(s.pos + glm::dvec3(0.0, monster ? 2.1 : 2.5, 0.0), sx, sy))
      continue;
    const float fade = std::clamp((32.0f - dist) / 8.0f, 0.0f, 1.0f);
    const float k = std::clamp(1.25f - dist / 40.0f, 0.7f, 1.2f) * sc;
    if (player) {
      ui::textCentered(bg, F.semibold, 16.0f * k, sx, sy - 18.0f * k, ui::withAlpha(IM_COL32(150, 210, 255, 255), fade), r.name);
      continue;
    }
    const MonsterDef &md = monsterDef(r.last.type);
    const std::string name = ui::prettyName(md.name);
    ui::textCentered(bg, F.semibold, 15.0f * k, sx, sy - 30.0f * k,
                     ui::withAlpha(IM_COL32(255, 206, 150, 255), fade), name);
    const float bw = 92.0f * k, bh = 8.0f * k;
    const float f = r.last.maxHp ? std::clamp(float(r.last.hp) / float(r.last.maxHp), 0.0f, 1.0f) : 1.0f;
    const ImU32 fill = r.hitFlash > 0.0f ? IM_COL32(255, 240, 220, 255) : color::Health;
    ui::bar(bg, ImVec2(sx - bw * 0.5f, sy - 10.0f * k), ImVec2(sx + bw * 0.5f, sy - 10.0f * k + bh), f, f,
            fill, color::HealthHi, 3.0f, fade);
  }

  // --- floating damage numbers (pop, then rise and fade) -----------------------------
  for (const FloatingText &f : floating_) {
    float sx, sy;
    const glm::dvec3 p = f.world + glm::dvec3(0.0, f.age * 1.2, 0.0);
    if (!worldToScreen(p, sx, sy))
      continue;
    const float t = f.age / f.life;
    const float pop = f.age < 0.12f ? 1.0f + (0.12f - f.age) / 0.12f * 0.6f : 1.0f;
    const float a = t < 0.7f ? 1.0f : 1.0f - (t - 0.7f) / 0.3f;
    ui::textCentered(fg, F.bold, 26.0f * sc * pop, sx, sy, ui::withAlpha(col(f.color) | IM_COL32_A_MASK, a), f.text);
  }

  // --- crosshair + mining progress ---------------------------------------------------
  {
    const ImVec2 c(screen.x * 0.5f, screen.y * 0.5f);
    const ImU32 w = IM_COL32(255, 255, 255, 230), sh = IM_COL32(0, 0, 0, 120);
    fg->AddCircleFilled(c, 3.0f * sc, sh);
    fg->AddCircleFilled(c, 1.8f * sc, w);
    for (int i = 0; i < 4; ++i) {
      const float ang = 0.785398f + float(i) * 1.570796f;
      const ImVec2 d(std::cos(ang), std::sin(ang));
      const ImVec2 p0(c.x + d.x * 7 * sc, c.y + d.y * 7 * sc), p1(c.x + d.x * 12 * sc, c.y + d.y * 12 * sc);
      fg->AddLine(p0, p1, sh, 3.5f);
      fg->AddLine(p0, p1, w, 1.8f);
    }
    if (mineProgress_ > 0.0f) {
      const float r = 20.0f * sc;
      fg->AddCircle(c, r, IM_COL32(0, 0, 0, 120), 40, 5.0f);
      fg->PathArcTo(c, r, -1.5708f, -1.5708f + 6.2832f * std::min(mineProgress_, 1.0f), 40);
      fg->PathStroke(color::Gold, 0, 3.0f);
    }
  }

  // --- player frame (top left) -------------------------------------------------------
  std::array<uint32_t, kSkillCount> levels{};
  uint64_t totalLevel = 0;
  for (int i = 0; i < kSkillCount; ++i) {
    levels[size_t(i)] = levelForXp(skillXp_[size_t(i)]);
    if (Skill(i) == Skill::Hitpoints)
      levels[size_t(i)] = std::max<uint32_t>(levels[size_t(i)], 10);
    totalLevel += levels[size_t(i)];
  }
  const uint32_t cmb = combatLevel(levels);
  {
    const ImVec2 a(18 * sc, 18 * sc), b(a.x + 330 * sc, a.y + 84 * sc);
    ui::panel(fg, a, b, 12.0f * sc);
    const ImVec2 pc(a.x + 42 * sc, a.y + 42 * sc);
    const float pr = 30.0f * sc;
    fg->AddCircleFilled(pc, pr + 3, IM_COL32(0, 0, 0, 150), 40);
    fg->AddCircleFilled(pc, pr, IM_COL32(34, 38, 54, 255), 40);
    fg->AddCircle(pc, pr, color::Gold, 40, 2.0f);
    fg->AddCircle(pc, pr - 4 * sc, IM_COL32(236, 196, 104, 60), 40, 1.0f);
    char lv[8];
    std::snprintf(lv, sizeof(lv), "%u", cmb);
    const ImVec2 lm = ui::measure(F.display, 28.0f * sc, lv);
    ui::text(fg, F.display, 28.0f * sc, ImVec2(pc.x - lm.x * 0.5f, pc.y - lm.y * 0.55f), color::Gold, lv);
    const float x0 = a.x + 84 * sc;
    ui::text(fg, F.semibold, 18.0f * sc, ImVec2(x0, a.y + 10 * sc), color::Text, cfg_.name);
    char sub[48];
    std::snprintf(sub, sizeof(sub), "Combat %u  \xC2\xB7  Total %llu", cmb, static_cast<unsigned long long>(totalLevel));
    const ImVec2 nm = ui::measure(F.semibold, 18.0f * sc, cfg_.name);
    ui::text(fg, F.body, 13.0f * sc, ImVec2(x0 + nm.x + 10 * sc, a.y + 14 * sc), color::TextDim, sub, 0.5f);
    const ImVec2 ba(x0, a.y + 40 * sc), bb(b.x - 16 * sc, a.y + 58 * sc);
    ui::bar(fg, ba, bb, hpFrac, hpTrail_, color::Health, color::HealthHi, 4.0f * sc);
    char hpText[32];
    std::snprintf(hpText, sizeof(hpText), "%u / %u", unsigned(hp_), unsigned(maxHp_));
    ui::textCentered(fg, F.bold, 13.0f * sc, (ba.x + bb.x) * 0.5f, ba.y + 1 * sc, color::Text, hpText);
    ui::text(fg, F.body, 12.0f * sc, ImVec2(x0, a.y + 62 * sc), color::TextDim, "HP", 0.4f);
  }

  // --- compass (top centre) with nearby monsters ------------------------------------
  {
    const float w = 440.0f * sc, h = 30.0f * sc;
    const ImVec2 a((screen.x - w) * 0.5f, 18 * sc), b(a.x + w, a.y + h);
    ui::panel(fg, a, b, h * 0.5f, 0.9f);
    const float heading = std::fmod(-camYaw_ * 57.29578f + 720.0f, 360.0f);
    const float pxPerDeg = w / 150.0f;
    const float cx = (a.x + b.x) * 0.5f;
    fg->PushClipRect(ImVec2(a.x + 8, a.y), ImVec2(b.x - 8, b.y), true);
    static const char *labels[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"};
    for (int deg = 0; deg < 360; deg += 15) {
      float off = std::fmod(float(deg) - heading + 540.0f, 360.0f) - 180.0f;
      if (std::abs(off) > 80.0f)
        continue;
      const float x = cx + off * pxPerDeg;
      const float edge = 1.0f - std::clamp((std::abs(off) - 45.0f) / 30.0f, 0.0f, 1.0f);
      if (deg % 45 == 0) {
        const bool cardinal = deg % 90 == 0;
        ui::textCentered(fg, cardinal ? F.bold : F.semibold, (cardinal ? 16.0f : 13.0f) * sc, x,
                         a.y + (cardinal ? 6.0f : 8.0f) * sc,
                         ui::withAlpha(deg == 0 ? color::Gold : color::Text, edge), labels[deg / 45], 0.5f);
      } else {
        fg->AddLine(ImVec2(x, b.y - 9 * sc), ImVec2(x, b.y - 4 * sc), ui::withAlpha(color::TextDim, edge), 1.0f);
      }
    }
    // Threat markers: monsters within 40 blocks at their bearing.
    for (const auto &[id, r] : remotes_) {
      if (r.last.kind != EntityKind::Monster || (r.last.flags & 4u) || r.track.empty())
        continue;
      const glm::dvec3 d = r.track.sample(renderTick).pos - me;
      const double dist = std::sqrt(d.x * d.x + d.z * d.z);
      if (dist > 40.0 || dist < 0.5)
        continue;
      const float bearing = float(std::atan2(d.x, -d.z)) * 57.29578f;
      const float off = std::fmod(bearing - heading + 540.0f, 360.0f) - 180.0f;
      if (std::abs(off) > 75.0f)
        continue;
      const float x = cx + off * pxPerDeg;
      const float closeness = 1.0f - float(dist) / 40.0f;
      fg->AddTriangleFilled(ImVec2(x - 4 * sc, b.y - 1), ImVec2(x + 4 * sc, b.y - 1), ImVec2(x, b.y - 8 * sc),
                            IM_COL32(236, 84, 70, int(120 + 135 * closeness)));
    }
    fg->PopClipRect();
    fg->AddTriangleFilled(ImVec2(cx - 6 * sc, a.y - 2), ImVec2(cx + 6 * sc, a.y - 2), ImVec2(cx, a.y + 6 * sc), color::Gold);
  }

  // --- hotbar (bottom centre) --------------------------------------------------------
  {
    const float slot = 58.0f * sc, gap = 6.0f * sc;
    const float totalW = 9 * slot + 8 * gap;
    const ImVec2 h0((screen.x - totalW) * 0.5f, screen.y - slot - 22.0f * sc);
    ui::panel(fg, ImVec2(h0.x - 10 * sc, h0.y - 10 * sc), ImVec2(h0.x + totalW + 10 * sc, h0.y + slot + 10 * sc),
              12.0f * sc, 0.85f);
    for (int i = 0; i < 9; ++i) {
      const ImVec2 a(h0.x + float(i) * (slot + gap), h0.y);
      char key[4];
      std::snprintf(key, sizeof(key), "%d", i + 1);
      drawSlot(fg, a, slot, inventory_[size_t(i)], i == hotbar_, false, blocks_, key, sc);
    }
    // Selected item name above the bar.
    const ItemStack &cur = inventory_[size_t(hotbar_)];
    if (cur.item) {
      const ItemDef &cd = itemDef(cur.item);
      const char *hint = cd.kind == ItemKind::Armour ? "Right-click to wear"
                         : cd.kind == ItemKind::Food ? "Right-click to eat"
                         : cd.kind == ItemKind::Block ? "Right-click to place"
                                                      : nullptr;
      const float y = h0.y - (hint ? 52.0f : 36.0f) * sc;
      ui::textCentered(fg, F.semibold, 16.0f * sc, screen.x * 0.5f, y, ui::withAlpha(color::Text, 0.9f),
                       ui::prettyName(cd.name));
      if (hint)
        ui::textCentered(fg, F.body, 13.0f * sc, screen.x * 0.5f, y + 20.0f * sc, color::TextDim, hint, 0.6f);
    }
  }

  // --- XP drops (right of the crosshair, rising) ------------------------------------
  {
    float y = screen.y * 0.5f - 30.0f * sc;
    for (const auto &x : xpDrops_) {
      char t[64];
      std::snprintf(t, sizeof(t), "+%u  %s", x.amount, std::string(skillName(x.skill)).c_str());
      const float a = x.age < 0.15f ? x.age / 0.15f : 1.0f - std::max(0.0f, x.age - 1.2f) / 0.6f;
      const float fs = 16.0f * sc;
      const ImVec2 m = ui::measure(F.semibold, fs, t);
      const ImVec2 p(screen.x * 0.5f + 70.0f * sc, y - x.age * 36.0f * sc);
      fg->AddRectFilled(ImVec2(p.x - 10 * sc, p.y - 3 * sc), ImVec2(p.x + m.x + 10 * sc, p.y + m.y + 3 * sc),
                        IM_COL32(10, 30, 50, int(150 * std::clamp(a, 0.0f, 1.0f))), 20.0f);
      ui::text(fg, F.semibold, fs, p, ui::withAlpha(color::Xp, a), t);
      y -= 30.0f * sc;
    }
  }

  // --- level-up / rare-drop banners (top centre) -------------------------------------
  {
    float y = 110.0f * sc;
    for (const auto &b : banners_) {
      const float a = std::clamp(b.age < 0.3f ? b.age / 0.3f : (b.age > 2.8f ? (3.5f - b.age) / 0.7f : 1.0f), 0.0f, 1.0f);
      const float fs = 38.0f * sc * (b.age < 0.3f ? 0.85f + 0.15f * (b.age / 0.3f) : 1.0f);
      const ImVec2 m = ui::measure(F.display, fs, b.text);
      const float cx = screen.x * 0.5f;
      const float half = m.x * 0.5f + 30.0f * sc;
      fg->AddRectFilledMultiColor(ImVec2(cx - half - 120 * sc, y - 8 * sc), ImVec2(cx, y + m.y + 8 * sc),
                                  IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, int(150 * a)),
                                  IM_COL32(0, 0, 0, int(150 * a)), IM_COL32(0, 0, 0, 0));
      fg->AddRectFilledMultiColor(ImVec2(cx, y - 8 * sc), ImVec2(cx + half + 120 * sc, y + m.y + 8 * sc),
                                  IM_COL32(0, 0, 0, int(150 * a)), IM_COL32(0, 0, 0, 0),
                                  IM_COL32(0, 0, 0, 0), IM_COL32(0, 0, 0, int(150 * a)));
      const ImU32 line = ui::withAlpha(color::Gold, a);
      fg->AddLine(ImVec2(cx - half - 90 * sc, y - 8 * sc), ImVec2(cx + half + 90 * sc, y - 8 * sc), ui::withAlpha(color::Gold, a * 0.6f), 1.0f);
      fg->AddLine(ImVec2(cx - half - 90 * sc, y + m.y + 8 * sc), ImVec2(cx + half + 90 * sc, y + m.y + 8 * sc), ui::withAlpha(color::Gold, a * 0.6f), 1.0f);
      for (float side : {-1.0f, 1.0f}) {
        const ImVec2 d(cx + side * (half + 14 * sc), y + m.y * 0.5f);
        fg->AddQuadFilled(ImVec2(d.x, d.y - 6 * sc), ImVec2(d.x + 6 * sc, d.y), ImVec2(d.x, d.y + 6 * sc),
                          ImVec2(d.x - 6 * sc, d.y), line);
      }
      ui::textCentered(fg, F.display, fs, cx, y, ui::withAlpha(IM_COL32(255, 222, 140, 255), a), b.text);
      y += m.y + 34.0f * sc;
    }
  }

  // --- chat (bottom left): fades out when idle --------------------------------------
  {
    chatIdle_ += dt;
    const float fade = chatOpen_ ? 1.0f : std::clamp(1.0f - (chatIdle_ - 8.0f) / 2.0f, 0.0f, 1.0f);
    if (fade > 0.0f) {
      ImGui::SetNextWindowPos(ImVec2(18 * sc, screen.y - 330 * sc), ImGuiCond_Always);
      ImGui::SetNextWindowSize(ImVec2(470 * sc, 220 * sc), ImGuiCond_Always);
      ImGui::SetNextWindowBgAlpha(chatOpen_ ? 0.85f : 0.45f * fade);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, fade);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12 * sc, 10 * sc));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6 * sc, 3 * sc));
      ImGui::Begin("##chat", nullptr,
                   ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                       ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                       (chatOpen_ ? 0 : ImGuiWindowFlags_NoInputs));
      ImGui::PushFont(F.body, 15.0f * sc);
      ImGui::BeginChild("##lines", ImVec2(0, chatOpen_ ? -36 * sc : 0));
      for (const auto &line : chat_)
        ImGui::TextWrapped("%s", line.c_str());
      if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f)
        ImGui::SetScrollHereY(1.0f);
      ImGui::EndChild();
      if (chatOpen_) {
        ImGui::SetKeyboardFocusHere();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputTextWithHint("##say", "Say something...", chatInput_, sizeof(chatInput_),
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
          if (chatInput_[0]) {
            proto::ChatSend msg;
            msg.text = chatInput_;
            net_.send(msg, atm::net2::Channel::ReliableOrdered);
            chatInput_[0] = 0;
          }
          chatOpen_ = false;
          SDL_SetWindowRelativeMouseMode(window_, true);
          mouseCaptured_ = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
          chatOpen_ = false;
          SDL_SetWindowRelativeMouseMode(window_, true);
          mouseCaptured_ = true;
        }
      }
      ImGui::PopFont();
      ImGui::End();
      ImGui::PopStyleVar(3);
    }
  }

  // --- inventory + equipment (Tab / I) -------------------------------------------------
  if (showInventory_) {
    const float slot = 56.0f * sc, gap = 8.0f * sc;
    const float gridW = 7 * slot + 6 * gap;
    ImGui::SetNextWindowPos(ImVec2(screen.x - gridW - 80 * sc, 110 * sc), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(gridW + 34 * sc, 0), ImGuiCond_Always);
    ImGui::PushFont(F.display, 20.0f * sc);
    const bool open = ImGui::Begin("Inventory", &showInventory_,
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopFont();
    if (open) {
      ImGui::PushFont(F.body, 15.0f * sc);
      ImDrawList *dl = ImGui::GetWindowDrawList();
      static const char *slotNames[] = {"Head", "Torso", "Hands", "Legs", "Feet", "Back", "Main Hand", "Off Hand"};
      ImGui::SeparatorText("Equipment");
      const float eqW = (gridW - gap) / 2.0f;
      for (int i = 0; i < atm::model::kEquipSlotCount; ++i) {
        const ItemId it = ItemId(equipped_[size_t(i)]);
        if (i % 2)
          ImGui::SameLine(0, gap);
        ImGui::PushID(100 + i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("eq", ImVec2(eqW, slot));
        const bool hov = ImGui::IsItemHovered();
        ItemStack st;
        st.item = it;
        st.count = 1;
        drawSlot(dl, p, slot, st, false, hov, blocks_, nullptr, sc);
        ui::text(dl, F.semibold, 13.0f * sc, ImVec2(p.x + slot + 6 * sc, p.y + 8 * sc), color::TextDim, slotNames[i], 0.3f);
        if (it) {
          const std::string nm = ui::prettyName(itemDef(it).name);
          dl->PushClipRect(p, ImVec2(p.x + eqW, p.y + slot), true);
          ui::text(dl, F.body, 12.0f * sc, ImVec2(p.x + slot + 6 * sc, p.y + 28 * sc), color::Text, nm, 0.3f);
          dl->PopClipRect();
          if (hov) {
            itemTooltip(st);
          }
        }
        if (clicked && it) {
          proto::Equip e;
          e.inventorySlot = 0;
          e.equipSlot = uint8_t(i);
          e.unequip = true;
          net_.send(e, atm::net2::Channel::ReliableOrdered);
        }
        ImGui::PopID();
      }
      ImGui::Dummy(ImVec2(0, 4 * sc));
      ImGui::SeparatorText("Backpack");
      for (int i = 0; i < kInventorySlots; ++i) {
        const ItemStack &s = inventory_[size_t(i)];
        if (i % 7)
          ImGui::SameLine(0, gap);
        ImGui::PushID(i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const bool clicked = ImGui::InvisibleButton("slot", ImVec2(slot, slot));
        const bool hov = ImGui::IsItemHovered();
        char key[4];
        std::snprintf(key, sizeof(key), "%d", i + 1);
        drawSlot(dl, p, slot, s, i == hotbar_, hov, blocks_, i < 9 ? key : nullptr, sc);
        if (hov && s.item)
          itemTooltip(s);
        if (clicked && s.item) {
          const ItemDef &d = itemDef(s.item);
          if (d.kind == ItemKind::Weapon || d.kind == ItemKind::Tool || d.kind == ItemKind::Armour) {
            proto::Equip e;
            e.inventorySlot = uint8_t(i);
            e.equipSlot = uint8_t(d.slot);
            e.unequip = false;
            net_.send(e, atm::net2::Channel::ReliableOrdered);
          }
        }
        ImGui::PopID();
      }
      ImGui::Dummy(ImVec2(0, 2 * sc));
      ImGui::TextDisabled("Slots 1-9 are your hotbar. Click gear to equip.");
      ImGui::PopFont();
    }
    ImGui::End();
    if (!showInventory_ && !chatOpen_) {
      SDL_SetWindowRelativeMouseMode(window_, true);
      mouseCaptured_ = true;
    }
  }

  // --- skills (K): RuneScape-style grid ------------------------------------------------
  if (showSkills_) {
    const float cardW = 150.0f * sc, cardH = 54.0f * sc, gap = 8.0f * sc;
    ImGui::SetNextWindowPos(ImVec2(24 * sc, 130 * sc), ImGuiCond_FirstUseEver);
    ImGui::PushFont(F.display, 20.0f * sc);
    const bool open = ImGui::Begin("Skills", &showSkills_,
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                       ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopFont();
    if (open) {
      ImGui::PushFont(F.body, 15.0f * sc);
      ImDrawList *dl = ImGui::GetWindowDrawList();
      uint64_t totalXp = 0;
      for (int i = 0; i < kSkillCount; ++i) {
        if (i % 3)
          ImGui::SameLine(0, gap);
        ImGui::PushID(i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("skill", ImVec2(cardW, cardH));
        const bool hov = ImGui::IsItemHovered();
        const uint32_t lvl = levels[size_t(i)];
        const uint64_t cur = skillXp_[size_t(i)];
        totalXp += cur;
        const uint64_t lo = xpForLevel(lvl), hi = xpForLevel(lvl + 1);
        const float frac = hi > lo ? float(double(cur - lo) / double(hi - lo)) : 1.0f;
        const ImVec2 q(p.x + cardW, p.y + cardH);
        const ImU32 acc = skillAccent(i);
        dl->AddRectFilledMultiColor(p, q, IM_COL32(36, 40, 54, 230), IM_COL32(36, 40, 54, 230),
                                    IM_COL32(20, 22, 32, 235), IM_COL32(20, 22, 32, 235));
        dl->AddRectFilled(p, ImVec2(p.x + 3 * sc, q.y), acc);
        dl->AddRect(p, q, hov ? IM_COL32(236, 196, 104, 160) : IM_COL32(255, 255, 255, 26), 6.0f * sc);
        ui::text(dl, F.semibold, 14.0f * sc, ImVec2(p.x + 12 * sc, p.y + 8 * sc), color::Text,
                 std::string(skillName(Skill(i))), 0.3f);
        char lv[16];
        std::snprintf(lv, sizeof(lv), "%u", lvl);
        const ImVec2 lm = ui::measure(F.bold, 22.0f * sc, lv);
        ui::text(dl, F.bold, 22.0f * sc, ImVec2(q.x - lm.x - 10 * sc, p.y + 6 * sc),
                 lvl >= 99 ? color::Gold : color::Text, lv, 0.4f);
        const ImVec2 ba(p.x + 12 * sc, q.y - 12 * sc), bb(q.x - 10 * sc, q.y - 8 * sc);
        dl->AddRectFilled(ba, bb, IM_COL32(0, 0, 0, 140), 2.0f);
        dl->AddRectFilled(ba, ImVec2(ba.x + (bb.x - ba.x) * std::clamp(frac, 0.0f, 1.0f), bb.y), acc, 2.0f);
        if (hov)
          ImGui::SetTooltip("%s\n%llu XP\nNext level at %llu XP (%llu to go)",
                            std::string(skillName(Skill(i))).c_str(), static_cast<unsigned long long>(cur),
                            static_cast<unsigned long long>(hi),
                            static_cast<unsigned long long>(hi > cur ? hi - cur : 0));
        ImGui::PopID();
      }
      ImGui::Dummy(ImVec2(0, 4 * sc));
      ImGui::Separator();
      ImGui::PushFont(F.semibold, 16.0f * sc);
      ImGui::Text("Total level %llu", static_cast<unsigned long long>(totalLevel));
      ImGui::SameLine(0, 24 * sc);
      ImGui::TextColored(ImVec4(0.93f, 0.77f, 0.41f, 1.0f), "Combat level %u", cmb);
      ImGui::PopFont();
      ImGui::TextDisabled("Total XP %llu", static_cast<unsigned long long>(totalXp));
      ImGui::PopFont();
    }
    ImGui::End();
  }

  // --- debug overlay (F3, top right) ---------------------------------------------------
  if (showDebug_) {
    ImGui::SetNextWindowPos(ImVec2(screen.x - 18 * sc, 18 * sc), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.75f);
    ImGui::Begin("##debug", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);
    ImGui::PushFont(F.body, 14.0f * sc);
    const auto &rs = renderer_.stats();
    const auto ns = net_.stats();
    const MoveState &ms = prediction_.current();
    ImGui::PushFont(F.bold, 18.0f * sc);
    ImGui::TextColored(ImVec4(0.93f, 0.77f, 0.41f, 1.0f), "%.0f FPS", fps_);
    ImGui::PopFont();
    ImGui::Text("CPU %.2f ms   GPU %.2f ms", rs.cpuMs, rs.gpuMs);
    ImGui::Text("Draws %u   Chunks %u/%u   Faces %llu", rs.drawCalls, rs.chunksVisible, rs.chunksResident,
                static_cast<unsigned long long>(rs.facesDrawn));
    ImGui::Text("Arena %.1f / %.1f MB   Upload %.1f KB   Models %u", double(rs.chunkArenaUsed) / 1048576.0,
                double(rs.chunkArenaCapacity) / 1048576.0, double(rs.uploadBytes) / 1024.0, rs.modelInstances);
    if (world_) {
      const auto ws = world_->stats();
      ImGui::Text("World %zu chunks, %zu jobs, mesh %.0f us, gen %.0f us", ws.loadedChunks, ws.pendingJobs,
                  ws.avgMeshMicros, ws.avgGenMicros);
    }
    ImGui::Text("Net RTT %.0f ms   loss %.1f%%", ns.rttMs, ns.lossPercent);
    ImGui::Text("Pos %.1f %.1f %.1f   correction %.3f", ms.pos.x, ms.pos.y, ms.pos.z,
                double(prediction_.lastCorrectionMicroBlocks()) / 1e6);
    ImGui::Text("Entities %zu   tick %u", remotes_.size(), clientTick_);
    ImGui::PopFont();
    ImGui::End();
  }
}

} // namespace ao::client
