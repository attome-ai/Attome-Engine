// Minimap (top right, rotates with the camera like RuneScape's) and the world
// map (M): explored terrain from world/MapCache, map regions from
// data/maps/overworld.json, the player, monsters, other players and loot.

#include "app/App.h"
#include "ui/UiTheme.h"

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_internal.h> // ImTextureDataQueueUpload, RegisterUserTexture

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace ao::client {

namespace {

constexpr float kPi = 3.14159265f;

ImU32 toIm(uint32_t c, float a = 1.0f) { return ui::withAlpha(ImU32(c | 0xFF000000u), a); }

void playerArrow(ImDrawList *dl, ImVec2 c, float angle, float s, ImU32 col) {
  // angle: screen-space rotation (0 = pointing up).
  auto rot = [&](float x, float y) {
    const float cs = std::cos(angle), sn = std::sin(angle);
    return ImVec2(c.x + x * cs - y * sn, c.y + x * sn + y * cs);
  };
  const ImVec2 a = rot(0, -s), b = rot(s * 0.7f, s * 0.8f), m = rot(0, s * 0.35f), d = rot(-s * 0.7f, s * 0.8f);
  dl->AddQuadFilled(a, b, m, d, IM_COL32(0, 0, 0, 160));
  const ImVec2 a2 = rot(0, -s * 0.8f), b2 = rot(s * 0.55f, s * 0.62f), m2 = rot(0, s * 0.25f),
               d2 = rot(-s * 0.55f, s * 0.62f);
  dl->AddQuadFilled(a2, b2, m2, d2, col);
}

} // namespace

void App::drawMinimap() {
  const ImVec2 screen = ImGui::GetIO().DisplaySize;
  const float sc = std::clamp(screen.y / 900.0f, 0.8f, 2.0f);
  const ui::Fonts &F = ui::fonts();
  ImDrawList *dl = ImGui::GetForegroundDrawList();

  const float R = 96.0f * sc;                       // radius on screen
  const ImVec2 c(screen.x - R - 22.0f * sc, R + 22.0f * sc);
  constexpr int kRange = 60;                        // blocks shown from centre to rim
  const float ppb = R / float(kRange);              // pixels per block
  constexpr int kCell = 2;                          // blocks per drawn cell

  const glm::dvec3 me = prediction_.current().pos;
  const float yaw = camYaw_;
  // World (dx, dz) -> screen, with the camera's forward pointing up.
  const float fx = -std::sin(yaw), fz = -std::cos(yaw), rx = std::cos(yaw), rz = -std::sin(yaw);
  auto toScreen = [&](double wx, double wz) {
    const float dx = float(wx - me.x), dz = float(wz - me.z);
    return ImVec2(c.x + (dx * rx + dz * rz) * ppb, c.y - (dx * fx + dz * fz) * ppb);
  };

  // Frame + background.
  dl->AddCircleFilled(ImVec2(c.x, c.y + 3), R + 6 * sc, IM_COL32(0, 0, 0, 90), 64);
  dl->AddCircleFilled(c, R + 4 * sc, IM_COL32(20, 24, 34, 230), 64);
  dl->AddCircleFilled(c, R, IM_COL32(28, 40, 58, 255), 64);

  // Terrain: one texture, 1 pixel per block around the player, rebuilt when
  // we step onto another block or a few times a second (explored terrain
  // fills in), drawn as a single rotated quad. Was ~2800 quads + map lookups
  // every frame.
  (void)kCell;
  constexpr int kTex = 128, kHalf = kTex / 2;
  const int32_t px = int32_t(std::floor(me.x)), pz = int32_t(std::floor(me.z));
  if (!miniTex_) {
    miniTex_ = IM_NEW(ImTextureData)();
    miniTex_->Create(ImTextureFormat_RGBA32, kTex, kTex);
    miniTex_->UseColors = true;
    ImGui::RegisterUserTexture(miniTex_);
  }
  miniAge_ += ImGui::GetIO().DeltaTime;
  if (px != miniX_ || pz != miniZ_ || miniAge_ > 0.25f) {
    miniX_ = px, miniZ_ = pz, miniAge_ = 0.0f;
    uint32_t *pix = reinterpret_cast<uint32_t *>(miniTex_->Pixels);
    const double r2 = double(kRange - 1) * (kRange - 1);
    for (int j = 0; j < kTex; ++j)
      for (int i = 0; i < kTex; ++i) {
        const int32_t x = px - kHalf + i, z = pz - kHalf + j;
        const double ddx = x + 0.5 - (px + 0.5), ddz = z + 0.5 - (pz + 0.5);
        uint32_t col = 0;
        if (ddx * ddx + ddz * ddz <= r2) {
          col = mapCache_.colorAt(x, z);
          if ((col >> 24) != 0) col |= 0xFF000000u;
        }
        pix[j * kTex + i] = col;
      }
    ImTextureDataQueueUpload(miniTex_, 0, 0, kTex, kTex);
  }
  {
    const double x0 = px - kHalf, z0 = pz - kHalf, x1 = x0 + kTex, z1 = z0 + kTex;
    dl->AddImageQuad(miniTex_->GetTexRef(), toScreen(x0, z0), toScreen(x1, z0), toScreen(x1, z1), toScreen(x0, z1),
                     ImVec2(0, 0), ImVec2(1, 0), ImVec2(1, 1), ImVec2(0, 1));
  }

  // Markers: loot (rarity), monsters (red), players (blue).
  const float renderTick = serverTickEstimate_ - 3.0f;
  for (const auto &[id, r] : remotes_) {
    if (r.track.empty() || (r.last.flags & 4u))
      continue;
    const glm::dvec3 p = r.track.sample(renderTick).pos;
    const double dx = p.x - me.x, dz = p.z - me.z;
    if (dx * dx + dz * dz > double(kRange - 2) * (kRange - 2))
      continue;
    const ImVec2 s = toScreen(p.x, p.z);
    switch (r.last.kind) {
    case EntityKind::Monster:
      dl->AddCircleFilled(s, 3.2f * sc, IM_COL32(0, 0, 0, 160));
      dl->AddCircleFilled(s, 2.4f * sc, IM_COL32(240, 70, 60, 255));
      break;
    case EntityKind::Player:
      dl->AddCircleFilled(s, 3.4f * sc, IM_COL32(0, 0, 0, 160));
      dl->AddCircleFilled(s, 2.6f * sc, IM_COL32(110, 190, 255, 255));
      break;
    case EntityKind::DroppedItem:
      dl->AddRectFilled(ImVec2(s.x - 2.2f * sc, s.y - 2.2f * sc), ImVec2(s.x + 2.2f * sc, s.y + 2.2f * sc),
                        ui::rarityColor(itemDef(r.last.item).rarity));
      break;
    default:
      break;
    }
  }

  // You: arrow along the body facing, relative to the camera (up = camera).
  playerArrow(dl, c, -(bodyYaw_ - yaw), 7.0f * sc, IM_COL32(255, 255, 255, 255));

  // Rim, north marker, region name + coordinates under the map.
  dl->AddCircle(c, R + 1, ui::color::Border, 64, 2.0f);
  dl->AddCircle(c, R + 4 * sc, IM_COL32(236, 196, 104, 110), 64, 1.5f);
  {
    const ImVec2 n = toScreen(me.x, me.z - double(kRange) - 0.0);
    const ImVec2 dir(n.x - c.x, n.y - c.y);
    const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    const ImVec2 at(c.x + dir.x / len * (R + 2 * sc), c.y + dir.y / len * (R + 2 * sc));
    dl->AddCircleFilled(at, 9.0f * sc, IM_COL32(20, 24, 34, 240));
    dl->AddCircle(at, 9.0f * sc, ui::color::Gold, 20, 1.5f);
    ui::textCentered(dl, F.bold, 12.0f * sc, at.x, at.y - 7.5f * sc, ui::color::Gold, "N", 0.4f);
  }
  const MapRegion *reg = regionAt(me.x, me.z);
  char line[96];
  std::snprintf(line, sizeof(line), "%s", reg ? std::string(reg->name).c_str() : "");
  ui::textCentered(dl, F.semibold, 14.0f * sc, c.x, c.y + R + 10 * sc, ui::color::Text, line);
  std::snprintf(line, sizeof(line), "%d, %d   (M: world map)", int(std::floor(me.x)), int(std::floor(me.z)));
  ui::textCentered(dl, F.body, 12.0f * sc, c.x, c.y + R + 28 * sc, ui::color::TextDim, line, 0.4f);
}

void App::drawWorldMap() {
  const ImVec2 screen = ImGui::GetIO().DisplaySize;
  const float sc = std::clamp(screen.y / 900.0f, 0.8f, 2.0f);
  const ui::Fonts &F = ui::fonts();

  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(screen);
  ImGui::SetNextWindowBgAlpha(0.0f);
  ImGui::Begin("##worldmap", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  ImDrawList *dl = ImGui::GetWindowDrawList();
  const ImVec2 a(40 * sc, 40 * sc), b(screen.x - 40 * sc, screen.y - 40 * sc);
  ui::panel(dl, ImVec2(a.x - 8, a.y - 8), ImVec2(b.x + 8, b.y + 8), 14.0f * sc);
  dl->AddRectFilled(a, b, IM_COL32(24, 34, 50, 255), 8.0f * sc);

  // Pan by dragging, zoom with the wheel (handled here while the map is open).
  const ImGuiIO &io = ImGui::GetIO();
  ImGui::SetCursorScreenPos(a);
  ImGui::InvisibleButton("##mapdrag", ImVec2(b.x - a.x, b.y - a.y));
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
    worldMapPan_.x -= double(io.MouseDelta.x / worldMapZoom_);
    worldMapPan_.y -= double(io.MouseDelta.y / worldMapZoom_);
  }
  if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f)
    worldMapZoom_ = std::clamp(worldMapZoom_ * (io.MouseWheel > 0 ? 1.25f : 0.8f), 0.25f, 8.0f);

  const glm::dvec3 me = prediction_.current().pos;
  const ImVec2 mid((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f);
  const double cx = me.x + worldMapPan_.x, cz = me.z + worldMapPan_.y;
  const float z = worldMapZoom_; // pixels per block, north up (-Z)
  auto toScreen = [&](double wx, double wz) { return ImVec2(mid.x + float(wx - cx) * z, mid.y + float(wz - cz) * z); };

  dl->PushClipRect(a, b, true);
  // Explored tiles; cells merge blocks so each is at least ~3 px.
  const int cell = std::max(1, int(std::ceil(3.0f / z)));
  for (const auto &[key, t] : mapCache_.tiles()) {
    const int32_t tx = MapCache::tileX(key) * MapCache::kTile, tz = MapCache::tileZ(key) * MapCache::kTile;
    const ImVec2 t0 = toScreen(tx, tz), t1 = toScreen(tx + MapCache::kTile, tz + MapCache::kTile);
    if (t1.x < a.x || t1.y < a.y || t0.x > b.x || t0.y > b.y)
      continue;
    for (int lz = 0; lz < MapCache::kTile; lz += cell)
      for (int lx = 0; lx < MapCache::kTile; lx += cell) {
        const uint32_t col = t.color[size_t(lz) * MapCache::kTile + size_t(lx)];
        if ((col >> 24) == 0)
          continue;
        dl->AddRectFilled(toScreen(tx + lx, tz + lz), toScreen(tx + lx + cell, tz + lz + cell), toIm(col));
      }
  }
  // Regions: boundary rings and names (at the ring's north side).
  for (const MapRegion &r : mapRegions()) {
    if (!r.hasShape)
      continue;
    const ImVec2 rc = toScreen(r.centerX, r.centerZ);
    if (r.maxRadius < 1e8)
      dl->AddCircle(rc, float(r.maxRadius) * z, toIm(r.colour, 0.85f), 128, 2.0f);
    const double labelR = (r.minRadius + std::min(r.maxRadius, r.minRadius + 400.0)) * 0.5;
    const ImVec2 lp = toScreen(r.centerX, r.centerZ - labelR);
    char label[96];
    std::snprintf(label, sizeof(label), "%s", std::string(r.name).c_str());
    ui::textCentered(dl, F.display, 17.0f * sc, lp.x, lp.y - 18 * sc, ui::color::Gold, label);
    if (!r.levels.empty()) {
      std::snprintf(label, sizeof(label), "Levels %s", std::string(r.levels).c_str());
      ui::textCentered(dl, F.body, 13.0f * sc, lp.x, lp.y + 2 * sc, ui::color::Text, label, 0.6f);
    }
  }
  // Spawn point, loot, monsters, you.
  {
    const ImVec2 sp = toScreen(0.5, 0.5);
    dl->AddCircleFilled(sp, 5 * sc, IM_COL32(255, 230, 140, 255));
    ui::textCentered(dl, F.semibold, 12 * sc, sp.x, sp.y + 6 * sc, IM_COL32(255, 230, 140, 255), "Spawn");
  }
  const float renderTick = serverTickEstimate_ - 3.0f;
  for (const auto &[id, r] : remotes_) {
    if (r.track.empty() || (r.last.flags & 4u))
      continue;
    const glm::dvec3 p = r.track.sample(renderTick).pos;
    const ImVec2 s = toScreen(p.x, p.z);
    if (r.last.kind == EntityKind::Monster)
      dl->AddCircleFilled(s, 3.0f * sc, IM_COL32(240, 70, 60, 255));
    else if (r.last.kind == EntityKind::Player)
      dl->AddCircleFilled(s, 3.5f * sc, IM_COL32(110, 190, 255, 255));
    else if (r.last.kind == EntityKind::DroppedItem)
      dl->AddRectFilled(ImVec2(s.x - 2.5f * sc, s.y - 2.5f * sc), ImVec2(s.x + 2.5f * sc, s.y + 2.5f * sc),
                        ui::rarityColor(itemDef(r.last.item).rarity));
  }
  // North-up map: the arrow shows the body facing (yaw 0 = north).
  playerArrow(dl, toScreen(me.x, me.z), -bodyYaw_, 9.0f * sc, IM_COL32(255, 255, 255, 255));
  dl->PopClipRect();

  // Title and help.
  char title[96];
  std::snprintf(title, sizeof(title), "%s", std::string(mapName()).c_str());
  ui::textCentered(dl, F.display, 26.0f * sc, mid.x, a.y + 10 * sc, ui::color::Gold, title);
  ui::text(dl, F.body, 13.0f * sc, ImVec2(a.x + 14 * sc, b.y - 26 * sc), ui::color::TextDim,
           "Drag to pan  |  Wheel to zoom  |  Space to re-centre  |  M or Esc to close", 0.5f);
  if (ImGui::IsKeyPressed(ImGuiKey_Space))
    worldMapPan_ = glm::dvec2(0.0);
  ImGui::End();
}

} // namespace ao::client
