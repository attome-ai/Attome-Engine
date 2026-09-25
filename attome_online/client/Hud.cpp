// HUD and UI panels (ImGui): health, hotbar, crosshair + mining progress,
// XP drops, level-up banners, floating damage numbers, chat, inventory with
// equipping, RuneScape-style skills panel, debug overlay.

#include "App.h"

#include <SDL3/SDL.h>
#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace ao::client {

namespace {

ImU32 col(uint32_t c) {
  return IM_COL32(c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF, (c >> 24) & 0xFF);
}

void outlinedText(ImDrawList *dl, ImVec2 p, ImU32 color, const char *text, float scale = 1.0f) {
  ImFont *font = ImGui::GetFont();
  const float size = ImGui::GetFontSize() * scale;
  const ImU32 shadow = IM_COL32(0, 0, 0, 200);
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      if (dx || dy)
        dl->AddText(font, size, ImVec2(p.x + dx, p.y + dy), shadow, text);
  dl->AddText(font, size, p, color, text);
}

const char *shortItemName(ItemId id) {
  const ItemDef &d = itemDef(id);
  return d.name.empty() ? "?" : d.name.data();
}

} // namespace

void App::drawHud(float dt) {
  (void)dt;
  const ImGuiIO &io = ImGui::GetIO();
  const ImVec2 screen = io.DisplaySize;
  ImDrawList *fg = ImGui::GetForegroundDrawList();

  // Connection status (before the world exists).
  if (!status_.empty()) {
    const ImVec2 ts = ImGui::CalcTextSize(status_.c_str());
    outlinedText(fg, ImVec2((screen.x - ts.x * 1.5f) * 0.5f, screen.y * 0.45f),
                 IM_COL32(255, 255, 255, 255), status_.c_str(), 1.5f);
  }
  if (!welcomed_)
    return;

  // Crosshair + mining progress.
  {
    const ImVec2 c(screen.x * 0.5f, screen.y * 0.5f);
    const ImU32 w = IM_COL32(255, 255, 255, 220);
    fg->AddLine(ImVec2(c.x - 8, c.y), ImVec2(c.x - 3, c.y), w, 2.0f);
    fg->AddLine(ImVec2(c.x + 3, c.y), ImVec2(c.x + 8, c.y), w, 2.0f);
    fg->AddLine(ImVec2(c.x, c.y - 8), ImVec2(c.x, c.y - 3), w, 2.0f);
    fg->AddLine(ImVec2(c.x, c.y + 3), ImVec2(c.x, c.y + 8), w, 2.0f);
    if (mineProgress_ > 0.0f) {
      const float r = 14.0f;
      fg->PathArcTo(c, r, -1.5708f, -1.5708f + 6.2832f * std::min(mineProgress_, 1.0f), 32);
      fg->PathStroke(IM_COL32(255, 220, 90, 230), 0, 3.0f);
    }
  }

  // Floating texts (damage numbers) projected into the world.
  for (const FloatingText &f : floating_) {
    float sx, sy;
    const glm::dvec3 p = f.world + glm::dvec3(0.0, f.age * 1.2, 0.0);
    if (!worldToScreen(p, sx, sy))
      continue;
    const float a = 1.0f - f.age / f.life;
    const uint32_t c = (f.color & 0x00FFFFFFu) | (uint32_t(a * 255.0f) << 24);
    outlinedText(fg, ImVec2(sx - 10, sy), col(c), f.text.c_str(), 1.4f);
  }

  // Names over remote players.
  const float renderTick = serverTickEstimate_ - 3.0f;
  for (auto &[id, r] : remotes_) {
    if (r.last.kind != EntityKind::Player || r.name.empty() || r.track.empty())
      continue;
    const RemoteSample s = r.track.sample(renderTick);
    float sx, sy;
    if (worldToScreen(s.pos + glm::dvec3(0.0, 2.4, 0.0), sx, sy)) {
      const ImVec2 ts = ImGui::CalcTextSize(r.name.c_str());
      outlinedText(fg, ImVec2(sx - ts.x * 0.5f, sy), IM_COL32(255, 255, 255, 230), r.name.c_str());
    }
  }

  // Health bar (bottom centre) and hotbar.
  {
    const float barW = 360.0f, barH = 18.0f;
    const ImVec2 p0((screen.x - barW) * 0.5f, screen.y - 110.0f);
    fg->AddRectFilled(p0, ImVec2(p0.x + barW, p0.y + barH), IM_COL32(20, 20, 30, 200), 4.0f);
    const float frac = maxHp_ ? std::clamp(float(hp_) / float(maxHp_), 0.0f, 1.0f) : 0.0f;
    fg->AddRectFilled(p0, ImVec2(p0.x + barW * frac, p0.y + barH), IM_COL32(220, 60, 70, 235), 4.0f);
    char hpText[32];
    std::snprintf(hpText, sizeof(hpText), "%u / %u", unsigned(hp_), unsigned(maxHp_));
    const ImVec2 ts = ImGui::CalcTextSize(hpText);
    outlinedText(fg, ImVec2(p0.x + (barW - ts.x) * 0.5f, p0.y + 1), IM_COL32(255, 255, 255, 255), hpText);

    const float slot = 52.0f, gap = 6.0f;
    const float totalW = 9 * slot + 8 * gap;
    const ImVec2 h0((screen.x - totalW) * 0.5f, screen.y - 80.0f);
    for (int i = 0; i < 9; ++i) {
      const ImVec2 a(h0.x + i * (slot + gap), h0.y);
      const ImVec2 b(a.x + slot, a.y + slot);
      fg->AddRectFilled(a, b, IM_COL32(15, 20, 35, 190), 6.0f);
      fg->AddRect(a, b, i == hotbar_ ? IM_COL32(255, 220, 90, 255) : IM_COL32(255, 255, 255, 60),
                  6.0f, 0, i == hotbar_ ? 3.0f : 1.0f);
      const ItemStack &s = inventory_[size_t(i)];
      if (s.item) {
        char label[40];
        std::snprintf(label, sizeof(label), "%.7s", shortItemName(s.item));
        outlinedText(fg, ImVec2(a.x + 4, a.y + 6), IM_COL32(255, 255, 255, 240), label, 0.85f);
        if (s.count > 1) {
          char cnt[12];
          std::snprintf(cnt, sizeof(cnt), "%u", unsigned(s.count));
          outlinedText(fg, ImVec2(a.x + 4, b.y - 18), IM_COL32(255, 230, 120, 255), cnt, 0.9f);
        }
      }
      char key[4];
      std::snprintf(key, sizeof(key), "%d", i + 1);
      fg->AddText(ImVec2(b.x - 11, a.y + 2), IM_COL32(255, 255, 255, 120), key);
    }
  }

  // XP drops (right of crosshair, rising).
  {
    float y = screen.y * 0.5f - 40.0f;
    for (const auto &x : xpDrops_) {
      char t[64];
      std::snprintf(t, sizeof(t), "+%u %s XP", x.amount, std::string(skillName(x.skill)).c_str());
      const float a = 1.0f - x.age / 1.8f;
      outlinedText(fg, ImVec2(screen.x * 0.5f + 60.0f, y - x.age * 40.0f),
                   IM_COL32(120, 230, 255, int(a * 255)), t, 1.1f);
      y -= 22.0f;
    }
  }

  // Level-up / rare-drop banners (top centre).
  {
    float y = 90.0f;
    for (const auto &b : banners_) {
      const float a = b.age < 0.3f ? b.age / 0.3f : (b.age > 2.8f ? (3.5f - b.age) / 0.7f : 1.0f);
      const ImVec2 ts = ImGui::CalcTextSize(b.text.c_str());
      outlinedText(fg, ImVec2((screen.x - ts.x * 1.8f) * 0.5f, y),
                   IM_COL32(255, 215, 80, int(std::clamp(a, 0.0f, 1.0f) * 255)), b.text.c_str(), 1.8f);
      y += 40.0f;
    }
  }

  // Chat (bottom left).
  {
    ImGui::SetNextWindowPos(ImVec2(12, screen.y - 290), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(460, 200), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(chatOpen_ ? 0.6f : 0.25f);
    ImGui::Begin("##chat", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                     (chatOpen_ ? 0 : ImGuiWindowFlags_NoInputs));
    ImGui::BeginChild("##lines", ImVec2(0, chatOpen_ ? -30 : 0));
    for (const auto &line : chat_)
      ImGui::TextWrapped("%s", line.c_str());
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f)
      ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
    if (chatOpen_) {
      ImGui::SetKeyboardFocusHere();
      if (ImGui::InputText("##say", chatInput_, sizeof(chatInput_),
                           ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (chatInput_[0]) {
          proto::ChatSend msg;
          msg.text = chatInput_;
          net_.send(msg, atm::net2::Channel::ReliableOrdered);
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
    ImGui::End();
  }

  // Inventory + equipment (Tab).
  if (showInventory_) {
    ImGui::SetNextWindowPos(ImVec2(screen.x - 420, 80), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 560), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Inventory", &showInventory_)) {
      ImGui::TextDisabled("Click an item to equip it. Click equipment to remove it.");
      static const char *slotNames[] = {"Head", "Torso", "Hands", "Legs", "Feet", "Back", "Main hand", "Off hand"};
      ImGui::SeparatorText("Equipment");
      for (int i = 0; i < atm::model::kEquipSlotCount; ++i) {
        const ItemId it = ItemId(equipped_[size_t(i)]);
        ImGui::Text("%-10s", slotNames[i]);
        ImGui::SameLine(110);
        ImGui::PushID(100 + i);
        if (ImGui::Button(it ? shortItemName(it) : "-", ImVec2(200, 0)) && it) {
          proto::Equip e;
          e.inventorySlot = 0;
          e.equipSlot = uint8_t(i);
          e.unequip = true;
          net_.send(e, atm::net2::Channel::ReliableOrdered);
        }
        ImGui::PopID();
      }
      ImGui::SeparatorText("Backpack (28)");
      for (int i = 0; i < kInventorySlots; ++i) {
        const ItemStack &s = inventory_[size_t(i)];
        ImGui::PushID(i);
        char label[48];
        if (s.item)
          std::snprintf(label, sizeof(label), "%.9s\n%u", shortItemName(s.item), unsigned(s.count));
        else
          std::snprintf(label, sizeof(label), " ");
        if (ImGui::Button(label, ImVec2(84, 46)) && s.item) {
          const ItemDef &d = itemDef(s.item);
          if (d.kind == ItemKind::Weapon || d.kind == ItemKind::Tool || d.kind == ItemKind::Armour) {
            proto::Equip e;
            e.inventorySlot = uint8_t(i);
            e.equipSlot = uint8_t(d.slot);
            e.unequip = false;
            net_.send(e, atm::net2::Channel::ReliableOrdered);
          }
        }
        if (ImGui::IsItemHovered() && s.item)
          ImGui::SetTooltip("%s", std::string(itemDef(s.item).name).c_str());
        ImGui::PopID();
        if ((i + 1) % 4 != 0)
          ImGui::SameLine();
      }
    }
    ImGui::End();
    if (!showInventory_ && !chatOpen_) {
      SDL_SetWindowRelativeMouseMode(window_, true);
      mouseCaptured_ = true;
    }
  }

  // Skills (K): RuneScape-style grid with levels, total level and combat level.
  if (showSkills_) {
    ImGui::SetNextWindowPos(ImVec2(20, 80), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 420), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Skills", &showSkills_)) {
      std::array<uint32_t, kSkillCount> levels{};
      uint64_t totalLevel = 0, totalXp = 0;
      for (int i = 0; i < kSkillCount; ++i) {
        levels[size_t(i)] = levelForXp(skillXp_[size_t(i)]);
        if (Skill(i) == Skill::Hitpoints)
          levels[size_t(i)] = std::max<uint32_t>(levels[size_t(i)], 10);
        totalLevel += levels[size_t(i)];
        totalXp += skillXp_[size_t(i)];
      }
      if (ImGui::BeginTable("skills", 3, ImGuiTableFlags_BordersInnerV)) {
        for (int i = 0; i < kSkillCount; ++i) {
          ImGui::TableNextColumn();
          const uint32_t lvl = levels[size_t(i)];
          const uint64_t cur = skillXp_[size_t(i)];
          const uint64_t lo = xpForLevel(lvl), hi = xpForLevel(lvl + 1);
          const float frac = hi > lo ? float(double(cur - lo) / double(hi - lo)) : 1.0f;
          ImGui::Text("%s", std::string(skillName(Skill(i))).c_str());
          ImGui::SameLine(100);
          if (lvl >= 99)
            ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.3f, 1.0f), "%u", lvl);
          else
            ImGui::Text("%u", lvl);
          ImGui::ProgressBar(frac, ImVec2(-1, 4), "");
          if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%llu XP\nNext level at %llu XP",
                              static_cast<unsigned long long>(cur),
                              static_cast<unsigned long long>(hi));
        }
        ImGui::EndTable();
      }
      ImGui::Separator();
      ImGui::Text("Total level: %llu   Total XP: %llu   Combat level: %u",
                  static_cast<unsigned long long>(totalLevel),
                  static_cast<unsigned long long>(totalXp), combatLevel(levels));
    }
    ImGui::End();
  }

  // Debug overlay (F3).
  if (showDebug_) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.55f);
    ImGui::Begin("##debug", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);
    const auto &rs = renderer_.stats();
    const auto ns = net_.stats();
    const MoveState &me = prediction_.current();
    ImGui::Text("FPS %.0f  CPU %.2f ms  GPU %.2f ms", fps_, rs.cpuMs, rs.gpuMs);
    ImGui::Text("Draws %u  Chunks %u/%u visible  Faces %llu", rs.drawCalls, rs.chunksVisible,
                rs.chunksResident, static_cast<unsigned long long>(rs.facesDrawn));
    ImGui::Text("Arena %.1f / %.1f MB  Upload %.1f KB/frame  Models %u",
                double(rs.chunkArenaUsed) / 1048576.0, double(rs.chunkArenaCapacity) / 1048576.0,
                double(rs.uploadBytes) / 1024.0, rs.modelInstances);
    if (world_) {
      const auto ws = world_->stats();
      ImGui::Text("World: %zu chunks, %zu jobs, mesh %.0f us, gen %.0f us, %.1f MB",
                  ws.loadedChunks, ws.pendingJobs, ws.avgMeshMicros, ws.avgGenMicros,
                  double(ws.memoryBytes) / 1048576.0);
    }
    ImGui::Text("Net: RTT %.0f ms  loss %.1f%%  in %.1f KB  out %.1f KB", ns.rttMs, ns.lossPercent,
                double(ns.bytesReceived) / 1024.0, double(ns.bytesSent) / 1024.0);
    ImGui::Text("Pos %.2f %.2f %.2f  ground %d glide %d water %d  correction %.3f",
                me.pos.x, me.pos.y, me.pos.z, me.onGround, me.gliding, me.inWater,
                double(prediction_.lastCorrectionMicroBlocks()) / 1e6);
    ImGui::Text("Entities %zu  tick %u  snapshot %u", remotes_.size(), clientTick_, newestSnapshotTick_);
    ImGui::End();
  }
}

} // namespace ao::client
