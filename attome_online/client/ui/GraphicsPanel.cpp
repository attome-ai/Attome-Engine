// Graphics panel (F10): live look tuning, backed by ui/LookSettings.

#include "app/App.h"
#include "ui/UiTheme.h"

#include "../../../engine/ATMConfig.h"

#include <SDL3/SDL.h>
#include <imgui.h>

#include <algorithm>
#include <string>
#include <string_view>

namespace ao::client {

// F10: live look tuning. Every change applies immediately; Save writes
// config/graphics.json (loaded at startup), Defaults restores the shipped look.
void App::drawLookPanel() {
  const ImVec2 screen = ImGui::GetIO().DisplaySize;
  const float sc = std::clamp(screen.y / 900.0f, 0.8f, 2.0f);
  const ui::Fonts &F = ui::fonts();
  ImGui::SetNextWindowPos(ImVec2(screen.x - 440 * sc, 70 * sc), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(420 * sc, 640 * sc), ImGuiCond_FirstUseEver);
  ImGui::PushFont(F.display, 20.0f * sc);
  const bool open = ImGui::Begin("Graphics", &showLook_, ImGuiWindowFlags_NoCollapse);
  ImGui::PopFont();
  if (!showLook_ && !chatOpen_) {
    SDL_SetWindowRelativeMouseMode(window_, true);
    mouseCaptured_ = true;
  }
  if (!open) {
    ImGui::End();
    return;
  }
  ImGui::PushFont(F.body, 15.0f * sc);
  ImGui::TextDisabled("Changes apply live. F10 or Esc closes.");

  const std::string path = atm::resolve_path("config/graphics.json");
  if (ImGui::Button("Save")) {
    lookStatus_ = look_.save(path) ? "Saved to config/graphics.json" : "Could not write " + path;
  }
  ImGui::SameLine();
  if (ImGui::Button("Defaults")) {
    look_ = LookSettings{};
    lookStatus_ = "Defaults restored (not saved yet)";
  }
  ImGui::SameLine();
  if (ImGui::Button("Reload")) {
    std::string err;
    LookSettings l;
    lookStatus_ = l.load(path, &err) ? (look_ = l, "Reloaded config/graphics.json") : "No saved file yet";
  }
  if (!lookStatus_.empty())
    ImGui::TextColored(ImVec4(0.93f, 0.77f, 0.41f, 1.0f), "%s", lookStatus_.c_str());

  ImGui::Checkbox("VSync (cap FPS at the monitor refresh)", &look_.vsync);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("On: smooth, no tearing, less heat. Off: uncapped FPS (use F7 to profile).");

  ImGui::BeginChild("##look", ImVec2(0, 0));
  ImGui::PushItemWidth(-150 * sc);
  const char *group = "";
  for (const LookSettings::Field &f : look_.fields()) {
    if (std::string_view(group) != f.group) {
      group = f.group;
      ImGui::Dummy(ImVec2(0, 2 * sc));
      ImGui::SeparatorText(group);
      // Extra controls that are not plain sliders.
      if (std::string_view(group) == "Light") {
        ImGui::ColorEdit3("Sky colour", &look_.env.skyColor.x, ImGuiColorEditFlags_Float);
        ImGui::ColorEdit3("Haze / horizon", &look_.env.fogColor.x, ImGuiColorEditFlags_Float);
      }
      if (std::string_view(group) == "Shadows") {
        int mode = look_.env.contactShadows ? 1 : 0;
        ImGui::RadioButton("Simple", &mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Contact-hardening", &mode, 1);
        if (ImGui::IsItemHovered())
          ImGui::SetTooltip("Sharp where things touch, softer further away.");
        look_.env.contactShadows = mode == 1;
      }
    }
    const float range = f.max - f.min;
    const char *fmt = range <= 0.05f ? "%.4f" : (range >= 50.0f ? "%.0f" : "%.2f");
    ImGui::SliderFloat(f.label, f.value, f.min, f.max, fmt);
    if (ImGui::IsItemHovered() && f.help)
      ImGui::SetTooltip("%s", f.help);
  }
  ImGui::PopItemWidth();
  ImGui::EndChild();
  ImGui::PopFont();
  ImGui::End();
}

} // namespace ao::client
