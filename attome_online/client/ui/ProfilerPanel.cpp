// Profiler panel (F7): the ATM_PROFILE_SCOPE zone tree of the last frames,
// slowest first, with smoothed average, 1 s peak, share of the frame and
// calls per frame. Read-only overlay (does not take the mouse).

#include "app/App.h"
#include "ui/UiTheme.h"

#include "../../../engine/ATMFrameProfiler.h"

#include <imgui.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace ao::client {

void App::drawProfilerPanel() {
#if ATM_PROFILING
  const ImVec2 screen = ImGui::GetIO().DisplaySize;
  const float sc = std::clamp(screen.y / 900.0f, 0.8f, 2.0f);
  const ui::Fonts &F = ui::fonts();
  const atm::prof::Profiler &prof = atm::prof::Profiler::get();
  const auto &zones = prof.zones();

  ImGui::SetNextWindowPos(ImVec2(18 * sc, 118 * sc), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(470 * sc, 0), ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.82f);
  ImGui::Begin("##profiler", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);
  ImGui::PushFont(F.bold, 16.0f * sc);
  const double frame = prof.frameAvgMs();
  ImGui::TextColored(ImVec4(0.93f, 0.77f, 0.41f, 1.0f), "Profiler  %.3f ms/frame  (%.0f FPS)", frame,
                     frame > 0.0 ? 1000.0 / frame : 0.0);
  ImGui::PopFont();
  ImGui::PushFont(F.body, 13.5f * sc);
  const auto &rs = renderer_.stats();
  ImGui::TextDisabled("GPU %.3f ms   |   F7 to close   |   CPU zones, main thread", rs.gpuMs);

  if (ImGui::BeginTable("zones", 5, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg)) {
    ImGui::TableSetupColumn("Zone", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("avg ms");
    ImGui::TableSetupColumn("peak");
    ImGui::TableSetupColumn("%");
    ImGui::TableSetupColumn("calls");
    ImGui::TableHeadersRow();

    // Children per parent, slowest first; skip zones not seen for 2 s.
    const uint32_t cur = prof.frameIndex();
    std::vector<std::vector<int>> kids(zones.size() + 1);
    for (size_t i = 0; i < zones.size(); ++i) {
      if (cur - zones[i].lastSeenFrame > 2000u)
        continue;
      kids[size_t(zones[i].parent + 1)].push_back(int(i));
    }
    for (auto &k : kids)
      std::sort(k.begin(), k.end(), [&](int a, int b) { return zones[size_t(a)].avgMs > zones[size_t(b)].avgMs; });

    std::function<void(int, int)> row = [&](int parentSlot, int depth) {
      for (int id : kids[size_t(parentSlot)]) {
        const atm::prof::Zone &z = zones[size_t(id)];
        const double pct = frame > 0.0 ? z.avgMs / frame * 100.0 : 0.0;
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Indent(float(depth) * 14.0f * sc);
        const ImVec4 col = pct > 25.0 ? ImVec4(1.0f, 0.45f, 0.4f, 1.0f)
                           : pct > 8.0 ? ImVec4(1.0f, 0.8f, 0.4f, 1.0f)
                                       : ImVec4(0.92f, 0.91f, 0.87f, 1.0f);
        ImGui::TextColored(col, "%s", z.name);
        ImGui::Unindent(float(depth) * 14.0f * sc);
        ImGui::TableNextColumn();
        ImGui::TextColored(col, "%.3f", z.avgMs);
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%.2f", z.peakMs);
        ImGui::TableNextColumn();
        ImGui::Text("%4.1f", pct);
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%.0f", double(z.callsPerFrame));
        row(id + 1, depth + 1);
      }
    };
    row(0, 0);
    ImGui::EndTable();
  }
  ImGui::PopFont();
  ImGui::End();
#endif
}

} // namespace ao::client
