#include "left-panel.h"
#include "toolkit/logging.h"
#include "imgui/imgui.h"

std::string LabelPrefix(char const* const label)
{
    float width = ImGui::CalcItemWidth();

    float x = ImGui::GetCursorPosX();
    ImGui::Text("%s", label);
    ImGui::SameLine();
    // ImGui::SetCursorPosX(x + width * 0.5f + ImGui::GetStyle().ItemInnerSpacing.x);
    // ImGui::SetNextItemWidth(-1);

    std::string labelID = "##";
    labelID += label;

    return labelID;
}

void LeftPanel::Draw()
{
    static bool show_demo_window = false;
    static std::string user_name(100, '\0');

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_Once);

    ImGui::Begin("left-panel", nullptr,
                 ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_MenuBar |
                     ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoResize);

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Tools")) {
            ImGui::MenuItem("Demo Window", NULL, &show_demo_window);
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (ImGui::CollapsingHeader("User")) {
        ImGui::InputText(LabelPrefix("Name").c_str(), user_name.data(), user_name.size());
    }

    ImGui::End();

    if (show_demo_window) {
        ImGui::ShowDemoWindow();
    }
}
