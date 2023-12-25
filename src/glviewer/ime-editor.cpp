#include "ime-editor.h"
#include "imgui/imgui.h"
#include <sstream>

void ImeEditor::Draw()
{
    auto& ctx = DisplayContext::Instance();
    if (!ctx.ime.actived || !ctx.ime.state.isComposing) {
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(300, -1));
    ImGui::SetNextWindowPos(
        ImVec2(ctx.ime.input_region.x, ctx.ime.input_region.y + ctx.ime.input_region.lh));
    ImGui::Begin("ImeEditor", nullptr,
                 ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoDecoration);
    ImGui::Text("%s", GetComposition(ctx.ime.state).c_str());
    ImGui::TextWrapped("%s", GetMenu(ctx.ime.state).c_str());
    ImGui::End();
}

std::string ImeEditor::GetComposition(ImeState& state)
{
    std::ostringstream ss;
    size_t len = state.preEdit.size();
    size_t start = state.selStart;
    size_t end = state.selEnd;
    size_t cursor = state.cursorPos;
    for (size_t i = 0; i <= len; ++i) {
        // if (start < end)
        // {
        //     if (i == start)
        //     {
        //         ss << '[';
        //     }
        //     else if (i == end)
        //     {
        //         ss << ']';
        //     }
        // }
        if (i == cursor) ss << '|';
        if (i < len) ss << state.preEdit[i];
    }
    return ss.str();
}

std::string ImeEditor::GetMenu(ImeState& state)
{
    std::ostringstream ss;
    for (int i = 0; i < state.candidates.size(); ++i) {
        if (i > 0) {
            ss << "  ";
        }
        ss << i + 1 << ". " << state.candidates.at(i);
    }
    return ss.str();
}
