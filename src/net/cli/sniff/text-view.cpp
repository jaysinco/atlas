#include "text-view.h"
#include <ftxui/component/component.hpp>

TextView::TextView() { selected_ = 0; }

void TextView::clear() { texts_.clear(); }

void TextView::addText(std::string_view content) { texts_.emplace_back(content); }

ftxui::Element TextView::Render()
{
    std::vector<ftxui::Element> elems;
    for (int i = 0; i < texts_.size(); ++i) {
        if (i == selected_) {
            elems.push_back(ftxui::paragraph(texts_.at(i)) | ftxui::inverted | ftxui::focus);
        } else {
            elems.push_back(ftxui::paragraph(texts_.at(i)));
        }
    }
    return ftxui::vbox(std::move(elems)) | ftxui::vscroll_indicator | ftxui::yframe | ftxui::yflex;
}

bool TextView::OnEvent(ftxui::Event event)
{
    if (event.is_mouse() && event.mouse().button == ftxui::Mouse::WheelUp) {
        prevSelection();
        return true;
    }
    if (event.is_mouse() && event.mouse().button == ftxui::Mouse::WheelDown) {
        nextSelection();
        return true;
    }
    return false;
}

bool TextView::Focusable() const { return true; }

void TextView::nextSelection()
{
    ++selected_;
    updateSelection();
}

void TextView::prevSelection()
{
    --selected_;
    updateSelection();
}

void TextView::updateSelection()
{
    selected_ = std::max(0, std::min(static_cast<int>(texts_.size() - 1), selected_));
}