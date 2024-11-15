#pragma once
#include <ftxui/component/component_base.hpp>

class TextView: public ftxui::ComponentBase
{
public:
    TextView();
    ftxui::Element Render() override;
    bool OnEvent(ftxui::Event event) override;
    bool Focusable() const override;

    void clear();
    void addText(std::string_view content);

private:
    void nextSelection();
    void prevSelection();
    void updateSelection();

private:
    std::vector<std::string> texts_;
    int selected_;
};