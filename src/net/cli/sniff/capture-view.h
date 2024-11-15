#pragma once
#include <ftxui/component/component_base.hpp>

class CaptureView: public ftxui::ComponentBase
{
public:
    CaptureView();

private:
    ftxui::Component body_;
    std::vector<std::string> tab_entries_;
    int tab_selected_;
};
