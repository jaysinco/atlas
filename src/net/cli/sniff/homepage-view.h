#pragma once
#include <ftxui/component/component_base.hpp>

class LogView;
class CaptureView;

class HomepageView: public ftxui::ComponentBase
{
public:
    HomepageView();
    ftxui::Element Render() override;
    bool OnEvent(ftxui::Event event) override;

private:
    ftxui::Component body_;
    ftxui::Component tab_toggle_;
    ftxui::Component tab_container_;
    std::shared_ptr<LogView> log_view_;
    std::shared_ptr<CaptureView> capture_view_;
    std::vector<std::string> tab_values_;
    int tab_selected_;
};
