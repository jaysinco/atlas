#pragma once
#include <ftxui/component/component_base.hpp>

class TextView;

class LogView: public ftxui::ComponentBase
{
public:
    explicit LogView(int line_per_page);
    ftxui::Element Render() override;

    void beginPage();
    void endPage();
    void nextPage();
    void prevPage();

private:
    void updateCurrPage();

private:
    ftxui::Component body_;
    std::shared_ptr<TextView> text_view_;
    int const line_per_page_;
    int64_t curr_page_;
    int64_t total_page_;
};