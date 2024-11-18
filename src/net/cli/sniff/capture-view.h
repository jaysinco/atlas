#pragma once
#include <ftxui/component/component_base.hpp>
#include <ftxui/dom/elements.hpp>

class CaptureView: public ftxui::ComponentBase
{
public:
    explicit CaptureView(int line_per_page);
    ftxui::Element Render() override;

    void beginPage();
    void endPage();
    void nextPage();
    void prevPage();
    void switchCapture();
    void clearCapture();

private:
    void updateCurrPage();
    int64_t getTotalPage();
    ftxui::Element renderPackets();

private:
    ftxui::Component body_;
    bool is_capturing_;
    int const line_per_page_;
    int64_t curr_page_;
};
