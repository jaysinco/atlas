#include "homepage-view.h"
#include <ftxui/component/component.hpp>
#include "capture-view.h"
#include "log-view.h"

HomepageView::HomepageView(): tab_values_({"capture", "log"})
{
    log_view_ = ftxui::Make<LogView>(20);
    capture_view_ = ftxui::Make<CaptureView>(20);

    tab_toggle_ = ftxui::Toggle(&tab_values_, &tab_selected_);
    tab_container_ = ftxui::Container::Tab(
        {
            capture_view_,
            log_view_,
        },
        &tab_selected_);

    body_ = ftxui::Container::Vertical({
        tab_toggle_,
        tab_container_,
    });

    Add(body_);
}

ftxui::Element HomepageView::Render()
{
    return ftxui::vbox({
               tab_toggle_->Render(),
               ftxui::separator(),
               tab_container_->Render(),
           }) |
           ftxui::border;
}

bool HomepageView::OnEvent(ftxui::Event event)
{
    if (tab_selected_ == 0) {
        if (event == ftxui::Event::PageDown) {
            capture_view_->nextPage();
            return true;
        } else if (event == ftxui::Event::PageUp) {
            capture_view_->prevPage();
            return true;
        } else if (event == ftxui::Event::Home) {
            capture_view_->beginPage();
            return true;
        } else if (event == ftxui::Event::End) {
            capture_view_->endPage();
            return true;
        } else if (event == ftxui::Event::Character(' ')) {
            capture_view_->switchCapture();
            return true;
        }
    } else if (tab_selected_ == 1) {
        if (event == ftxui::Event::PageDown) {
            log_view_->nextPage();
            return true;
        } else if (event == ftxui::Event::PageUp) {
            log_view_->prevPage();
            return true;
        } else if (event == ftxui::Event::Home) {
            log_view_->beginPage();
            return true;
        } else if (event == ftxui::Event::End) {
            log_view_->endPage();
            return true;
        }
    }
    return ftxui::ComponentBase::OnEvent(event);
}