#include "log-view.h"
#include "text-view.h"
#include "context.h"
#include "toolkit/format.h"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/elements.hpp>

LogView::LogView(int line_per_page): line_per_page_(line_per_page)
{
    curr_page_ = 1;
    total_page_ = 0;
    text_view_ = ftxui::Make<TextView>();
    body_ = ftxui::Container::Vertical({
        ftxui::Container::Horizontal({
            Button(
                "<<", [&] { beginPage(); }, ftxui::ButtonOption::Ascii()),
            Button(
                "<", [&] { prevPage(); }, ftxui::ButtonOption::Ascii()),
            ftxui::Renderer([&] { return ftxui::text(FSTR(" {}/{} ", curr_page_, total_page_)); }),
            Button(
                ">", [&] { nextPage(); }, ftxui::ButtonOption::Ascii()),
            Button(
                ">>", [&] { endPage(); }, ftxui::ButtonOption::Ascii()),
        }),
        text_view_,
    });
    Add(body_);
}

ftxui::Element LogView::Render()
{
    text_view_->clear();
    {
        int64_t total_store = Context::instance().getLogSize();
        total_page_ = std::ceil(total_store / static_cast<float>(line_per_page_));
        updateCurrPage();
        for (int i = (curr_page_ - 1) * line_per_page_ + 1; i <= curr_page_ * line_per_page_; ++i) {
            if (i <= total_store) {
                toolkit::LogLevel level;
                std::string_view mesg;
                Context::instance().getLog(i - 1, level, mesg);
                text_view_->addText(mesg);
            }
        }
    }
    return ftxui::ComponentBase::Render();
}

void LogView::beginPage() { curr_page_ = 1; }

void LogView::endPage() { curr_page_ = total_page_; }

void LogView::nextPage()
{
    ++curr_page_;
    updateCurrPage();
}

void LogView::prevPage()
{
    --curr_page_;
    updateCurrPage();
}

void LogView::updateCurrPage() { curr_page_ = std::max(1L, std::min(total_page_, curr_page_)); }
