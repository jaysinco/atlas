#include "capture-view.h"
#include "context.h"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/table.hpp>
#include <ftxui/dom/elements.hpp>
#include "toolkit/format.h"

CaptureView::CaptureView(int line_per_page): line_per_page_(line_per_page)
{
    curr_page_ = 1;
    total_page_ = 0;
    is_capturing_ = false;

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
        ftxui::Renderer([&] { return renderPackets(); }),
    });
    Add(body_);
}

ftxui::Element CaptureView::renderPackets()
{
    int64_t total_store = Context::instance().getPacketSize();
    total_page_ = std::ceil(total_store / static_cast<float>(line_per_page_));
    std::vector<std::vector<std::string>> pac_list;
    for (int i = (curr_page_ - 1) * line_per_page_ + 1; i <= curr_page_ * line_per_page_; ++i) {
        if (i <= total_store) {
            net::Packet pac;
            if (!Context::instance().getPacket(i - 1, pac)) {
                break;
            }
            std::vector<std::string> pac_info;
            pac_info.push_back(pac.toVariant().toJsonStr());
            pac_list.push_back(pac_info);
        }
    }
    ftxui::Table table(pac_list);
    return table.Render();
}

ftxui::Element CaptureView::Render() { return ftxui::ComponentBase::Render(); }

void CaptureView::beginPage()
{
    curr_page_ = 1;
    updateCurrPage();
}

void CaptureView::endPage()
{
    curr_page_ = total_page_;
    updateCurrPage();
}

void CaptureView::nextPage()
{
    ++curr_page_;
    updateCurrPage();
}

void CaptureView::prevPage()
{
    --curr_page_;
    updateCurrPage();
}

void CaptureView::updateCurrPage() { curr_page_ = std::max(1L, std::min(total_page_, curr_page_)); }

void CaptureView::startCapture()
{
    Context::instance().startCapture();
    is_capturing_ = true;
}

void CaptureView::stopCapture()
{
    Context::instance().stopCapture();
    is_capturing_ = false;
    curr_page_ = 1;
    total_page_ = 0;
}