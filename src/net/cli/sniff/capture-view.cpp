#include "capture-view.h"
#include "context.h"
#include <ftxui/component/component.hpp>
#include <ftxui/dom/table.hpp>
#include <ftxui/dom/elements.hpp>
#include "toolkit/format.h"

CaptureView::CaptureView(int line_per_page): line_per_page_(line_per_page)
{
    curr_page_ = 1;
    is_capturing_ = false;

    auto cap_btn_opt = ftxui::ButtonOption::Ascii();
    cap_btn_opt.transform = [&](ftxui::EntryState const& s) {
        std::string sep_l = s.focused ? "[" : " ";
        std::string sep_r = s.focused ? "]" : " ";
        std::string text = fmt::format("{}{}{}", sep_l, is_capturing_ ? "stop" : "start", sep_r);
        auto element = ftxui::text(text);
        return element;
    };

    body_ = ftxui::Container::Vertical({
        ftxui::Container::Horizontal({
            Button(
                "<<", [&] { beginPage(); }, ftxui::ButtonOption::Ascii()),
            Button(
                "<", [&] { prevPage(); }, ftxui::ButtonOption::Ascii()),
            ftxui::Renderer(
                [&] { return ftxui::text(FSTR(" {}/{} ", curr_page_, getTotalPage())); }),
            Button(
                ">", [&] { nextPage(); }, ftxui::ButtonOption::Ascii()),
            Button(
                ">>", [&] { endPage(); }, ftxui::ButtonOption::Ascii()),
            Button(
                "", [&] { switchCapture(); }, cap_btn_opt),
        }),
        ftxui::Renderer([&] { return renderPackets(); }),
    });
    Add(body_);
}

ftxui::Element CaptureView::renderPackets()
{
    std::vector<std::vector<std::string>> pac_list;
    pac_list.push_back({"time", "type"});
    for (int i = (curr_page_ - 1) * line_per_page_ + 1; i <= curr_page_ * line_per_page_; ++i) {
        net::Packet pac;
        if (!Context::instance().getPacket(i - 1, pac)) {
            break;
        }
        std::vector<std::string> pac_info;
        auto& layers = pac.getDetail().layers;
        // time
        std::time_t pac_time = std::chrono::system_clock::to_time_t(pac.getDetail().time);
        std::string pac_time_str(30, '\0');
        std::strftime(&pac_time_str[0], pac_time_str.size(), "%H:%M:%S", std::localtime(&pac_time));
        uint64_t pac_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   pac.getDetail().time -
                                   std::chrono::floor<std::chrono::seconds>(pac.getDetail().time))
                                   .count();
        pac_info.push_back(FSTR("{}.{:06d}", pac_time_str, pac_time_us));
        // type
        if (layers.size() >= 2) {
            pac_info.push_back(TOSTR(layers.at(1)->type()));
        } else {
            pac_info.push_back(TOSTR(layers.at(0)->type()));
        }
        pac_list.push_back(pac_info);
    }
    ftxui::Table table(pac_list);
    table.SelectAll().SeparatorVertical(ftxui::LIGHT);
    table.SelectAll().Border(ftxui::LIGHT);
    table.SelectRow(0).Border(ftxui::LIGHT);
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
    curr_page_ = std::numeric_limits<int64_t>::max();
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

void CaptureView::updateCurrPage()
{
    curr_page_ = std::max(1L, std::min(getTotalPage(), curr_page_));
}

void CaptureView::switchCapture()
{
    if (!is_capturing_) {
        curr_page_ = 1;
        is_capturing_ = true;
        Context::instance().startCapture();
    } else {
        is_capturing_ = false;
        Context::instance().stopCapture();
    }
}

void CaptureView::clearCapture()
{
    Context::instance().clearCapture();
    updateCurrPage();
}

int64_t CaptureView::getTotalPage()
{
    int64_t total_store = Context::instance().getPacketSize();
    return std::ceil(total_store / static_cast<float>(line_per_page_));
}
