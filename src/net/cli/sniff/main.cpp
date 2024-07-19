#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include <mutex>
#include <atomic>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

std::mutex g_log_lock;
std::vector<std::pair<toolkit::LogLevel, std::string>> g_log_store;
std::mutex g_packet_lock;
std::vector<net::Packet> g_packet_store;
std::atomic<bool> g_capture_should_stop;

void startCapture(std::string const& ipstr, std::string const& filter)
{
    std::thread([=]() -> MyErrCode {
        g_capture_should_stop = false;

        auto& apt = net::Adaptor::fit(!ipstr.empty() ? net::Ip4(ipstr) : net::Ip4::kZeros);
        void* handle;
        CHECK_ERR_RET(net::Transport::open(apt, handle));
        auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });
        ILOG(apt.toVariant().toJsonStr(3));

        if (!filter.empty()) {
            ILOG("set filter \"{}\", mask={}", filter, apt.mask);
            CHECK_ERR_RET(net::Transport::setFilter(handle, filter, apt.mask));
        }

        ILOG("begin to capture...");
        CHECK_ERR_RET(net::Transport::recv(handle, [&](net::Packet&& p) -> bool {
            std::lock_guard<std::mutex> packet_guard(g_packet_lock);
            g_packet_store.push_back(std::move(p));
            return g_capture_should_stop;
        }));
        ILOG("capture stopped");

        return MyErrCode::kOk;
    }).detach();
}

void stopCapture() { g_capture_should_stop = true; }

class TextView: public ftxui::ComponentBase
{
public:
    TextView() { selected_ = 0; }

    void clear() { texts_.clear(); }

    void addText(std::string const& content) { texts_.push_back(content); }

    ftxui::Element Render() override
    {
        std::vector<ftxui::Element> elems;
        for (int i = 0; i < texts_.size(); ++i) {
            if (i == selected_) {
                elems.push_back(ftxui::paragraph(texts_.at(i)) | ftxui::inverted | ftxui::focus);
            } else {
                elems.push_back(ftxui::paragraph(texts_.at(i)));
            }
        }
        return ftxui::vbox(std::move(elems)) | ftxui::vscroll_indicator | ftxui::yframe |
               ftxui::yflex;
    }

    bool OnEvent(ftxui::Event event) override
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

    bool Focusable() const override { return true; }

private:
    void nextSelection()
    {
        ++selected_;
        updateSelection();
    }

    void prevSelection()
    {
        --selected_;
        updateSelection();
    }

    void updateSelection()
    {
        selected_ = std::max(0, std::min(static_cast<int>(texts_.size() - 1), selected_));
    }

private:
    std::vector<std::string> texts_;
    int selected_;
};

class LogView: public ftxui::ComponentBase
{
public:
    LogView(int line_per_page): line_per_page_(line_per_page)
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
                ftxui::Renderer(
                    [&] { return ftxui::text(FSTR(" {}/{} ", curr_page_, total_page_)); }),
                Button(
                    ">", [&] { nextPage(); }, ftxui::ButtonOption::Ascii()),
                Button(
                    ">>", [&] { endPage(); }, ftxui::ButtonOption::Ascii()),
            }),
            text_view_,
        });
        Add(body_);
    }

    ftxui::Element Render() override
    {
        text_view_->clear();
        {
            std::lock_guard<std::mutex> log_guard(g_log_lock);
            int64_t total_store = g_log_store.size();
            total_page_ = std::ceil(total_store / static_cast<float>(line_per_page_));
            updateCurrPage();
            for (int i = (curr_page_ - 1) * line_per_page_ + 1; i <= curr_page_ * line_per_page_;
                 ++i) {
                if (i <= total_store) {
                    text_view_->addText(g_log_store.at(i - 1).second);
                }
            }
        }
        return ftxui::ComponentBase::Render();
    }

    void beginPage() { curr_page_ = 1; }

    void endPage() { curr_page_ = total_page_; }

    void nextPage()
    {
        ++curr_page_;
        updateCurrPage();
    }

    void prevPage()
    {
        --curr_page_;
        updateCurrPage();
    }

private:
    void updateCurrPage() { curr_page_ = std::max(1L, std::min(total_page_, curr_page_)); }

private:
    ftxui::Component body_;
    std::shared_ptr<TextView> text_view_;
    int line_per_page_;
    int64_t curr_page_;
    int64_t total_page_;
};

class CaptureView: public ftxui::ComponentBase
{
public:
    CaptureView()
        : tab_entries_({
              "Forest",
              "Water",
              "I don't know",
          })
    {
        body_ = ftxui::Radiobox(&tab_entries_, &tab_selected_);
        Add(body_);
    }

private:
    ftxui::Component body_;
    std::vector<std::string> tab_entries_;
    int tab_selected_;
};

class MainView: public ftxui::ComponentBase
{
public:
    MainView(): tab_values_({"capture", "log"})
    {
        log_view_ = ftxui::Make<LogView>(20);
        capture_view_ = ftxui::Make<CaptureView>();

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

    ftxui::Element Render() override
    {
        return ftxui::vbox({
                   tab_toggle_->Render(),
                   ftxui::separator(),
                   tab_container_->Render(),
               }) |
               ftxui::border;
    }

    bool OnEvent(ftxui::Event event) override
    {
        if (tab_selected_ == 1) {
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

private:
    ftxui::Component body_;
    ftxui::Component tab_toggle_;
    ftxui::Component tab_container_;
    std::shared_ptr<LogView> log_view_;
    std::shared_ptr<CaptureView> capture_view_;
    std::vector<std::string> tab_values_;
    int tab_selected_;
};

MyErrCode drawTui(toolkit::Args const& args)
{
    auto opt_ip = args.get<std::string>("ip");
    auto opt_filter = args.get<std::string>("filter");

    auto screen = ftxui::ScreenInteractive::Fullscreen();
    std::atomic<bool> refresh_ui_continue = true;
    std::thread refresh_ui([&] {
        while (refresh_ui_continue) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            screen.Post(ftxui::Event::Custom);
        }
    });

    ftxui::Component main_view = ftxui::Make<MainView>();
    main_view = CatchEvent(main_view, [&](ftxui::Event event) {
        if (event == ftxui::Event::Character('q')) {
            screen.Exit();
            return true;
        }
        return false;
    });

    screen.Loop(main_view);
    refresh_ui_continue = false;
    refresh_ui.join();

    return MyErrCode::kOk;
};

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::LoggerOption logger_opt;
    logger_opt.logtofile = false;
    logger_opt.logtostderr = false;
    logger_opt.callback = [](toolkit::LogLevel level, std::string_view mesg) {
        std::lock_guard<std::mutex> log_guard(g_log_lock);
        g_log_store.emplace_back(level, std::string(mesg));
    };
    CHECK_ERR_RET_INT(toolkit::initLogger(std::move(logger_opt)));

    toolkit::Args args(argc, argv);
    args.positional("ip", po::value<std::string>()->default_value(""), "ipv4 address", 1);
    args.optional("filter,f", po::value<std::string>()->default_value(""), "capture filter");
    CHECK_ERR_RET_INT(args.parse(false));
    CHECK_ERR_RET_INT(drawTui(args));

    MY_CATCH_RET_INT
}