#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include <mutex>
#include <atomic>
#include <future>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/table.hpp>

std::mutex g_log_lock;
std::vector<std::pair<toolkit::LogLevel, std::string>> g_log_store;

std::mutex g_packet_lock;
std::vector<net::Packet> g_packet_store;

std::atomic<bool> g_capture_should_stop;
std::future<MyErrCode> g_capture_res;

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

MyErrCode drawTui(toolkit::Args const& args)
{
    using namespace ftxui;

    auto opt_ip = args.get<std::string>("ip");
    auto opt_filter = args.get<std::string>("filter");

    auto screen = ftxui::ScreenInteractive::Fullscreen();

    std::vector<std::string> tab_values{
        "capture",
        "log",
    };
    int tab_selected = 0;
    auto tab_toggle = Toggle(&tab_values, &tab_selected);

    std::vector<std::string> tab_1_entries{
        "Forest",
        "Water",
        "I don't know",
    };
    int tab_1_selected = 0;

    auto tab_container = Container::Tab(
        {
            Radiobox(&tab_1_entries, &tab_1_selected),
            Renderer([&] {
                auto table = Table({
                    {"Version", "Marketing name", "Release date", "API level", "Runtime"},
                    {"2.3", "Gingerbread", "February 9 2011", "10", "Dalvik 1.4.0"},
                    {"4.0", "Ice Cream Sandwich", "October 19 2011", "15", "Dalvik"},
                    {"4.1", "Jelly Bean", "July 9 2012", "16", "Dalvik"},
                    {"4.2", "Jelly Bean", "November 13 2012", "17", "Dalvik"},
                    {"4.3", "Jelly Bean", "July 24 2013", "18", "Dalvik"},
                    {"4.4", "KitKat", "October 31 2013", "19", "Dalvik and ART"},
                    {"5.0", "Lollipop", "November 3 2014", "21", "ART"},
                    {"5.1", "Lollipop", "March 9 2015", "22", "ART"},
                    {"6.0", "Marshmallow", "October 5 2015", "23", "ART"},
                    {"7.0", "Nougat", "August 22 2016", "24", "ART"},
                    {"7.1", "Nougat", "October 4 2016", "25", "ART"},
                    {"8.0", "Oreo", "August 21 2017", "26", "ART"},
                    {"8.1", "Oreo", "December 5 2017", "27", "ART"},
                    {"9", "Pie", "August 6 2018", "28", "ART"},
                    {"10", "10", "September 3 2019", "29", "ART"},
                    {"11", "11", "September 8 2020", "30", "ART"},
                });
                table.SelectAll().Border(LIGHT);

                // Add border around the first column.
                table.SelectColumn(0).Border(LIGHT);

                // Make first row bold with a double border.
                table.SelectRow(0).Decorate(bold);
                table.SelectRow(0).SeparatorVertical(LIGHT);
                table.SelectRow(0).Border(DOUBLE);

                // Align right the "Release date" column.
                table.SelectColumn(2).DecorateCells(align_right);

                // Select row from the second to the last.
                auto content = table.SelectRows(1, -1);
                // Alternate in between 3 colors.
                content.DecorateCellsAlternateRow(color(Color::Blue), 3, 0);
                content.DecorateCellsAlternateRow(color(Color::Cyan), 3, 1);
                content.DecorateCellsAlternateRow(color(Color::White), 3, 2);

                return table.Render();
            }),
        },
        &tab_selected);

    auto container = Container::Vertical({
        tab_toggle,
        tab_container,
    });

    auto renderer = Renderer(container, [&] {
        return vbox({
                   tab_toggle->Render(),
                   separator(),
                   tab_container->Render(),
               }) |
               border;
    });

    screen.Loop(renderer);
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