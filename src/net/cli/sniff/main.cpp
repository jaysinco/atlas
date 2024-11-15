
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "homepage-view.h"
#include "context.h"
#include <atomic>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

MyErrCode drawTui(toolkit::Args const& args)
{
    auto opt_ip = args.get<std::string>("ip");
    auto opt_filter = args.get<std::string>("filter");
    Context::instance().setupCapture(opt_ip, opt_filter);

    auto screen = ftxui::ScreenInteractive::Fullscreen();
    std::atomic<bool> refresh_ui_continue = true;
    std::thread refresh_ui([&] {
        while (refresh_ui_continue) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            screen.Post(ftxui::Event::Custom);
        }
    });

    ftxui::Component main_view = ftxui::Make<HomepageView>();
    main_view = ftxui::CatchEvent(main_view, [&](ftxui::Event event) {
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
        Context::instance().pushLog(level, mesg);
    };
    CHECK_ERR_RET_INT(toolkit::initLogger(std::move(logger_opt)));

    toolkit::Args args(argc, argv);
    args.positional("ip", po::value<std::string>()->default_value(""), "ipv4 address", 1);
    args.optional("filter,f", po::value<std::string>()->default_value(""), "capture filter");
    CHECK_ERR_RET_INT(args.parse(false));
    CHECK_ERR_RET_INT(drawTui(args));

    MY_CATCH_RET_INT
}