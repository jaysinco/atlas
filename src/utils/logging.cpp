#include "logging.h"
#include <fmt/chrono.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace utils
{

static spdlog::level::level_enum level(LogLevel level)
{
    if (level < 0 || level >= kTOTAL) {
        MY_THROW("invalid log level: {}", level);
    }
    return static_cast<spdlog::level::level_enum>(level);
}

void initLogger(std::string const& program, bool logtostderr, bool logtofile, LogLevel minloglevel,
                LogLevel logbuflevel, int logbufsecs, std::filesystem::path const& logdir,
                int maxlogsize)
{
    if (spdlog::get(program)) {
        return;
    }
    std::vector<spdlog::sink_ptr> sinks;
    if (logtofile) {
        std::filesystem::path fpath(logdir);
        std::string fname =
            FSTR("{}_{:%Y%m%d.%H%M%S}_{}.log", std::filesystem::path(program).stem().string(),
                 spdlog::details::os::localtime(), spdlog::details::os::pid());
        fpath /= fname;

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            fpath.string(), maxlogsize * 1024 * 1024, 10, true);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("%L%m%d %H:%M:%S.%f %t %s:%#] %v");
        sinks.push_back(file_sink);
    }
    if (logtostderr) {
        auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);  //
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        sinks.push_back(console_sink);
    }

    auto logger = std::make_shared<spdlog::logger>(program, sinks.begin(), sinks.end());
    logger->set_level(level(minloglevel));
    logger->flush_on(level(logbuflevel));
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(logbufsecs));

    DLOG("### GIT HASH: {} ###", _GIT_HASH);
    DLOG("### GIT BRANCH: {} ###", _GIT_BRANCH);
    DLOG("### BUILD AT: {} {} ###", __DATE__, __TIME__);
}

}  // namespace utils
