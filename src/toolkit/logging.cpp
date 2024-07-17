#include "logging.h"
#include "toolkit.h"
#include <spdlog/spdlog.h>
#include <spdlog/details/os.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace toolkit
{

LoggerOption::LoggerOption()
{
    program = currentExeName();
    logtostderr = true;
    logtofile = false;
    loglevel = LogLevel::kINFO;
    logbuflevel = LogLevel::kERROR;
    logbufsecs = 30;
    logdir = getLoggingDir();
    maxlogsize = 100;
}

static spdlog::level::level_enum level(LogLevel level)
{
    if (static_cast<int>(level) < 0 || level >= LogLevel::kTOTAL) {
        ELOG("invalid log level: {}", static_cast<int>(level));
        return spdlog::level::info;
    }
    return static_cast<spdlog::level::level_enum>(level);
}

MyErrCode initLogger(LoggerOption const& opt)
{
    if (spdlog::get(opt.program)) {
        return MyErrCode::kOk;
    }

    std::vector<spdlog::sink_ptr> sinks;
    if (opt.logtofile) {
        std::filesystem::path fpath(opt.logdir);
        std::string fname =
            FSTR("{}_{:%Y%m%d.%H%M%S}_{}.log", std::filesystem::path(opt.program).stem().string(),
                 spdlog::details::os::localtime(), spdlog::details::os::pid());
        fpath /= fname;

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            fpath.string(), opt.maxlogsize * 1024 * 1024, 10, true);
        file_sink->set_level(spdlog::level::trace);
        file_sink->set_pattern("%L%m%d %P %t %H:%M:%S.%f] %v");
        sinks.push_back(file_sink);
    }
    if (opt.logtostderr) {
        auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        console_sink->set_level(spdlog::level::trace);  //
        console_sink->set_pattern("%^%L%m%d %P %t %H:%M:%S.%f] %v%$");
        sinks.push_back(console_sink);
    }

    auto logger = std::make_shared<spdlog::logger>(opt.program, sinks.begin(), sinks.end());
    logger->set_level(level(opt.loglevel));
    logger->flush_on(level(opt.logbuflevel));
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(opt.logbufsecs));

    ILOG("### GIT HASH: {} ###", _GIT_HASH);
    ILOG("### GIT BRANCH: {} ###", _GIT_BRANCH);
    ILOG("### BUILD AT: {} {} ###", __DATE__, __TIME__);
    return MyErrCode::kOk;
}

void logPrint(LogLevel level, std::string_view content)
{
    spdlog::default_logger_raw()->log(static_cast<spdlog::level::level_enum>(level), content);
}

void logPrint(LogLevel level, char const* filepath, int line, std::string_view content)
{
    logPrint(level, FSTR("[{}:{}] {}", std::filesystem::path(filepath).filename().string(), line,
                         content));
}

}  // namespace toolkit
