#pragma once
#include "base.h"
#include <spdlog/spdlog.h>

#define FSTR(f, ...) (fmt::format(f, __VA_ARGS__))
#define LOG_FSTR(f, ...) (FSTR("[{}:{}] " f, CURR_FILENAME, __LINE__, __VA_ARGS__))
#define MY_THROW(f, ...) throw std::runtime_error(LOG_FSTR(f, __VA_ARGS__))
#define MY_TRY try {
#define MY_CATCH                            \
    }                                       \
    catch (const std::exception& err)       \
    {                                       \
        ELOG("[exception] {}", err.what()); \
    }
#define LOG_FUNC(level, ...) SPDLOG_LOGGER_CALL(spdlog::default_logger_raw(), level, __VA_ARGS__)
#define TLOG(...) (LOG_FUNC(spdlog::level::trace, __VA_ARGS__))
#define DLOG(...) (LOG_FUNC(spdlog::level::debug, __VA_ARGS__))
#define ILOG(...) (LOG_FUNC(spdlog::level::info, __VA_ARGS__))
#define WLOG(...) (LOG_FUNC(spdlog::level::warn, __VA_ARGS__))
#define ELOG(...) (LOG_FUNC(spdlog::level::err, __VA_ARGS__))

namespace utils
{

enum class LogLevel : int
{
    kTRACE = 0,
    kDEBUG = 1,
    kINFO = 2,
    kWARN = 3,
    kERROR = 4,
    kCRITICAL = 5,
    kOFF = 6,
    kTOTAL,
};

MyErrCode initLogger(std::string const& program = currentExeName(), bool logtostderr = true,
                     bool logtofile = false, LogLevel minloglevel = LogLevel::kINFO,
                     LogLevel logbuflevel = LogLevel::kERROR, int logbufsecs = 30,
                     std::filesystem::path const& logdir = defaultLoggingDir(),
                     int maxlogsize = 100);

}  // namespace utils
