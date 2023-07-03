#pragma once
#include "fs.h"
#include "encoding.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#define FSTR(f, ...) (fmt::format(f, __VA_ARGS__))
#define LOG_FSTR(f, ...) (FSTR("[{}:{}] " f, CURR_FILENAME, __LINE__, __VA_ARGS__))
#define MY_THROW(f, ...) throw utils::Error(LOG_FSTR(f, __VA_ARGS__))
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

enum LogLevel : int
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

void initLogger(std::string const& program, bool logtostderr, bool logtofile, LogLevel minloglevel,
                LogLevel logbuflevel, int logbufsecs, std::filesystem::path const& logdir,
                int maxlogsize);

}  // namespace utils
