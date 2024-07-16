#pragma once
#include "toolkit.h"
#include "format.h"

#define MY_TRY try {
#define MY_CATCH_FUNC(x)                    \
    }                                       \
    catch (std::exception const& err)       \
    {                                       \
        ELOG("[exception] {}", err.what()); \
        x;                                  \
    }
#define MY_CATCH_RET MY_CATCH_FUNC(return MyErrCode::kException)
#define MY_CATCH_RET_INT MY_CATCH_FUNC(return static_cast<int>(MyErrCode::kException))

#define LOG_FUNC(level, ...) toolkit::logPrint(level, FSTR(__VA_ARGS__))
#define LOG_FUNC_DETAILED(level, ...) \
    toolkit::logPrint(level, __FILE__, __LINE__, FSTR(__VA_ARGS__))

#define TLOG(...) (LOG_FUNC(toolkit::LogLevel::kTRACE, __VA_ARGS__))
#define DLOG(...) (LOG_FUNC(toolkit::LogLevel::kDEBUG, __VA_ARGS__))
#define ILOG(...) (LOG_FUNC(toolkit::LogLevel::kINFO, __VA_ARGS__))
#define WLOG(...) (LOG_FUNC_DETAILED(toolkit::LogLevel::kWARN, __VA_ARGS__))
#define ELOG(...) (LOG_FUNC_DETAILED(toolkit::LogLevel::kERROR, __VA_ARGS__))

namespace toolkit
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
                     std::filesystem::path const& logdir = getLoggingDir(), int maxlogsize = 100);

void logPrint(LogLevel level, std::string_view content);
void logPrint(LogLevel level, char const* filepath, int line, std::string_view content);

}  // namespace toolkit
