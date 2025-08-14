#pragma once
#include "error.h"
#include "format.h"
#include <functional>

#define MY_TRY try {
#define MY_CATCH_RET                        \
    }                                       \
    catch (std::exception const& err)       \
    {                                       \
        ELOG("[exception] {}", err.what()); \
        return MyErrCode::kException;       \
    }

#define LOG_FUNC(level, ...) toolkit::logPrint(level, FSTR(__VA_ARGS__))
#define LOG_FUNC_DETAILED(level, ...) \
    toolkit::logPrint(level, __FILE__, __LINE__, FSTR(__VA_ARGS__))

#ifdef TLOG
#undef TLOG
#endif
#ifdef DLOG
#undef DLOG
#endif
#ifdef ILOG
#undef ILOG
#endif
#ifdef WLOG
#undef WLOG
#endif
#ifdef ELOG
#undef ELOG
#endif
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

using LoggerCallback = std::function<void(LogLevel, std::string_view)>;

struct LoggerOption
{
    std::string program;
    bool logtostderr;
    bool logtofile;
    LogLevel loglevel;
    LogLevel logbuflevel;
    int logbufsecs;
    std::filesystem::path logdir;
    int maxlogsize;
    LoggerCallback callback;

    LoggerOption();
};

MyErrCode initLogger(LoggerOption&& opt = LoggerOption{});
void logPrint(LogLevel level, std::string_view content);
void logPrint(LogLevel level, char const* filepath, int line, std::string_view content);

}  // namespace toolkit
