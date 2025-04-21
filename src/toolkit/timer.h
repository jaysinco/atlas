#pragma once
#include "logging.h"
#include <boost/timer/timer.hpp>

#define MY_TIMER_BEGIN(level, ...)                                               \
    {                                                                            \
        toolkit::AutoCpuTimer MY_CONCAT(_my_timer_, __LINE__)(FSTR(__VA_ARGS__), \
                                                              toolkit::LogLevel::k##level);
#define MY_TIMER_END() }

namespace toolkit
{

class AutoCpuTimer: public boost::timer::cpu_timer
{
public:
    AutoCpuTimer(std::string const& desc, toolkit::LogLevel level = toolkit::LogLevel::kINFO)
        : desc_(desc), level_(level)
    {
        // start();
    }

    ~AutoCpuTimer()
    {
        stop();
        auto tm = elapsed();
        LOG_FUNC(level_, "{}: {:.1f}ms elapsed", desc_, tm.wall * 1e-6);
    }

private:
    std::string desc_;
    toolkit::LogLevel level_;
};

}  // namespace toolkit
