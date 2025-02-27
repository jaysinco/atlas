#pragma once
#include "logging.h"
#include <boost/timer/timer.hpp>

#define MY_TIMER_BEGIN(level, desc) \
    {                               \
        toolkit::AutoCpuTimer MY_CONCAT(_my_timer_, __LINE__)(desc, toolkit::LogLevel::k##level);
#define MY_TIMER_END }

namespace toolkit
{

class AutoCpuTimer: public boost::timer::cpu_timer
{
public:
    AutoCpuTimer(std::string const& desc, toolkit::LogLevel level = toolkit::LogLevel::kINFO,
                 int16_t places = 3,
                 std::string const& format = "%ws wall, %us user + %ss system = %ts CPU (%p%)")
        : desc_(desc), level_(level), places_(places), format_(format)
    {
        start();
    }

    ~AutoCpuTimer()
    {
        stop();
        LOG_FUNC(level_, "time of {}: {}", desc_, format(places_, format_));
    }

private:
    std::string desc_;
    toolkit::LogLevel level_;
    int16_t places_;
    std::string format_;
};

}  // namespace toolkit
