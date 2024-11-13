#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <chrono>
#include "toolkit/logging.h"

#define TIMER_BEGIN(x) auto timer_##x = std::chrono::system_clock::now();
#define TIMER_END(x, desc)                                                                        \
    ILOG("{}, elapsed={:.1f}ms", (desc),                                                          \
         std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - \
                                                               timer_##x)                         \
                 .count() /                                                                       \
             1000.);

namespace ocl
{

MyErrCode getDefaultDevice(cl::Device& dev);
MyErrCode buildProgram(cl::Context const& ctx, cl::Device const& dev, std::string const& src,
                       cl::Program& prog);

MyErrCode txiGuided(int argc, char** argv);

}  // namespace ocl