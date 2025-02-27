#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <chrono>
#include "toolkit/logging.h"

namespace ocl
{

MyErrCode getDefaultDevice(cl::Device& dev);
MyErrCode buildProgram(cl::Context const& ctx, cl::Device const& dev, std::string const& src,
                       cl::Program& prog);

MyErrCode txiGuided(int argc, char** argv);
MyErrCode spatialDenoise(int argc, char** argv);

}  // namespace ocl