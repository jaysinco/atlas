#include "./common.h"

namespace ocl
{

MyErrCode getDefaultDevice(cl::Device& dev)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        ELOG("no platforms found!");
        return MyErrCode::kFailed;
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (devices.empty()) {
        ELOG("no devices found!");
        return MyErrCode::kFailed;
    }

    dev = devices.front();
    return MyErrCode::kOk;
}

MyErrCode buildProgram(cl::Context const& ctx, cl::Device const& dev, std::string const& src,
                       cl::Program& prog)
{
    prog = cl::Program(ctx, {src});
    if (auto err = prog.build(dev, "-cl-std=CL2.0"); err != CL_BUILD_SUCCESS) {
        ELOG("failed to build opencl kernel: ({}) {}",
             prog.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev),
             prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

}  // namespace ocl