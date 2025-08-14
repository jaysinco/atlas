#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <iostream>
#include "toolkit/toolkit.h"

cl::Device getDefaultDevice()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto& p: platforms) {
        std::cout << "===== PLATFORM =====\n";
        std::cout << "CL_PLATFORM_PROFILE: " << p.getInfo<CL_PLATFORM_PROFILE>() << "\n";
        std::cout << "CL_PLATFORM_VERSION: " << p.getInfo<CL_PLATFORM_VERSION>() << "\n";
        std::cout << "CL_PLATFORM_NAME: " << p.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cout << "CL_PLATFORM_VENDOR: " << p.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cout << "CL_PLATFORM_EXTENSIONS: " << p.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
        std::cout << "CL_PLATFORM_HOST_TIMER_RESOLUTION: "
                  << p.getInfo<CL_PLATFORM_HOST_TIMER_RESOLUTION>() << "\n";
    }

    if (platforms.empty()) {
        std::cerr << "No platforms found!" << std::endl;
        exit(1);
    }

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    for (auto& d: devices) {
        std::cout << "===== DEVICE =====\n";
        std::cout << "CL_DEVICE_TYPE: " << d.getInfo<CL_DEVICE_TYPE>() << "\n";
        std::cout << "CL_DEVICE_NAME: " << d.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "CL_DEVICE_VENDOR: " << d.getInfo<CL_DEVICE_VENDOR>() << "\n";
        std::cout << "CL_DRIVER_VERSION: " << d.getInfo<CL_DRIVER_VERSION>() << "\n";
        std::cout << "CL_DEVICE_PROFILE: " << d.getInfo<CL_DEVICE_PROFILE>() << "\n";
        std::cout << "CL_DEVICE_VERSION: " << d.getInfo<CL_DEVICE_VERSION>() << "\n";
        std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
                  << " bytes\n";
        std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
                  << "M\n";
        std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                  << "\n";

        auto max_items = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        for (int i = 0; i < max_items.size(); ++i) {
            std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES[" << i << "]: " << max_items[i] << "\n";
        }
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                  << "\n";
    }

    if (devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    return devices.front();
}

void sumArrayGpu(int* a, int* b, int* c, int const n, cl::Context& context, cl::Program& program,
                 cl::Device& device)
{
    cl::Buffer buf_a(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     n * sizeof(int), a);
    cl::Buffer buf_b(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                     n * sizeof(int), b);
    cl::Buffer buf_c(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n * sizeof(int));

    cl::Kernel kernel(program, "sumArrays");
    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_c);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(1024));
    queue.enqueueReadBuffer(buf_c, CL_TRUE, 0, n * sizeof(int), c);
}

void sumArrayCpu(int const* a, int const* b, int* c, int const n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

bool checkEqual(int const* c1, int const* c2, int const n)
{
    for (int i = 0; i < n; i++) {
        if (c1[i] != c2[i]) {
            return false;
        }
    }
    return true;
}

// NOLINTNEXTLINE
XXD_DECLARE_RES(DATA_HELLO_CL)

MY_MAIN
{
    int const count = 10;
    int const narr = 1 << 25;
    std::vector<int> a(narr, 3);
    std::vector<int> b(narr, 5);

    // cpu
    std::vector<int> c1(narr);
    clock_t start1 = clock();
    for (int i = 0; i < count; i++) {
        sumArrayCpu(a.data(), b.data(), c1.data(), narr);
    }
    clock_t end1 = clock();
    double time1 = (1e3 * (end1 - start1)) / CLOCKS_PER_SEC / count;

    // gpu
    auto device = getDefaultDevice();
    std::string src(XXD_GET_RES(DATA_HELLO_CL));
    cl::Context context({device});
    cl::Program program(context, {src});

    auto err = program.build(device, "-cl-std=CL2.0");
    if (err != CL_BUILD_SUCCESS) {
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
                  << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                  << std::endl;
        exit(1);
    }

    std::vector<int> c2(narr);
    clock_t start2 = clock();
    for (int i = 0; i < count; i++) {
        sumArrayGpu(a.data(), b.data(), c2.data(), narr, context, program, device);
    }
    clock_t end2 = clock();
    double time2 = (1e3 * (end2 - start2)) / CLOCKS_PER_SEC / count;

    // report
    bool equal = checkEqual(c1.data(), c2.data(), narr);
    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0]
              << "\n\tc[0] = a[0] + b[0] = " << c1[0] << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << time1 << " ms;\n\tParallel: " << time2
              << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (time1 - time2) / time2) << "%\n";

    return MyErrCode::kOk;
}
