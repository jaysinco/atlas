#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include "utils/logging.h"

cl::Device get_default_device()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (auto& p: platforms) {
        std::cerr << "**********************\n";
        std::cerr << "CL_PLATFORM_PROFILE: " << p.getInfo<CL_PLATFORM_PROFILE>() << "\n";
        std::cerr << "CL_PLATFORM_VERSION: " << p.getInfo<CL_PLATFORM_VERSION>() << "\n";
        std::cerr << "CL_PLATFORM_NAME: " << p.getInfo<CL_PLATFORM_NAME>() << "\n";
        std::cerr << "CL_PLATFORM_VENDOR: " << p.getInfo<CL_PLATFORM_VENDOR>() << "\n";
        std::cerr << "CL_PLATFORM_EXTENSIONS: " << p.getInfo<CL_PLATFORM_EXTENSIONS>() << "\n";
        std::cerr << "CL_PLATFORM_HOST_TIMER_RESOLUTION: "
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
        std::cerr << "**********************\n";
        std::cerr << "CL_DEVICE_TYPE: " << d.getInfo<CL_DEVICE_TYPE>() << "\n";
        std::cerr << "CL_DEVICE_NAME: " << d.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cerr << "CL_DEVICE_VENDOR: " << d.getInfo<CL_DEVICE_VENDOR>() << "\n";
        std::cerr << "CL_DRIVER_VERSION: " << d.getInfo<CL_DRIVER_VERSION>() << "\n";
        std::cerr << "CL_DEVICE_PROFILE: " << d.getInfo<CL_DEVICE_PROFILE>() << "\n";
        std::cerr << "CL_DEVICE_VERSION: " << d.getInfo<CL_DEVICE_VERSION>() << "\n";
        std::cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
                  << " bytes\n";

        std::cerr << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()
                  << "M\n";
        std::cerr << "CL_DEVICE_MAX_COMPUTE_UNITS: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
                  << "\n";

        auto maxItemSizes = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
        for (int i = 0; i < maxItemSizes.size(); ++i) {
            std::cerr << "CL_DEVICE_MAX_WORK_ITEM_SIZES[" << i << "]: " << maxItemSizes[i] << "\n";
        }
        std::cerr << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
                  << "\n";
    }

    if (devices.empty()) {
        std::cerr << "No devices found!" << std::endl;
        exit(1);
    }

    return devices.front();
}

void arr_sum_gpu(int* a, int* b, int* c, int const n, cl::Context& context, cl::Program& program,
                 cl::Device& device)
{
    cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                    n * sizeof(int), a);
    cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
                    n * sizeof(int), b);
    cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, n * sizeof(int));

    cl::Kernel kernel(program, "sumArrays");

    kernel.setArg(0, aBuf);
    kernel.setArg(1, bBuf);
    kernel.setArg(2, cBuf);

    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(1024));
    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, n * sizeof(int), c);
}

void arr_sum_cpu(int* a, int* b, int* c, int const n)
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

bool check_equal(int* c1, int* c2, int const n)
{
    for (int i = 0; i < n; i++) {
        if (c1[i] != c2[i]) {
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    int const count = 10;
    int const narr = 1 << 25;

    std::vector<int> a(narr, 3);
    std::vector<int> b(narr, 5);

    // cpu
    std::vector<int> c1(narr);
    clock_t start1 = clock();
    for (int i = 0; i < count; i++) {
        arr_sum_cpu(a.data(), b.data(), c1.data(), narr);
    }
    clock_t end1 = clock();
    double seqTime1 = ((double)10e3 * (end1 - start1)) / CLOCKS_PER_SEC / count;

    // gpu
    auto device = get_default_device();

    std::string src = *utils::readFile(utils::projectRoot() / "src/test/res/hello.cl");

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
        arr_sum_gpu(a.data(), b.data(), c2.data(), narr, context, program, device);
    }
    clock_t end2 = clock();
    double seqTime2 = ((double)10e3 * (end2 - start2)) / CLOCKS_PER_SEC / count;

    // report
    bool equal = check_equal(c1.data(), c2.data(), narr);
    std::cout << "Status: " << (equal ? "SUCCESS!" : "FAILED!") << std::endl;
    std::cout << "Results: \n\ta[0] = " << a[0] << "\n\tb[0] = " << b[0]
              << "\n\tc[0] = a[0] + b[0] = " << c1[0] << std::endl;
    std::cout << "Mean execution time: \n\tSequential: " << seqTime1
              << " ms;\n\tParallel: " << seqTime2 << " ms." << std::endl;
    std::cout << "Performance gain: " << (100 * (seqTime1 - seqTime2) / seqTime2) << "%\n";

    return 0;
}
