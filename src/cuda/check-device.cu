#include "./fwd.h"
#include <cuda_runtime.h>
#include <stdio.h>

MyErrCode checkDevice(int argc, char** argv)
{
    printf("%s Starting...\n", argv[0]);
    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id),
               cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (device_count == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", device_count);
    }
    int dev, driver_version = 0, runtime_version = 0;
    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("Device %d: \"%s\"\n", dev, device_prop.name);
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driver_version / 1000,
           (driver_version % 100) / 10, runtime_version / 1000, (runtime_version % 100) / 10);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n", device_prop.major,
           device_prop.minor);
    printf(" Total amount of global memory: %.2f GBytes (%llu bytes)\n",
           static_cast<float>(device_prop.totalGlobalMem) / (pow(1024.0, 3)),
           static_cast<unsigned long long>(device_prop.totalGlobalMem));
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", device_prop.clockRate * 1e-3f,
           device_prop.clockRate * 1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n", device_prop.memoryClockRate * 1e-3f);
    printf(" Memory Bus Width: %d-bit\n", device_prop.memoryBusWidth);
    if (device_prop.l2CacheSize) {
        printf(" L2 Cache Size: %d bytes\n", device_prop.l2CacheSize);
    }
    printf(
        " Max Texture Dimension Size (x,y,z) "
        " 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
        device_prop.maxTexture1D, device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
        device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
    printf(" Max Layered Texture Size (dim) x layers\n  1D = (%d) x % d\n  2D = (%d, % d) x % d\n ",
           device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1],
           device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1],
           device_prop.maxTexture2DLayered[2]);
    printf(" Total amount of constant memory: %zu bytes\n", device_prop.totalConstMem);
    printf(" Total amount of shared memory per block: %zu bytes\n", device_prop.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n", device_prop.regsPerBlock);
    printf(" Warp size: %d\n", device_prop.warpSize);
    printf(" Multi Processor Count: %d\n", device_prop.multiProcessorCount);
    printf(" Maximum number of threads per multiprocessor: %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n", device_prop.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
           device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
           device_prop.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n", device_prop.maxGridSize[0],
           device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf(" Maximum memory pitch: %zu bytes\n", device_prop.memPitch);

    return MyErrCode::kOk;
}