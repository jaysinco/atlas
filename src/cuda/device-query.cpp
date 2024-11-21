#include <cuda_runtime.h>
#include <string>

int* p_argc = nullptr;
char** p_argv = nullptr;

static char const* cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }

template <typename T>
static void check(T result, char const* const func, char const* const file, int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERRORS(val) check((val), #val, __FILE__, __LINE__)

static int convertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int sm;  // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int cores;
    } SSMtoCores;

    SSMtoCores n_gpu_arch_cores_per_sm[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
        {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
        {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;

    while (n_gpu_arch_cores_per_sm[index].sm != -1) {
        if (n_gpu_arch_cores_per_sm[index].sm == ((major << 4) + minor)) {
            return n_gpu_arch_cores_per_sm[index].cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one
    // to run properly
    printf(
        "MapSMtoCores for SM %d.%d is undefined."
        "  Default to use %d Cores/SM\n",
        major, minor, n_gpu_arch_cores_per_sm[index - 1].cores);
    return n_gpu_arch_cores_per_sm[index - 1].cores;
}

int main(int argc, char** argv)
{
    p_argc = &argc;
    p_argv = argv;

    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int device_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&device_count);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id),
               cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (device_count == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", device_count);
    }

    int dev, driver_version = 0, runtime_version = 0;

    for (dev = 0; dev < device_count; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);

        printf("\nDevice %d: \"%s\"\n", dev, device_prop.name);

        // Console log
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driver_version / 1000, (driver_version % 100) / 10, runtime_version / 1000,
               (runtime_version % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", device_prop.major,
               device_prop.minor);

        char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        sprintf_s(msg, sizeof(msg),
                  "  Total amount of global memory:                 %.0f MBytes "
                  "(%llu bytes)\n",
                  static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                  (unsigned long long)deviceProp.totalGlobalMem);
#else
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes "
                 "(%llu bytes)\n",
                 static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
                 static_cast<unsigned long long>(device_prop.totalGlobalMem));
#endif
        printf("%s", msg);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               device_prop.multiProcessorCount,
               convertSMVer2Cores(device_prop.major, device_prop.minor),
               convertSMVer2Cores(device_prop.major, device_prop.minor) *
                   device_prop.multiProcessorCount);
        printf(
            "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
            "GHz)\n",
            device_prop.clockRate * 1e-3f, device_prop.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               device_prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               device_prop.memoryBusWidth);

        if (device_prop.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   device_prop.l2CacheSize);
        }

#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }

#endif

        printf(
            "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
            "%d), 3D=(%d, %d, %d)\n",
            device_prop.maxTexture1D, device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
            device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               device_prop.maxTexture1DLayered[0], device_prop.maxTexture1DLayered[1]);
        printf(
            "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
            "layers\n",
            device_prop.maxTexture2DLayered[0], device_prop.maxTexture2DLayered[1],
            device_prop.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %zu bytes\n",
               device_prop.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               device_prop.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
               device_prop.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n", device_prop.regsPerBlock);
        printf("  Warp size:                                     %d\n", device_prop.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               device_prop.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               device_prop.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
               device_prop.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n",
               device_prop.memPitch);
        printf("  Texture alignment:                             %zu bytes\n",
               device_prop.textureAlignment);
        printf(
            "  Concurrent copy and kernel execution:          %s with %d copy "
            "engine(s)\n",
            (device_prop.deviceOverlap ? "Yes" : "No"), device_prop.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               device_prop.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               device_prop.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               device_prop.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               device_prop.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               device_prop.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n",
               device_prop.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               device_prop.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               device_prop.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               device_prop.pciDomainID, device_prop.pciBusID, device_prop.pciDeviceID);

        char const* s_compute_mode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this "
            "device)",
            "Exclusive Process (many threads in one process is able to use "
            "::cudaSetDevice() with this device)",
            "Unknown",
            nullptr};
        printf("  Compute Mode:\n");
        printf("     < %s >\n", s_compute_mode[device_prop.computeMode]);
    }

    // If there are 2 or more GPUs, query to determine whether RDMA is supported
    if (device_count >= 2) {
        cudaDeviceProp prop[64];
        int gpuid[64];  // we want to find the first two GPUs that can support P2P
        int gpu_p2p_count = 0;

        for (int i = 0; i < device_count; i++) {
            CHECK_CUDA_ERRORS(cudaGetDeviceProperties(&prop[i], i));

            // Only boards based on Fermi or later can support P2P
            if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
                // on Windows (64-bit), the Tesla Compute Cluster driver for windows
                // must be enabled to support this
                && prop[i].tccDriver
#endif
            ) {
                // This is an array of P2P capable GPUs
                gpuid[gpu_p2p_count++] = i;
            }
        }

        // Show all the combinations of support P2P GPUs
        int can_access_peer;

        if (gpu_p2p_count >= 2) {
            for (int i = 0; i < gpu_p2p_count; i++) {
                for (int j = 0; j < gpu_p2p_count; j++) {
                    if (gpuid[i] == gpuid[j]) {
                        continue;
                    }
                    CHECK_CUDA_ERRORS(
                        cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
                    printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                           prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                           can_access_peer ? "Yes" : "No");
                }
            }
        }
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string s_profile_string = "deviceQuery, CUDA Driver = CUDART";
    char c_temp[16];

    // driver version
    s_profile_string += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
    snprintf(c_temp, sizeof(c_temp), "%d.%d", driver_version / 1000, (driver_version % 100) / 10);
#endif
    s_profile_string += c_temp;

    // Runtime version
    s_profile_string += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
    snprintf(c_temp, sizeof(c_temp), "%d.%d", runtime_version / 1000, (runtime_version % 100) / 10);
#endif
    s_profile_string += c_temp;

    // Device count
    s_profile_string += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    snprintf(c_temp, sizeof(c_temp), "%d", device_count);
#endif
    s_profile_string += c_temp;
    s_profile_string += "\n";
    printf("%s", s_profile_string.c_str());

    printf("Result = PASS\n");

    // finish
    exit(EXIT_SUCCESS);
}
