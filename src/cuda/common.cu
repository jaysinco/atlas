#include "./common.cuh"

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

double seconds()
{
    auto now = std::chrono::system_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return micro.count() * 1e-6;
}

__global__ void addWarm()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warmUpGpu()
{
    addWarm<<<10 * 1024 * 1024, 1024>>>();
    CHECK(cudaDeviceSynchronize());
}
