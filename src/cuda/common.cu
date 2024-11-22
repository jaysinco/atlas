#include "./common.cuh"

std::ostream& operator<<(std::ostream& os, cuComplex const& cmp)
{
    return os << cmp.x << (cmp.y < 0 ? "" : "+") << cmp.y << "j";
}

namespace common
{

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__device__ cuComplex cexpf(cuComplex z)
{
    cuComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

double seconds()
{
    auto now = std::chrono::system_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return micro.count() * 1e-6;
}

static __global__ void addWarm()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warmUpGpu()
{
    addWarm<<<10 * 1024 * 1024, 1024>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
}

static __global__ void maxf(float const* data, int len, float* output)
{
    extern __shared__ float partial_max[];

    int lid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_group_size = min(len - blockIdx.x * blockDim.x, blockDim.x);
    bool is_valid = global_id < len;

    if (is_valid) {
        partial_max[lid] = data[global_id];
    }
    __syncthreads();

    for (int i = (valid_group_size + 1) / 2; i > 0;) {
        if (is_valid && lid < i && lid + i < valid_group_size) {
            partial_max[lid] = max(partial_max[lid], partial_max[lid + i]);
        }
        __syncthreads();
        i = (i == 1) ? 0 : (i + 1) / 2;
    }

    if (lid == 0) {
        output[blockIdx.x] = partial_max[0];
    }
}

float arrayMax(float const* d_data, int len)
{
    int block_size = 1024;
    int grid_size = (len + block_size - 1) / block_size;
    float* d_partial_output;
    CHECK_CUDA(cudaMalloc(&d_partial_output, grid_size * sizeof(float)));
    maxf<<<grid_size, block_size, block_size * sizeof(float)>>>(d_data, len, d_partial_output);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> partial_output(grid_size);
    CHECK_CUDA(cudaMemcpy(partial_output.data(), d_partial_output, grid_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_partial_output));

    float max_val = std::numeric_limits<float>::min();
    for (auto v: partial_output) {
        max_val = std::max(max_val, v);
    }

    return max_val;
}

}  // namespace common
