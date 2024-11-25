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

    int curr_size = valid_group_size;
    for (int i = (valid_group_size + 1) / 2; i > 0;) {
        if (is_valid && lid < i && lid + i < curr_size) {
            partial_max[lid] = max(partial_max[lid], partial_max[lid + i]);
        }
        __syncthreads();
        curr_size = i;
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

static __global__ void sumf(float const* data, int len, float* output)
{
    extern __shared__ float partial_sum[];

    int lid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int valid_group_size = min(len - blockIdx.x * blockDim.x, blockDim.x);
    bool is_valid = global_id < len;

    if (is_valid) {
        partial_sum[lid] = data[global_id];
    }
    __syncthreads();

    int curr_size = valid_group_size;
    for (int i = (valid_group_size + 1) / 2; i > 0;) {
        if (is_valid && lid < i && lid + i < curr_size) {
            partial_sum[lid] += partial_sum[lid + i];
        }
        __syncthreads();
        curr_size = i;
        i = (i == 1) ? 0 : (i + 1) / 2;
    }

    if (lid == 0) {
        output[blockIdx.x] = partial_sum[0];
    }
}

float arraySum(float const* d_data, int len)
{
    int block_size = 1024;
    int grid_size = (len + block_size - 1) / block_size;
    float* d_partial_output;
    CHECK_CUDA(cudaMalloc(&d_partial_output, grid_size * sizeof(float)));
    sumf<<<grid_size, block_size, block_size * sizeof(float)>>>(d_data, len, d_partial_output);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> partial_output(grid_size);
    CHECK_CUDA(cudaMemcpy(partial_output.data(), d_partial_output, grid_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_partial_output));

    float sum_val = 0.0;
    for (auto v: partial_output) {
        sum_val += v;
    }

    return sum_val;
}

static __global__ void gaussianGen(float* ker, float sigma, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    int n1 = pow(static_cast<int>(ic) - nc / 2, 2);
    int n2 = pow(static_cast<int>(ir) - nr / 2, 2);
    ker[ir * nc + ic] = exp(-1 * (n1 + n2) / (2 * pow(sigma, 2)));
}

static __global__ void gaussianNorm(float* ker, float sum, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    ker[ir * nc + ic] /= sum;
}

MyErrCode getGaussianKernel(int rows, int cols, float sigma, float*& d_ker)
{
    CHECK_CUDA(cudaMalloc(&d_ker, sizeof(float) * rows * cols));
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    gaussianGen<<<grid, block>>>(d_ker, sigma, cols, rows);
    CHECK_CUDA(cudaGetLastError());
    float total_sum = arraySum(d_ker, rows * cols);
    gaussianNorm<<<grid, block>>>(d_ker, total_sum, cols, rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return MyErrCode::kOk;
}

static __global__ void padarray(float const* arr, int nc, int nr, float* padded_arr,
                                int padding_col, int padding_row)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    int pc = nc + 2 * padding_col;
    int pr = nr + 2 * padding_row;
    if (ic >= pc || ir >= pr) {
        return;
    }
    int remap_c = ic - padding_col;
    int remap_r = ir - padding_row;
    remap_c = max(0, min(nc - 1, remap_c));
    remap_r = max(0, min(nr - 1, remap_r));
    padded_arr[ir * pc + ic] = arr[remap_r * nc + remap_c];
}

MyErrCode padArrayRepBoth(float* d_arr, int nc, int nr, float*& d_padded_arr, int padding_col,
                          int padding_row)
{
    int pc = nc + 2 * padding_col;
    int pr = nr + 2 * padding_row;
    CHECK_CUDA(cudaMalloc(&d_padded_arr, sizeof(float) * pc * pr));
    dim3 block(32, 32);
    dim3 grid((pc + block.x - 1) / block.x, (pr + block.y - 1) / block.y);
    padarray<<<grid, block>>>(d_arr, nc, nr, d_padded_arr, padding_col, padding_row);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return MyErrCode::kOk;
}

}  // namespace common
