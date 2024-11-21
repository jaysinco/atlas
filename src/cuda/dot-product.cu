#include "./common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

int const kN = 33 * 1024;
int const kThreadsPerBlock = 256;
int const kBlocksPerGrid = std::min(32, (kN + kThreadsPerBlock - 1) / kThreadsPerBlock);

__global__ void dot(float const* a, float const* b, float* c)
{
    __shared__ float cache[kThreadsPerBlock];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0;
    while (tid < kN) {
        temp += a[tid] * b[tid];
        tid += gridDim.x * blockDim.x;
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadIdx.x == 0) {
        c[blockIdx.x] = cache[0];
    }
}

#define SUM_SQUARES(x) (x * (x + 1) * (2 * x + 1) / 6)

MyErrCode dotProduct(int argc, char** argv)
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = static_cast<float*>(malloc(kN * sizeof(float)));
    b = static_cast<float*>(malloc(kN * sizeof(float)));
    partial_c = static_cast<float*>(malloc(kBlocksPerGrid * sizeof(float)));

    // allocate the memory on the GPU
    CHECK_CUDA(cudaMalloc(&dev_a, kN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_b, kN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dev_partial_c, kBlocksPerGrid * sizeof(float)));

    // fill in the host memory with data
    for (int i = 0; i < kN; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays ‘a’ and ‘b’ to the GPU
    CHECK_CUDA(cudaMemcpy(dev_a, a, kN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_b, b, kN * sizeof(float), cudaMemcpyHostToDevice));
    dot<<<kBlocksPerGrid, kThreadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    CHECK_CUDA(cudaPeekAtLastError());

    // copy the array 'c' back from the GPU to the CPU
    CHECK_CUDA(cudaMemcpy(partial_c, dev_partial_c, kBlocksPerGrid * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < kBlocksPerGrid; i++) {
        c += partial_c[i];
    }

    printf("Does GPU value %.6g = %.6g?\n", c, 2 * SUM_SQUARES((float)(kN - 1)));

    // free memory on the GPU side
    CHECK_CUDA(cudaFree(dev_a));
    CHECK_CUDA(cudaFree(dev_b));
    CHECK_CUDA(cudaFree(dev_partial_c));

    // free memory on the CPU side
    free(a);
    free(b);
    free(partial_c);

    return MyErrCode::kOk;
}
