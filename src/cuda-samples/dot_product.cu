#include "./fwd.cuh"
#include "./common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

int const N = 33 * 1024;
int const threadsPerBlock = 256;
int const blocksPerGrid = std::min(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float* a, float* b, float* c)
{
    __shared__ float cache[threadsPerBlock];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0;
    while (tid < N) {
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

#define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)

int dot_product(int argc, char** argv)
{
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    // allocate memory on the CPU side
    a = (float*)malloc(N * sizeof(float));
    b = (float*)malloc(N * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the GPU
    CHECK(cudaMalloc(&dev_a, N * sizeof(float)));
    CHECK(cudaMalloc(&dev_b, N * sizeof(float)));
    CHECK(cudaMalloc(&dev_partial_c, blocksPerGrid * sizeof(float)));

    // fill in the host memory with data
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // copy the arrays ‘a’ and ‘b’ to the GPU
    CHECK(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);
    CHECK(cudaPeekAtLastError());

    // copy the array 'c' back from the GPU to the CPU
    CHECK(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
                     cudaMemcpyDeviceToHost));

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

    // free memory on the GPU side
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_partial_c));

    // free memory on the CPU side
    free(a);
    free(b);
    free(partial_c);

    return 0;
}
