#include "./common.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * A simple example of nested kernel launches from the GPU. Each thread displays
 * its information when execution begins, and also diagnostics when the next
 * lowest nesting layer completes.
 */

__global__ void nested(int const i_size, int i_depth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", i_depth, tid, blockIdx.x);

    // condition to stop recursive execution
    if (i_size == 1) {
        return;
    }

    // reduce block size to half
    int nthreads = i_size >> 1;

    // thread 0 launches child grid recursively
    if (tid == 0 && nthreads > 0) {
        nested<<<1, nthreads>>>(nthreads, ++i_depth);
        printf("-------> nested execution depth: %d\n", i_depth);
    }
}

MyErrCode nestedHelloWorld(int argc, char** argv)
{
    int size = 8;
    int blocksize = 8;  // initial block size
    int igrid = 1;

    if (argc > 1) {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x, block.x);

    nested<<<grid, block>>>(block.x, 0);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceReset());
    return MyErrCode::kOk;
}