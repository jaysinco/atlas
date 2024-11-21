#include "./common.cuh"
#include <cuda_runtime.h>

__global__ void testPrint()
{
    printf("Hello World from GPU1!\n");
    printf("Hello World from GPU2!\n");
    printf("Hello World from GPU3!\n");
}

MyErrCode helloWorld(int argc, char** argv)
{
    testPrint<<<1, 1>>>();
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    return MyErrCode::kOk;
}
