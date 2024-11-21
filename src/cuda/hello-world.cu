#include "./common.cuh"
#include <cuda_runtime.h>

__global__ void testPrint()
{
    printf("Hello World from GPU1!\n");
    printf("Hello World from GPU2!\n");
    printf("Hello World from GPU3!\n");
}

int helloWorld(int argc, char** argv)
{
    testPrint<<<1, 1>>>();
    CHECK(cudaPeekAtLastError());
    CHECK(cudaDeviceSynchronize());
    return 0;
}
