#include "./common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

using common::seconds;

void initialData(float* ip, int const size)
{
    int i;

    for (i = 0; i < size; i++) {
        ip[i] = static_cast<float>(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float* a, float* b, float* c, int const nx, int const ny)
{
    float* ia = a;
    float* ib = b;
    float* ic = c;

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

void checkResult(float* host_ref, float* gpu_ref, int const n)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < n; i++) {
        if (abs(host_ref[i] - gpu_ref[i]) > epsilon) {
            printf("host %f gpu %f ", host_ref[i], gpu_ref[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float const* a, float const* b, float* c, int nx, int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        c[idx] = a[idx] + b[idx];
    }
}

MyErrCode sumMatrix(int argc, char** argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp device_prop;
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, dev));
    CHECK_CUDA(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int n_bytes = nxy * sizeof(float);

    // malloc host memory
    float *h_a, *h_b, *host_ref, *gpu_ref;
    h_a = static_cast<float*>(malloc(n_bytes));
    h_b = static_cast<float*>(malloc(n_bytes));
    host_ref = static_cast<float*>(malloc(n_bytes));
    gpu_ref = static_cast<float*>(malloc(n_bytes));

    // initialize data at host side
    double i_start = seconds();
    initialData(h_a, nxy);
    initialData(h_b, nxy);
    double i_elaps = seconds() - i_start;

    memset(host_ref, 0, n_bytes);
    memset(gpu_ref, 0, n_bytes);

    // add matrix at host side for result checks
    i_start = seconds();
    sumMatrixOnHost(h_a, h_b, host_ref, nx, ny);
    i_elaps = seconds() - i_start;
    printf("sumMatrixOnHost elapsed %.3f s\n", i_elaps);

    // malloc device global memory
    float *d_mat_a, *d_mat_b, *d_mat_c;
    CHECK_CUDA(cudaMalloc((void**)&d_mat_a, n_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_b, n_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_mat_c, n_bytes));

    // transfer data from host to device
    CHECK_CUDA(cudaMemcpy(d_mat_a, h_a, n_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat_b, h_b, n_bytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;

    if (argc > 2) {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // execute the kernel
    CHECK_CUDA(cudaDeviceSynchronize());
    i_start = seconds();
    sumMatrixOnGPU2D<<<grid, block>>>(d_mat_a, d_mat_b, d_mat_c, nx, ny);
    CHECK_CUDA(cudaDeviceSynchronize());
    i_elaps = seconds() - i_start;
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %.3f s\n", grid.x, grid.y, block.x,
           block.y, i_elaps);
    CHECK_CUDA(cudaGetLastError());

    // copy kernel result back to host side
    CHECK_CUDA(cudaMemcpy(gpu_ref, d_mat_c, n_bytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(host_ref, gpu_ref, nxy);

    // free device global memory
    CHECK_CUDA(cudaFree(d_mat_a));
    CHECK_CUDA(cudaFree(d_mat_b));
    CHECK_CUDA(cudaFree(d_mat_c));

    // free host memory
    free(h_a);
    free(h_b);
    free(host_ref);
    free(gpu_ref);

    // reset device
    CHECK_CUDA(cudaDeviceReset());
    std::cout << "done!\n";

    return MyErrCode::kOk;
}