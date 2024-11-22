#include "./common.cuh"
#include <cuda.h>
#include <cufft.h>

#define MY_PI 3.14159265358979323846

MyErrCode cufftTest(int argc, char** argv)
{
    int nx = 3;  // row
    int ny = 2;  // col
    int ns = nx * ny;
    cufftComplex *d_complex, *complex;

    // Allocate device memory
    complex = static_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * ns));
    CHECK_CUDA(cudaMalloc(&d_complex, sizeof(cufftComplex) * ns));

    // Input Generation
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            cufftComplex c;
            c.x = x + y;
            c.y = 0;
            complex[y + x * ny] = c;
        }
    }
    std::cout << "=== INPUT ===" << std::endl;
    print2D(complex, true, ny, nx, 0, 0, ny, nx);

    // Setup the cuFFT plan
    cufftHandle plan = 0;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

    // Transfer inputs into device memory
    CHECK_CUDA(cudaMemcpy(d_complex, complex, sizeof(cufftComplex) * ns, cudaMemcpyHostToDevice));

    // warm up
    common::warmUpGpu();

    // Execute a complex-to-complex 1D FFT
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, nullptr));

    CHECK_CUFFT(cufftExecC2C(plan, d_complex, d_complex, CUFFT_FORWARD));

    CHECK_CUDA(cudaEventRecord(stop, nullptr));
    CHECK_CUDA(cudaEventSynchronize(stop););

    float time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    ILOG("fft2 {}x{}: {:.3f}ms", nx, ny, time_ms);

    // Retrieve the results into host memory
    CHECK_CUDA(cudaMemcpy(complex, d_complex, sizeof(cufftComplex) * ns, cudaMemcpyDeviceToHost));

    std::cout << "=== OUTPUT ===" << std::endl;
    print2D(d_complex, false, ny, nx, 0, 0, ny, nx);

    free(complex);
    CHECK_CUDA(cudaFree(d_complex));
    CHECK_CUFFT(cufftDestroy(plan));

    return MyErrCode::kOk;
}