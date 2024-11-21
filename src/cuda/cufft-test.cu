#include "./common.cuh"
#include <iostream>
#include <cuda.h>
#include <cufft.h>

#define MY_PI 3.14159265358979323846

static void print2d(char const* header, cufftComplex* cs, int nx, int ny)
{
    std::cout << header << std::endl;
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            cufftComplex& it = cs[y + x * ny];
            std::cout << it.x << "+" << it.y << "i"
                      << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int cufftTest(int argc, char** argv)
{
    int nx = 1920;
    int ny = 1080;
    int ns = nx * ny;
    cufftComplex *d_complex, *complex;

    // Allocate device memory
    complex = static_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * ns));
    CHECK(cudaMalloc(&d_complex, sizeof(cufftComplex) * ns));

    // Input Generation
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            cufftComplex c;
            c.x = x + y;
            c.y = 0;
            complex[y + x * ny] = c;
        }
    }
    print2d("=== INPUT ===", complex, nx, ny);

    // Setup the cuFFT plan
    cufftHandle plan = 0;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

    // Transfer inputs into device memory
    CHECK(cudaMemcpy(d_complex, complex, sizeof(cufftComplex) * ns, cudaMemcpyHostToDevice));

    // warm up
    warmUpGpu();

    // Execute a complex-to-complex 1D FFT
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start, nullptr));

    CHECK_CUFFT(cufftExecC2C(plan, d_complex, d_complex, CUFFT_FORWARD));

    CHECK(cudaEventRecord(stop, nullptr));
    CHECK(cudaEventSynchronize(stop););

    float time_ms;
    CHECK(cudaEventElapsedTime(&time_ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    ILOG("fft2 {}x{}: {:.3f}ms", nx, ny, time_ms);

    // Retrieve the results into host memory
    CHECK(cudaMemcpy(complex, d_complex, sizeof(cufftComplex) * ns, cudaMemcpyDeviceToHost));

    // print_2d("=== OUTPUT ===", complex, nx, ny);

    free(complex);
    CHECK(cudaFree(d_complex));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}