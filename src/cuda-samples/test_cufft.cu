#include "./fwd.cuh"
#include "./common.cuh"
#include <iostream>
#include <cuda.h>
#include <cufft.h>

#define MY_PI 3.14159265358979323846

void print_2d(char const* header, cufftComplex* cs, int nx, int ny)
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

int test_cufft(int argc, char** argv)
{
    int nx = 1920;
    int ny = 1080;
    int ns = nx * ny;
    cufftComplex *dComplex, *complex;

    // Allocate device memory
    complex = static_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * ns));
    CHECK(cudaMalloc(&dComplex, sizeof(cufftComplex) * ns));

    // Input Generation
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y) {
            cufftComplex c;
            c.x = x + y;
            c.y = 0;
            complex[y + x * ny] = c;
        }
    }
    // print_2d("=== INPUT ===", complex, nx, ny);

    // Setup the cuFFT plan
    cufftHandle plan = 0;
    CHECK_CUFFT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));

    // Transfer inputs into device memory
    CHECK(cudaMemcpy(dComplex, complex, sizeof(cufftComplex) * ns, cudaMemcpyHostToDevice));

    // Execute a complex-to-complex 1D FFT
    TIMER_BEGIN(fft2)
    CHECK_CUFFT(cufftExecC2C(plan, dComplex, dComplex, CUFFT_FORWARD));
    CHECK(cudaDeviceSynchronize());
    TIMER_END(fft2, fmt::format("fft2 {}x{}", nx, ny))

    // Retrieve the results into host memory
    CHECK(cudaMemcpy(complex, dComplex, sizeof(cufftComplex) * ns, cudaMemcpyDeviceToHost));

    // print_2d("=== OUTPUT ===", complex, nx, ny);

    free(complex);
    CHECK(cudaFree(dComplex));
    CHECK_CUFFT(cufftDestroy(plan));

    return 0;
}