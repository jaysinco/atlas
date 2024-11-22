#pragma once
#include <chrono>
#include "toolkit/logging.h"
#include <cuComplex.h>
#include <iostream>
#include "./fwd.h"

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            ELOG("cuda error: {} {}", int(error), cudaGetErrorString(error)); \
            exit(1);                                                          \
        }                                                                     \
    }

#define CHECK_CUFFT(call)                      \
    {                                          \
        cufftResult err = call;                \
        if (err != CUFFT_SUCCESS) {            \
            ELOG("cufft error: {}", int(err)); \
            exit(1);                           \
        }                                      \
    }

#define TIMER_BEGIN(x) auto timer_##x = std::chrono::system_clock::now();
#define TIMER_END(x, desc)                                                                        \
    ILOG("{}, elapsed={:.1f}ms", (desc),                                                          \
         std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - \
                                                               timer_##x)                         \
                 .count() /                                                                       \
             1000.);

template <typename T>
static void print2D(T* data, bool is_host_ptr, int data_width, int data_height, int xpos, int ypos,
                    int roi_width, int roi_height)
{
    T* host_data = new T[roi_width * roi_height];
    T* dev_data = data + ypos * data_width + xpos;
    CHECK_CUDA(cudaMemcpy2D(host_data, sizeof(T) * roi_width, dev_data, sizeof(T) * data_width,
                            sizeof(T) * roi_width, roi_height,
                            is_host_ptr ? cudaMemcpyHostToHost : cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    for (int y = 0; y < roi_height; ++y) {
        for (int x = 0; x < roi_width; ++x) {
            T& d = host_data[y * roi_width + x];
            std::cout << std::fixed << std::showpoint << std::setprecision(5) << d << "\t";
        }
        std::cout << std::endl;
    }
    delete[] host_data;
}

namespace common
{

__device__ float clamp(float x, float a, float b);
__device__ cuComplex cexpf(cuComplex z);

};  // namespace common

std::ostream& operator<<(std::ostream& os, cuComplex const& cmp);
double seconds();
void warmUpGpu();
