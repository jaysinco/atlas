#pragma once
#include <chrono>
#include "toolkit/logging.h"
#include "./fwd.h"

#define CHECK(call)                                                           \
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

__device__ float clamp(float x, float a, float b);

double seconds();
void warmUpGpu();
