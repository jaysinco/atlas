#pragma once
#include <chrono>
#include <stdio.h>

#define CHECK(call)                                                                      \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                                     \
        }                                                                                \
    }

#define CHECK_CUFFT(call)                                                              \
    {                                                                                  \
        cufftResult err;                                                               \
        if ((err = (call)) != CUFFT_SUCCESS) {                                         \
            fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, __LINE__); \
            exit(1);                                                                   \
        }                                                                              \
    }

#define TIMER_BEGIN(x) auto timer_##x = std::chrono::system_clock::now();
#define TIMER_END(x, desc)                                                                        \
    ILOG("{}, elapsed={:.1f}ms", (desc),                                                          \
         std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - \
                                                               timer_##x)                         \
                 .count() /                                                                       \
             1000.);

inline double seconds()
{
    auto now = std::chrono::system_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return micro.count() * 1e-6;
}
