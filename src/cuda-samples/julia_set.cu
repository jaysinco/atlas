#include "./fwd.cuh"
#include "./common.cuh"
#include <cuda_runtime.h>
#include "stb_image_write.h"

struct myComplex
{
    float r;
    float i;

    __device__ myComplex(float x, float y): r(x), i(y) {}

    __device__ float magnitude2() { return r * r + i * i; }

    __device__ myComplex operator*(myComplex const& a)
    {
        return myComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ myComplex operator+(myComplex const& a) { return myComplex(r + a.r, i + a.i); }
};

__device__ float julia(int w, int h, int image_width, int image_height)
{
    float const scale = 1.8;
    float const max_val = 1000.0;
    float x = scale * float(image_width / 2.0 - w) / (image_width / 2.0);
    float y = scale * ((float(image_height) / image_width) * float(image_height / 2.0 - h) /
                       (image_height / 2.0));

    myComplex c(-0.4, -0.59);
    myComplex a(x, y);

    for (int i = 0; i < 200; ++i) {
        a = a * a + c;
        if (a.magnitude2() > max_val) {
            return 0;
        }
    }
    return (max_val - a.magnitude2()) / max_val;
}

__global__ void calc_julia(int image_width, int image_height, uint8_t* pixels)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    if (w >= image_width || h >= image_height) {
        return;
    }
    int offset = image_width * h * 3 + w * 3;
    float val = julia(w, h, image_width, image_height);
    pixels[offset + 0] = 255 * val;
    pixels[offset + 1] = 255 * val;
    pixels[offset + 2] = 255 * val;
}

void fill_julia_set(int image_width, int image_height, uint8_t* pixels)
{
    int channel_num = 3;
    int pixels_size = image_width * image_height * channel_num;

    uint8_t* d_pixels;
    CHECK(cudaMalloc(&d_pixels, pixels_size));
    dim3 block(32, 32);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    calc_julia<<<grid, block>>>(image_width, image_height, d_pixels);
    CHECK(cudaPeekAtLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("ElapsedTime: %.3f ms\n", elapsed_ms);
    CHECK(cudaEventDestroy(start))
    CHECK(cudaEventDestroy(stop))

    CHECK(cudaMemcpy(pixels, d_pixels, pixels_size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_pixels))
}

int julia_set(int argc, char** argv)
{
    int image_width = 2560;
    int image_height = 1440;
    int channel_num = 3;
    int pixels_size = image_width * image_height * channel_num;

    uint8_t* pixels = new uint8_t[pixels_size];
    fill_julia_set(image_width, image_height, pixels);
    stbi_write_jpg("julia_set.jpg", image_width, image_height, channel_num, pixels, 100);
    delete[] pixels;

    return 0;
}