#include "./fwd.cuh"
#include "./common.cuh"
#include "toolkit/logging.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace cg = cooperative_groups;

__device__ float clamp(float x, float a, float b) { return max(a, min(b, x)); }

__global__ void conv_x(uint8_t* img_in, float* ker, float* img_raw, int w_img, int h_img, int n_ker)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    if (w >= w_img || h >= h_img) {
        return;
    }

    int mw = w - n_ker / 2;

    for (int i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (int iw = 0; iw < n_ker; ++iw) {
            int ww = mw + iw;
            float pi =
                (ww >= 0 && ww < w_img) ? (img_in[(w_img * h * 4) + ww * 4 + i] / 255.0f) : 0.0f;
            float pk = ker[iw];
            sum += pi * pk;
        }
        img_raw[(w_img * h * 3) + w * 3 + i] = sum;
    }
}

__global__ void conv_y(uint8_t* img_in, float* img_raw, float* ker, uint8_t* img_out, int w_img,
                       int h_img, int n_ker, float enhance_k, float complex_k, int output_mode)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    if (w >= w_img || h >= h_img) {
        return;
    }

    int mh = h - n_ker / 2;

    for (int i = 0; i < 3; ++i) {
        float sum = 0.0f;
        for (int ih = 0; ih < n_ker; ++ih) {
            int hh = mh + ih;
            float pi = (hh >= 0 && hh < h_img) ? (img_raw[(w_img * hh * 3) + w * 3 + i]) : 0.0f;
            float pk = ker[ih];
            sum += pi * pk;
        }
        int idx = (w_img * h * 4) + w * 4 + i;
        float raw = img_in[idx] / 255.0f;
        float out;
        if (output_mode == 0) {  // structure
            out = enhance_k * (raw - sum) + sum;
        } else if (output_mode == 1) {  // texture
            out = enhance_k * (raw - sum) + raw;
        } else {  // complex
            out = enhance_k * (raw - sum) + complex_k * sum + (1 - complex_k) * raw;
        }
        img_out[idx] = clamp(round(out * 255), 0.0f, 255.0f);
    }

    img_out[(w_img * h * 4) + w * 4 + 3] = 255;
}

__global__ void warm_up_gpu()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

std::vector<float> GetGaussianKernel(int n, double sigmax)
{
    auto gauss = cv::getGaussianKernel(n, sigmax, CV_32F);
    std::vector<float> buf(n);
    std::memcpy(buf.data(), gauss.ptr<float>(), n * sizeof(float));
    return std::move(buf);
}

int txi_gaussian(int argc, char** argv)
{
    // parameter
    float sigma = 100;
    int radius = 5;
    float enhance_k = 1.5;
    float complex_k = 1.0;
    int output_mode = 1;
    int win_size = radius * 2 + 1;
    auto fpath = toolkit::getTempDir() / "hdr.jpg";

    // read image
    cv::Mat img = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img.data == nullptr) {
        ELOG("failed to load image file: {}", fpath.string());
        return 1;
    }
    cv::Mat mat_img_in;
    cv::cvtColor(img, mat_img_in, cv::COLOR_BGR2BGRA);
    int image_width = mat_img_in.cols;
    int image_height = mat_img_in.rows;
    int image_size = image_width * image_height;
    int image_byte_len = image_size * 4 * sizeof(uint8_t);

    // buffer alloc
    uint8_t* d_img_in;
    float* d_ker;
    float* d_img_raw;
    uint8_t* d_img_out;

    CHECK(cudaMalloc(&d_img_in, image_byte_len));
    CHECK(cudaMalloc(&d_ker, win_size * sizeof(float)));
    CHECK(cudaMalloc(&d_img_raw, image_size * 3 * sizeof(float)));
    CHECK(cudaMalloc(&d_img_out, image_byte_len));

    // warm up
    warm_up_gpu<<<10 * 1024 * 1024, 1024>>>();
    CHECK(cudaDeviceSynchronize());

    // copy data in
    TIMER_BEGIN(total)

    TIMER_BEGIN(copy_data_in)
    CHECK(cudaMemcpy(d_img_in, mat_img_in.data, image_byte_len, cudaMemcpyHostToDevice));
    TIMER_END(copy_data_in, fmt::format("host to device {} bytes", image_byte_len))

    std::vector<float> gaussian_kernel = GetGaussianKernel(win_size, sigma);
    CHECK(cudaMemcpy(d_ker, gaussian_kernel.data(), win_size * sizeof(float),
                     cudaMemcpyHostToDevice));

    // process
    CHECK(cudaDeviceSynchronize());

    TIMER_BEGIN(kernel_run)
    dim3 block(32, 32);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    conv_x<<<grid, block>>>(d_img_in, d_ker, d_img_raw, image_width, image_height, win_size);
    CHECK(cudaGetLastError());

    conv_y<<<grid, block>>>(d_img_in, d_img_raw, d_ker, d_img_out, image_width, image_height,
                            win_size, enhance_k, complex_k, output_mode);
    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());
    TIMER_END(kernel_run, fmt::format("kernel run on {}x{}", image_width, image_height))

    // copy data out
    cv::Mat mat_img_out(image_height, image_width, CV_8UC4);
    TIMER_BEGIN(copy_data_out)
    CHECK(cudaMemcpy(mat_img_out.data, d_img_out, image_byte_len, cudaMemcpyDeviceToHost));
    TIMER_END(copy_data_out, fmt::format("device to host {} bytes", image_byte_len))

    TIMER_END(total, "total")

    auto outfile = FSTR("{}-txi_gaussian{}", fpath.stem().string(), fpath.extension().string());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), mat_img_out);

    // free
    CHECK(cudaFree(d_img_in));
    CHECK(cudaFree(d_ker));
    CHECK(cudaFree(d_img_raw));
    CHECK(cudaFree(d_img_out));
    ILOG("done!");

    return 0;
}