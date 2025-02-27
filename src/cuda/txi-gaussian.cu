#include "./common.cuh"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#define USE_MANAGERD_MEMORY 0

__global__ void convX(uint8_t const* img_in, float const* ker, float* img_raw, int w_img, int h_img,
                      int n_ker)
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

__global__ void convY(uint8_t const* img_in, float const* img_raw, float const* ker,
                      uint8_t* img_out, int w_img, int h_img, int n_ker, float enhance_k,
                      float complex_k, int output_mode)
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
        img_out[idx] = common::clamp(round(out * 255), 0.0f, 255.0f);
    }

    img_out[(w_img * h * 4) + w * 4 + 3] = 255;
}

void getGaussianKernel(int n, double sigmax, float* data)
{
    auto gauss = cv::getGaussianKernel(n, sigmax, CV_32F);
    std::memcpy(data, gauss.ptr<float>(), n * sizeof(float));
}

MyErrCode txiGaussian(int argc, char** argv)
{
    // parameter
    float sigma = 100;
    int radius = 8;
    float enhance_k = 1.5;
    float complex_k = 1.0;
    int output_mode = 1;
    int win_size = radius * 2 + 1;
    auto fpath = toolkit::getDataDir() / "hdr.jpg";
    int target_image_width = 1920;   // 1920 2560 3840
    int target_image_height = 1080;  // 1080 1440 2160

    uint8_t* d_img_in;
    float* d_ker;
    float* d_img_raw;
    uint8_t* d_img_out;

    // read image
    cv::Mat img_file = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath.string());
        return MyErrCode::kFailed;
    }
    cv::Mat img;
    cv::resize(img_file, img, cv::Size(target_image_width, target_image_height));

    int image_width = img.cols;
    int image_height = img.rows;
    int image_size = image_width * image_height;
    int image_byte_len = image_size * 4 * sizeof(uint8_t);

    // warm up
    common::warmUpGpu();

    // buffer alloc
#if USE_MANAGERD_MEMORY
    CHECK_CUDA(cudaMalloc(&d_img_raw, image_size * 3 * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&d_img_in, image_byte_len, cudaMemAttachHost));
    CHECK_CUDA(cudaMallocManaged(&d_ker, win_size * sizeof(float), cudaMemAttachHost));
    CHECK_CUDA(cudaMallocManaged(&d_img_out, image_byte_len, cudaMemAttachGlobal));
#else
    CHECK_CUDA(cudaMalloc(&d_img_raw, image_size * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_img_in, image_byte_len));
    CHECK_CUDA(cudaMalloc(&d_ker, win_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_img_out, image_byte_len));
#endif

    MY_TIMER_BEGIN(INFO, "total")

// image convert
#if USE_MANAGERD_MEMORY
    cv::Mat mat_img_in(image_height, image_width, CV_8UC4, d_img_in);
#else
    cv::Mat mat_img_in;
#endif

    cv::cvtColor(img, mat_img_in, cv::COLOR_BGR2BGRA);

    // copy data in
#if !USE_MANAGERD_MEMORY
    MY_TIMER_BEGIN(INFO, FSTR("host to device {} bytes", image_byte_len))
    CHECK_CUDA(cudaMemcpy(d_img_in, mat_img_in.data, image_byte_len, cudaMemcpyHostToDevice));
    MY_TIMER_END
#endif

#if USE_MANAGERD_MEMORY
    GetGaussianKernel(win_size, sigma, d_ker);
#else
    std::vector<float> gaussian_kernel(win_size);
    getGaussianKernel(win_size, sigma, gaussian_kernel.data());
    CHECK_CUDA(cudaMemcpy(d_ker, gaussian_kernel.data(), win_size * sizeof(float),
                          cudaMemcpyHostToDevice));
#endif

    // process
    CHECK_CUDA(cudaDeviceSynchronize());

#if USE_MANAGERD_MEMORY
    // CHECK_CUDA(cudaStreamAttachMemAsync(nullptr, d_img_in, 0, cudaMemAttachGlobal));
    // CHECK_CUDA(cudaStreamAttachMemAsync(nullptr, d_ker, 0, cudaMemAttachGlobal));
#endif

    MY_TIMER_BEGIN(INFO, FSTR("kernel run on {}x{}", image_width, image_height))
    dim3 block(32, 32);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    convX<<<grid, block>>>(d_img_in, d_ker, d_img_raw, image_width, image_height, win_size);
    CHECK_CUDA(cudaGetLastError());

    convY<<<grid, block>>>(d_img_in, d_img_raw, d_ker, d_img_out, image_width, image_height,
                           win_size, enhance_k, complex_k, output_mode);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
    MY_TIMER_END

    // copy data out
#if !USE_MANAGERD_MEMORY
    std::vector<uint8_t> vec_img_out(image_byte_len);
    MY_TIMER_BEGIN(INFO, FSTR("device to host {} bytes", image_byte_len))
    CHECK_CUDA(cudaMemcpy(vec_img_out.data(), d_img_out, image_byte_len, cudaMemcpyDeviceToHost));
    MY_TIMER_END
#endif

    // write image
#if USE_MANAGERD_MEMORY
    // CHECK_CUDA(cudaStreamAttachMemAsync(nullptr, d_img_out, 0, cudaMemAttachHost));
    // CHECK_CUDA(cudaDeviceSynchronize());
    cv::Mat mat_img_out(image_height, image_width, CV_8UC4, d_img_out);
#else
    cv::Mat mat_img_out(image_height, image_width, CV_8UC4, vec_img_out.data());
#endif
    auto outfile =
        FSTR("{}-txi_gaussian-cuda{}", fpath.stem().string(), fpath.extension().string());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), mat_img_out);

    MY_TIMER_END

    // free
    CHECK_CUDA(cudaFree(d_img_in));
    CHECK_CUDA(cudaFree(d_ker));
    CHECK_CUDA(cudaFree(d_img_raw));
    CHECK_CUDA(cudaFree(d_img_out));
    ILOG("done!");

    return MyErrCode::kOk;
}