#include "./common.cuh"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <cuda_runtime.h>
#include <vector>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>

__global__ void separate(uint8_t* fin, float* rchan, float* gchan, float* bchan, int width,
                         int height)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;

    if (w >= width || h >= height) {
        return;
    }

    uint8_t* pin = fin + (width * h * 4) + w * 4;
    rchan[width * h + w] = pin[2] / 255.0f;
    gchan[width * h + w] = pin[1] / 255.0f;
    bchan[width * h + w] = pin[0] / 255.0f;
}

__global__ void calcAb(float const* fin, float* aout, float* bout, int width, int height, int r,
                       float eps)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;

    if (w >= width || h >= height) {
        return;
    }

    float sum = 0.f;
    float sum_square = 0.f;
    int n = 0;
    for (int iw = w - r; iw <= w + r; ++iw) {
        for (int ih = h - r; ih <= h + r; ++ih) {
            if (iw >= 0 && ih >= 0 && iw < width && ih < height) {
                float v = fin[width * ih + iw];
                sum += v;
                sum_square += v * v;
                n += 1;
            }
        }
    }

    float fi = sum / n;
    float fii = sum_square / n;
    float cov = fii - fi * fi;
    int idx = width * h + w;
    float a = cov / (cov + eps);
    aout[idx] = a;
    bout[idx] = (1 - a) * fi;
}

__global__ void linearConv(float const* fin, float const* a, float const* b, int r, uint8_t* fout,
                           int width, int height, int color_idx, float enhance_k, float complex_k,
                           int output_mode)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int h = blockDim.y * blockIdx.y + threadIdx.y;

    if (w >= width || h >= height) {
        return;
    }

    float sum_a = 0.f;
    float sum_b = 0.f;
    int n = 0;
    for (int iw = w - r; iw <= w + r; ++iw) {
        for (int ih = h - r; ih <= h + r; ++ih) {
            if (iw >= 0 && ih >= 0 && iw < width && ih < height) {
                sum_a += a[width * ih + iw];
                sum_b += b[width * ih + iw];
                n += 1;
            }
        }
    }

    float mean_a = sum_a / n;
    float mean_b = sum_b / n;
    int idx = width * h + w;
    float q = mean_a * fin[idx] + mean_b;
    float qo;
    if (output_mode == 0) {  // structure
        qo = enhance_k * (fin[idx] - q) + q;
    } else if (output_mode == 1) {  // texture
        qo = enhance_k * (fin[idx] - q) + fin[idx];
    } else {  // complex
        qo = enhance_k * (fin[idx] - q) + complex_k * q + (1 - complex_k) * fin[idx];
    }

    uchar* pout = fout + (width * h * 4) + w * 4;
    *(pout + color_idx) = common::clamp(round(qo * 255), 0.0, 255.0);
    *(pout + 3) = 255;
}

MyErrCode txiGuided(int argc, char** argv)
{
    // parameter
    float eps = 1000;
    int radius = 8;
    float enhance_k = 1.5;
    float complex_k = 1.0;
    int output_mode = 1;
    auto fpath = toolkit::getDataDir() / "hdr.jpg";
    int target_image_width = 1920;   // 1920 2560 3840
    int target_image_height = 1080;  // 1080 1440 2160

    uint8_t* d_img_in;
    float* d_r;
    float* d_g;
    float* d_b;
    float* d_pa;
    float* d_pb;
    uint8_t* d_img_out;

    // read image
    cv::Mat img_file = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath);
        return MyErrCode::kFailed;
    }
    cv::Mat img;
    cv::resize(img_file, img, cv::Size(target_image_width, target_image_height));

    int image_width = img.cols;
    int image_height = img.rows;
    int image_size = image_width * image_height;
    int image_byte_len = image_size * 4 * sizeof(uint8_t);

    // buffer alloc
    CHECK_CUDA(cudaMalloc(&d_img_in, image_byte_len));
    CHECK_CUDA(cudaMalloc(&d_r, image_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_g, image_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, image_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pa, image_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pb, image_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_img_out, image_byte_len));

    // warm up
    common::warmUpGpu();

    // image convert
    cv::Mat mat_img_in;
    cv::cvtColor(cv::InputArray{img}, cv::OutputArray{mat_img_in}, cv::COLOR_BGR2BGRA);
    std::vector<uint8_t> vec_img_out(image_byte_len);

    MY_TIMER_BEGIN(INFO, "total")

    // copy data in
    MY_TIMER_BEGIN(INFO, "host to device {} bytes", image_byte_len)
    CHECK_CUDA(cudaMemcpy(d_img_in, mat_img_in.data, image_byte_len, cudaMemcpyHostToDevice));
    MY_TIMER_END()

    // process
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 block(32, 32);
    dim3 grid((image_width + block.x - 1) / block.x, (image_height + block.y - 1) / block.y);

    MY_TIMER_BEGIN(INFO, "kernel separate on {}x{}", image_width, image_height)
    separate<<<grid, block>>>(d_img_in, d_r, d_g, d_b, image_width, image_height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    MY_TIMER_END()

    float* buf_arr[3] = {d_b, d_g, d_r};
    for (int color_idx = 0; color_idx < 3; ++color_idx) {
        MY_TIMER_BEGIN(INFO, "kernel filter on {}x{}", image_width, image_height)

        calcAb<<<grid, block>>>(buf_arr[color_idx], d_pa, d_pb, image_width, image_height, radius,
                                eps);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        linearConv<<<grid, block>>>(buf_arr[color_idx], d_pa, d_pb, radius, d_img_out, image_width,
                                    image_height, color_idx, enhance_k, complex_k, output_mode);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        MY_TIMER_END()
    }

    // copy data out
    MY_TIMER_BEGIN(INFO, "device to host {} bytes", image_byte_len)
    CHECK_CUDA(cudaMemcpy(vec_img_out.data(), d_img_out, image_byte_len, cudaMemcpyDeviceToHost));
    MY_TIMER_END()

    MY_TIMER_END()

    // write image
    cv::Mat mat_img_out(image_height, image_width, CV_8UC4, vec_img_out.data());
    auto outfile = FSTR("{}-txi_guided-cuda{}", fpath.stem(), fpath.extension());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), mat_img_out);

    // free
    CHECK_CUDA(cudaFree(d_img_in));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_g));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_pa));
    CHECK_CUDA(cudaFree(d_pb));
    CHECK_CUDA(cudaFree(d_img_out));
    ILOG("done!");

    return MyErrCode::kOk;
}