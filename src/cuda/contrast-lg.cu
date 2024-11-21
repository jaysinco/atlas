#include "./common.cuh"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <cuda_runtime.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cufft.h>
#include <cublas_v2.h>

__global__ void genRho(float* rho, float scala, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    int a = -nc / 2 + ic;
    int b = nr / 2 - ir;
    rho[ir * nc + ic] = sqrt(static_cast<float>(a * a + b * b)) / scala;
}

static MyErrCode filterLG(int num_r, int num_c, int n, int k, float scala, cufftComplex*& d_lrt)
{
    float* d_rho;
    float* d_theta;
    CHECK_CUDA(cudaMalloc(&d_lrt, sizeof(cufftComplex) * num_r * num_c));
    CHECK_CUDA(cudaMalloc(&d_rho, sizeof(float) * num_r * num_c));

    dim3 block(32, 32);
    dim3 grid((num_c + block.x - 1) / block.x, (num_r + block.y - 1) / block.y);
    genRho<<<grid, block>>>(d_rho, scala, num_c, num_r);
    CHECK_CUDA(cudaGetLastError());

    // print2D(d_rho, false, num_c, num_r, 600 - 1, 998 - 1, 10, 10);

    CHECK_CUDA(cudaFree(d_rho));
    return MyErrCode::kOk;
}

static MyErrCode prepareLG(int nrsize, int ncsize, std::vector<float> const& scalas, int n, int k,
                           float scala0, std::vector<cufftComplex*>& d_mat_lg, float*& d_denom,
                           cufftComplex*& d_hh)
{
    cufftComplex* f_lg;
    CHECK_ERR_RET(filterLG(nrsize, ncsize, n, k, scalas.at(0), f_lg));
    return MyErrCode::kOk;
}

MyErrCode contrastLG(int argc, char** argv)
{
    // parameter
    float degree = 1.5;
    std::vector<float> scalas = {14.405, 7.2025, 3.6013, 1.8006, 0.9003};
    int nn = 1;
    int k = 0;
    float scala0 = 40;
    auto fpath = toolkit::getDataDir() / "hdr.jpg";
    int target_image_width = 1920;   // 1920 2560 3840
    int target_image_height = 1080;  // 1080 1440 2160

    // read image
    cv::Mat img_file = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath.string());
        return MyErrCode::kFailed;
    }
    cv::Mat img_in;
    cv::resize(img_file, img_in, cv::Size(target_image_width, target_image_height));

    int image_width = img_in.cols;
    int image_height = img_in.rows;
    int image_size = image_width * image_height;
    int image_byte_len = image_size * 3 * sizeof(uint8_t);
    int padded_image_width = image_width * 3;
    int padded_image_height = image_height * 3;
    int padded_image_size = padded_image_width * padded_image_height;

    // buffer alloc
    uint8_t* d_img_in;
    uint8_t* d_img_out;
    std::vector<cufftComplex*> d_mat_lg;
    float* d_denom;
    cufftComplex* d_hh;

    CHECK_CUDA(cudaMalloc(&d_img_in, image_byte_len));
    CHECK_CUDA(cudaMalloc(&d_img_out, image_byte_len));
    CHECK_CUDA(cudaMalloc(&d_denom, sizeof(float) * image_size));
    CHECK_CUDA(cudaMalloc(&d_hh, sizeof(cufftComplex) * padded_image_size));

    // warm up
    warmUpGpu();

    // prepare lg
    CHECK_ERR_RET(
        prepareLG(image_height, image_width, scalas, nn, k, scala0, d_mat_lg, d_denom, d_hh));

    TIMER_BEGIN(total)

    // copy data in
    CHECK_CUDA(cudaMemcpy(d_img_in, img_in.data, image_byte_len, cudaMemcpyHostToDevice));

    // copy data out
    std::vector<uint8_t> vec_img_out(image_byte_len);
    CHECK_CUDA(cudaMemcpy(vec_img_out.data(), d_img_out, image_byte_len, cudaMemcpyDeviceToHost));

    TIMER_END(total, "total")

    // write image
    cv::Mat mat_img_out(image_height, image_width, CV_8UC3, vec_img_out.data());
    auto outfile = FSTR("{}-contrast-lg-cuda{}", fpath.stem().string(), fpath.extension().string());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), mat_img_out);

    // free
    CHECK_CUDA(cudaFree(d_img_in));
    CHECK_CUDA(cudaFree(d_img_out));
    for (auto d_mat: d_mat_lg) {
        CHECK_CUDA(cudaFree(d_mat));
    }
    CHECK_CUDA(cudaFree(d_denom));
    CHECK_CUDA(cudaFree(d_hh));
    ILOG("done!");

    return MyErrCode::kOk;
}
