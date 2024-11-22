#include "./common.cuh"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <cuda_runtime.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cufft.h>
#include <cublas_v2.h>
#include <cmath>

static __global__ void genRho(float* rho, float scala, int nc, int nr)
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

static __global__ void genTheta(float* theta, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    int a = -nc / 2 + ic;
    int b = nr / 2 - ir;
    theta[ir * nc + ic] = atan2(static_cast<float>(-b), static_cast<float>(-a));
}

static __global__ void genLrt(cuComplex* lrt, int n, int k, float scala, float const* theta,
                              float const* rho, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }

    float rho1 = rho[ir * nc + ic];
    float theta1 = theta[ir * nc + ic];
    float c1 = pow(-1, k) * pow(2, n + 1) / 2 * pow(M_PI, n / 2.0) / scala;
    float c2 = c1 * pow(rho1, n) * exp(-1 * M_PI * pow(rho1, 2));
    cuComplex c3 = common::cexpf(make_cuComplex(0, n * theta1));
    c3.x *= c2;
    c3.y *= c2;
    lrt[ir * nc + ic] = c3;
}

static __global__ void genDenom(float* denom, cuComplex const* fft_lg, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    cuComplex lg = fft_lg[ir * nc + ic];
    denom[ir * nc + ic] += pow(lg.x, 2) + pow(lg.y, 2);
}

static __global__ void addDenom(float* denom, float delta, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    denom[ir * nc + ic] += delta;
}

static MyErrCode filterLG(int num_r, int num_c, int n, int k, float scala, cuComplex*& d_lrt)
{
    float* d_rho;
    float* d_theta;
    CHECK_CUDA(cudaMalloc(&d_lrt, sizeof(cuComplex) * num_r * num_c));
    CHECK_CUDA(cudaMalloc(&d_rho, sizeof(float) * num_r * num_c));
    CHECK_CUDA(cudaMalloc(&d_theta, sizeof(float) * num_r * num_c));

    dim3 block(32, 32);
    dim3 grid((num_c + block.x - 1) / block.x, (num_r + block.y - 1) / block.y);
    genRho<<<grid, block>>>(d_rho, scala, num_c, num_r);
    CHECK_CUDA(cudaGetLastError());
    genTheta<<<grid, block>>>(d_theta, num_c, num_r);
    CHECK_CUDA(cudaGetLastError());
    genLrt<<<grid, block>>>(d_lrt, n, k, scala, d_theta, d_rho, num_c, num_r);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_rho));
    CHECK_CUDA(cudaFree(d_theta));
    return MyErrCode::kOk;
}

static MyErrCode prepareLG(int nrsize, int ncsize, std::vector<float> const& scalas, int n, int k,
                           float scala0, std::vector<cuComplex*>& d_mat_lg, float*& d_denom,
                           cuComplex*& d_hh)
{
    int padded_row = nrsize * 3;
    int padded_col = ncsize * 3;

    cufftHandle c2c_plan = 0;
    CHECK_CUFFT(cufftPlan2d(&c2c_plan, nrsize, ncsize, CUFFT_C2C));
    cufftHandle r2c_plan = 0;
    CHECK_CUFFT(cufftPlan2d(&r2c_plan, padded_row, padded_col, CUFFT_R2C));

    CHECK_CUDA(cudaMalloc(&d_denom, sizeof(float) * nrsize * ncsize));
    CHECK_CUDA(cudaMemset(d_denom, 0, sizeof(float) * nrsize * ncsize));
    CHECK_CUDA(cudaMalloc(&d_hh, sizeof(cuComplex) * padded_row * padded_col));

    dim3 block(32, 32);
    dim3 grid((ncsize + block.x - 1) / block.x, (nrsize + block.y - 1) / block.y);

    for (int i = 0; i < scalas.size(); ++i) {
        cuComplex* f_lg;
        CHECK_ERR_RET(filterLG(nrsize, ncsize, n, k, scalas[i], f_lg));
        CHECK_CUFFT(cufftExecC2C(c2c_plan, f_lg, f_lg, CUFFT_FORWARD));
        d_mat_lg.push_back(f_lg);
        genDenom<<<grid, block>>>(d_denom, f_lg, ncsize, nrsize);
        CHECK_CUDA(cudaGetLastError());
    }

    float denom_max = common::arrayMax(d_denom, nrsize * ncsize);
    addDenom<<<grid, block>>>(d_denom, denom_max * 0.001, ncsize, nrsize);
    CHECK_CUDA(cudaGetLastError());

    float* d_gau_ker;
    CHECK_ERR_RET(common::getGaussianKernel(padded_row, padded_col, scala0, d_gau_ker));
    print2D(d_gau_ker, false, padded_col, padded_row, 5756 - 1, 3236 - 1, 5, 5);

    ILOG("max={}", common::arrayMax(d_gau_ker, padded_col * padded_row));

    CHECK_CUFFT(cufftExecR2C(r2c_plan, d_gau_ker, d_hh));
    print2D(d_hh, false, padded_col, padded_row, 1 - 1, 1 - 1, 5, 5);

    CHECK_CUFFT(cufftDestroy(c2c_plan));
    CHECK_CUFFT(cufftDestroy(r2c_plan));
    CHECK_CUDA(cudaFree(d_gau_ker));

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

    ILOG("size={}x{}, padded={}x{}", image_width, image_height, padded_image_width,
         padded_image_height);

    // buffer alloc
    uint8_t* d_img_in;
    uint8_t* d_img_out;
    std::vector<cuComplex*> d_mat_lg;
    float* d_denom;
    cuComplex* d_hh;

    CHECK_CUDA(cudaMalloc(&d_img_in, image_byte_len));
    CHECK_CUDA(cudaMalloc(&d_img_out, image_byte_len));

    // warm up
    common::warmUpGpu();

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
