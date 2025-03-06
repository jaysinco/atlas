#define _USE_MATH_DEFINES
#include <cmath>
#include "./common.cuh"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <cuda_runtime.h>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cufft.h>
#include <cublas_v2.h>

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
    cufftHandle c2c_pad_plan = 0;
    CHECK_CUFFT(cufftPlan2d(&c2c_pad_plan, padded_row, padded_col, CUFFT_C2C));

    CHECK_CUDA(cudaMalloc(&d_denom, sizeof(float) * nrsize * ncsize));
    CHECK_CUDA(cudaMemset(d_denom, 0, sizeof(float) * nrsize * ncsize));
    CHECK_CUDA(cudaMalloc(&d_hh, sizeof(cuComplex) * padded_row * padded_col));
    CHECK_CUDA(cudaDeviceSynchronize());

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

    cuComplex* d_gau_ker;
    CHECK_ERR_RET(common::getGaussianKernel(padded_row, padded_col, scala0, d_gau_ker));
    // print2D(d_gau_ker, false, padded_col, padded_row, 1 - 1, 1 - 1, 5, 5);

    // ILOG("======");

    CHECK_CUFFT(cufftExecC2C(c2c_pad_plan, d_gau_ker, d_hh, CUFFT_FORWARD));
    // print2D(d_hh, false, padded_col, padded_row, 1 - 1, 1 - 1, 5, 5);

    CHECK_CUFFT(cufftDestroy(c2c_plan));
    CHECK_CUFFT(cufftDestroy(c2c_pad_plan));
    CHECK_CUDA(cudaFree(d_gau_ker));

    return MyErrCode::kOk;
}

static __global__ void rgb2ntsc(uint8_t const* img_in, int yiq_idx, float* yiq, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    float b = img_in[ir * nc * 3 + ic * 3 + 0] / 255.0;
    float g = img_in[ir * nc * 3 + ic * 3 + 1] / 255.0;
    float r = img_in[ir * nc * 3 + ic * 3 + 2] / 255.0;

    float v = 0;
    if (yiq_idx == 0) {
        v = 0.299 * r + 0.587 * g + 0.114 * b;
        v = log(v + 1);
    } else if (yiq_idx == 1) {
        v = 0.596 * r - 0.274 * g - 0.322 * b;
    } else if (yiq_idx == 2) {
        v = 0.211 * r - 0.523 * g + 0.312 * b;
    }

    yiq[ir * nc + ic] = v;
}

static __global__ void subLp(float const* d_yiq, cuComplex const* d_lp, cuComplex* d_fft, int nc,
                             int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }

    cuComplex lp = d_lp[(ir + nr) * 3 * nc + (ic + nc)];
    float yiq = d_yiq[ir * nc + ic];
    d_fft[ir * nc + ic] = make_cuComplex(yiq - lp.x, 0);
}

static __global__ void genEdge(cuComplex* fft_edge, cuComplex const* mat_lg, float const* denom,
                               int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }

    cuComplex edge = fft_edge[ir * nc + ic];
    cuComplex lg = cuConjf(mat_lg[ir * nc + ic]);
    float deno = denom[ir * nc + ic];
    cuComplex temp = cuCmulf(edge, lg);
    fft_edge[ir * nc + ic] = make_cuComplex(temp.x / deno, temp.y / deno);
}

static __global__ void addToOut(float* out_sum, cuComplex const* out_f, float scala, int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    out_sum[ir * nc + ic] += scala * out_f[ir * nc + ic].x;
}

static __global__ void addOutLp(float const* out_sum, cuComplex const* lp, float* yiq, int nc,
                                int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    yiq[ir * nc + ic] = out_sum[ir * nc + ic] + lp[(ir + nr) * 3 * nc + (ic + nc)].x;
}

static MyErrCode scaleChannel(int nrsize, int ncsize, uint8_t const* d_img_in, int yiq_idx,
                              float* d_yiq, cuComplex const* d_hh, std::vector<float> const& scalas,
                              std::vector<cuComplex*> const& d_mat_lg, float degree,
                              float const* d_denom)
{
    int padded_row = nrsize * 3;
    int padded_col = ncsize * 3;

    dim3 block(32, 32);
    dim3 grid((ncsize + block.x - 1) / block.x, (nrsize + block.y - 1) / block.y);
    dim3 padded_grid((padded_col + block.x - 1) / block.x, (padded_row + block.y - 1) / block.y);

    cuComplex* d_padded_yiq;
    cuComplex* d_fft_yiq;
    cuComplex* d_lp_yiq;
    cuComplex* d_fft_out;
    cuComplex* d_fft_edge;
    float* d_out_sum;
    CHECK_CUDA(cudaMalloc(&d_fft_yiq, sizeof(cuComplex) * padded_row * padded_col));
    CHECK_CUDA(cudaMalloc(&d_lp_yiq, sizeof(cuComplex) * padded_row * padded_col));
    CHECK_CUDA(cudaMalloc(&d_fft_out, sizeof(cuComplex) * nrsize * ncsize));
    CHECK_CUDA(cudaMalloc(&d_fft_edge, sizeof(cuComplex) * nrsize * ncsize));
    CHECK_CUDA(cudaMalloc(&d_out_sum, sizeof(float) * nrsize * ncsize));
    CHECK_CUDA(cudaMemset(d_out_sum, 0, sizeof(float) * nrsize * ncsize));

    cufftHandle c2c_pad_plan = 0;
    CHECK_CUFFT(cufftPlan2d(&c2c_pad_plan, padded_row, padded_col, CUFFT_C2C));
    cufftHandle c2c_plan = 0;
    CHECK_CUFFT(cufftPlan2d(&c2c_plan, nrsize, ncsize, CUFFT_C2C));

    // process
    rgb2ntsc<<<grid, block>>>(d_img_in, yiq_idx, d_yiq, ncsize, nrsize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_ERR_RET(common::padArrayRepBoth(d_yiq, ncsize, nrsize, d_padded_yiq, ncsize, nrsize));
    CHECK_CUFFT(cufftExecC2C(c2c_pad_plan, d_padded_yiq, d_fft_yiq, CUFFT_FORWARD));
    CHECK_ERR_RET(common::arrayMul(d_fft_yiq, d_hh, padded_row * padded_col));
    CHECK_CUFFT(cufftExecC2C(c2c_pad_plan, d_fft_yiq, d_lp_yiq, CUFFT_INVERSE));
    CHECK_ERR_RET(common::arrayDiv(d_lp_yiq, padded_row * padded_col, padded_row * padded_col));
    CHECK_ERR_RET(common::fftshift2(d_lp_yiq, padded_col, padded_row));
    subLp<<<grid, block>>>(d_yiq, d_lp_yiq, d_fft_out, ncsize, nrsize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUFFT(cufftExecC2C(c2c_plan, d_fft_out, d_fft_out, CUFFT_FORWARD));

    for (int i = 0; i < scalas.size(); ++i) {
        CHECK_ERR_RET(common::arrayMul(d_fft_out, d_mat_lg.at(i), d_fft_edge, nrsize * ncsize));
        CHECK_CUFFT(cufftExecC2C(c2c_plan, d_fft_edge, d_fft_edge, CUFFT_INVERSE));
        CHECK_ERR_RET(common::arrayDiv(d_fft_edge, nrsize * ncsize, nrsize * ncsize));
        CHECK_ERR_RET(common::arrayMul(d_fft_edge,
                                       1.0f / scalas.at(i) * degree / (yiq_idx == 0 ? 1.0 : 1.01),
                                       nrsize * ncsize));
        CHECK_CUFFT(cufftExecC2C(c2c_plan, d_fft_edge, d_fft_edge, CUFFT_FORWARD));
        genEdge<<<grid, block>>>(d_fft_edge, d_mat_lg.at(i), d_denom, ncsize, nrsize);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUFFT(cufftExecC2C(c2c_plan, d_fft_edge, d_fft_edge, CUFFT_INVERSE));
        CHECK_ERR_RET(common::arrayDiv(d_fft_edge, nrsize * ncsize, nrsize * ncsize));
        addToOut<<<grid, block>>>(d_out_sum, d_fft_edge, scalas.at(i), ncsize, nrsize);
        CHECK_CUDA(cudaGetLastError());
    }

    addOutLp<<<grid, block>>>(d_out_sum, d_lp_yiq, d_yiq, ncsize, nrsize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ILOG("================");
    // print2D(d_yiq, false, ncsize, nrsize, 1 - 1, 1 - 1, 5, 8, 10);

    CHECK_CUFFT(cufftDestroy(c2c_pad_plan));
    CHECK_CUFFT(cufftDestroy(c2c_plan));
    CHECK_CUDA(cudaFree(d_padded_yiq));
    CHECK_CUDA(cudaFree(d_fft_yiq));
    CHECK_CUDA(cudaFree(d_lp_yiq));
    CHECK_CUDA(cudaFree(d_fft_out));
    CHECK_CUDA(cudaFree(d_fft_edge));
    CHECK_CUDA(cudaFree(d_out_sum));

    return MyErrCode::kOk;
}

static __global__ void ntsc2rgb(float const* yy, float const* ii, float const* qq, uint8_t* img_out,
                                int nc, int nr)
{
    unsigned int ic = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ir = blockIdx.y * blockDim.y + threadIdx.y;
    if (ic >= nc || ir >= nr) {
        return;
    }
    float y = exp(yy[ir * nc + ic]) - 1;
    float i = ii[ir * nc + ic];
    float q = qq[ir * nc + ic];
    float r = y + 0.956 * i + 0.621 * q;
    float g = y - 0.272 * i - 0.647 * q;
    float b = y - 1.106 * i + 1.703 * q;
    img_out[ir * nc * 3 + ic * 3 + 0] = round(common::clamp(b, 0.0, 1.0) * 255.0);
    img_out[ir * nc * 3 + ic * 3 + 1] = round(common::clamp(g, 0.0, 1.0) * 255.0);
    img_out[ir * nc * 3 + ic * 3 + 2] = round(common::clamp(r, 0.0, 1.0) * 255.0);
}

static MyErrCode multiscale(int nrsize, int ncsize, uint8_t* d_img_in, uint8_t* d_img_out,
                            float degree, std::vector<cuComplex*> const& d_mat_lg, float* d_denom,
                            std::vector<float> const& scalas, cuComplex* d_hh)
{
    float* d_yy;
    float* d_ii;
    float* d_qq;
    CHECK_CUDA(cudaMalloc(&d_yy, nrsize * ncsize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ii, nrsize * ncsize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_qq, nrsize * ncsize * sizeof(float)));

    CHECK_ERR_RET(
        scaleChannel(nrsize, ncsize, d_img_in, 0, d_yy, d_hh, scalas, d_mat_lg, degree, d_denom));
    CHECK_ERR_RET(
        scaleChannel(nrsize, ncsize, d_img_in, 1, d_ii, d_hh, scalas, d_mat_lg, degree, d_denom));
    CHECK_ERR_RET(
        scaleChannel(nrsize, ncsize, d_img_in, 2, d_qq, d_hh, scalas, d_mat_lg, degree, d_denom));

    dim3 block(32, 32);
    dim3 grid((ncsize + block.x - 1) / block.x, (nrsize + block.y - 1) / block.y);
    ntsc2rgb<<<grid, block>>>(d_yy, d_ii, d_qq, d_img_out, ncsize, nrsize);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_yy));
    CHECK_CUDA(cudaFree(d_ii));
    CHECK_CUDA(cudaFree(d_qq));

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
        ELOG("failed to load image file: {}", fpath);
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

    // prepare
    MY_TIMER_BEGIN(INFO, "prepare")
    CHECK_ERR_RET(
        prepareLG(image_height, image_width, scalas, nn, k, scala0, d_mat_lg, d_denom, d_hh));
    CHECK_CUDA(cudaDeviceSynchronize());
    MY_TIMER_END()

    // process
    std::vector<uint8_t> vec_img_out(image_byte_len);
    MY_TIMER_BEGIN(INFO, "process")
    CHECK_CUDA(cudaMemcpy(d_img_in, img_in.data, image_byte_len, cudaMemcpyHostToDevice));
    CHECK_ERR_RET(multiscale(image_height, image_width, d_img_in, d_img_out, degree, d_mat_lg,
                             d_denom, scalas, d_hh));
    CHECK_CUDA(cudaMemcpy(vec_img_out.data(), d_img_out, image_byte_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    MY_TIMER_END()

    // write image
    cv::Mat mat_img_out(image_height, image_width, CV_8UC3, vec_img_out.data());
    auto outfile = FSTR("{}-contrast-lg-cuda{}", fpath.stem(), fpath.extension());
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
