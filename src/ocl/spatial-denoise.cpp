#include "./common.h"
#include "toolkit/toolkit.h"
#include "toolkit/timer.h"
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

XXD_DECLARE_RES(DATA_SPATIAL_DENOISE_CL)

namespace ocl
{

MyErrCode spatialDenoise(int argc, char** argv)
{
    // param
    float malpha = 0.6;
    float curve_th = 200;
    float factor = 0.03;
    int est_size = 16;
    int bmotion = 1;
    int bweight = 1;

    // setup
    cl::Device dev;
    CHECK_ERR_RET(getDefaultDevice(dev));
    std::string src(XXD_GET_RES(DATA_SPATIAL_DENOISE_CL));
    cl::Context ctx({dev});
    cl::Program prog;
    CHECK_ERR_RET(buildProgram(ctx, dev, src, prog));

    int frame_width = 1920;
    int frame_height = 1088;
    size_t frame_len = frame_width * frame_height * 3;

    cl::CommandQueue cmd_q(ctx, dev,
                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);

    // cl::Kernel ker_copy_first_frame(prog, "copy_first_frame");
    cl::Kernel ker_motion_est_tss(prog, "motionEstTSS");
    cl::Kernel ker_calc_alphab(prog, "calc_alpha");
    cl::Kernel ker_copy_comp_edge(prog, "copy_comp_edge");

    cl::Buffer buf_in(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, frame_len);
    cl::Buffer buf_out(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, frame_len);
    cl::Image2D buf_last_rgb(ctx, CL_MEM_READ_WRITE, cl::ImageFormat{CL_RGBA, CL_UNSIGNED_INT8},
                             frame_width, frame_height);
    cl::Image2D buf_comp_rgb(ctx, CL_MEM_READ_WRITE, cl::ImageFormat{CL_RGBA, CL_UNSIGNED_INT8},
                             frame_width, frame_height);
    cl::Image2D buf_comp_alpha(ctx, CL_MEM_READ_WRITE, cl::ImageFormat{CL_R, CL_FLOAT},
                               frame_width / est_size, frame_height / est_size);

    // read image
    auto fpath = toolkit::getDataDir() / "hdr.jpg";
    cv::Mat img_file = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath.string());
        return MyErrCode::kFailed;
    }
    cv::Mat img_in;
    cv::resize(img_file, img_in, cv::Size(frame_width, frame_height));
    std::vector<uint8_t> img_out_data(frame_len);

    MY_TIMER_BEGIN(INFO, "total")

    // write butter
    cl::Event ev_write_image;
    if (auto err = cmd_q.enqueueWriteBuffer(buf_in, CL_FALSE, 0, frame_len, img_in.data, nullptr,
                                            &ev_write_image);
        err != CL_SUCCESS) {
        ELOG("failed to write buffer: {}", err);
        return MyErrCode::kFailed;
    }
    ev_write_image.wait();

    // copy_comp_edge
    ker_copy_comp_edge.setArg(0, buf_last_rgb);
    ker_copy_comp_edge.setArg(1, buf_comp_rgb);
    ker_copy_comp_edge.setArg(2, frame_width);
    ker_copy_comp_edge.setArg(3, frame_height);
    ker_copy_comp_edge.setArg(4, frame_height / est_size * est_size);

    cl::Event ev_copy_comp_edge;
    std::vector<cl::Event> ev_deps_copy_comp_edge = {};
    if (auto err = cmd_q.enqueueNDRangeKernel(
            ker_copy_comp_edge, cl::NullRange, cl::NDRange(frame_width, frame_height % est_size),
            cl::NullRange, &ev_deps_copy_comp_edge, &ev_copy_comp_edge);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue copy_comp_edge kernel: {}", err);
        return MyErrCode::kFailed;
    }

    // motion_est_tss
    ker_motion_est_tss.setArg(0, buf_in);
    ker_motion_est_tss.setArg(1, buf_last_rgb);
    ker_motion_est_tss.setArg(2, est_size);
    ker_motion_est_tss.setArg(3, buf_comp_rgb);
    ker_motion_est_tss.setArg(4, frame_width);
    ker_motion_est_tss.setArg(5, frame_height);
    ker_motion_est_tss.setArg(6, std::pow(est_size, 2) * sizeof(cl_float), nullptr);
    ker_motion_est_tss.setArg(7, std::pow(est_size, 2) * sizeof(cl_float), nullptr);
    ker_motion_est_tss.setArg(8, 5 * 1 * sizeof(cl_float), nullptr);
    ker_motion_est_tss.setArg(9, sizeof(cl_int), nullptr);
    ker_motion_est_tss.setArg(10, sizeof(cl_int), nullptr);
    ker_motion_est_tss.setArg(11, buf_comp_alpha);
    ker_motion_est_tss.setArg(12, factor);
    ker_motion_est_tss.setArg(13, static_cast<float>(std::pow(curve_th, 3)));
    ker_motion_est_tss.setArg(14, bweight);

    cl::Event ev_motion_est_tss;
    std::vector<cl::Event> ev_deps_motion_est_tss = {};
    if (auto err = cmd_q.enqueueNDRangeKernel(
            ker_motion_est_tss, cl::NullRange,
            cl::NDRange(frame_width / est_size * 5, frame_height / est_size * 1), cl::NDRange(5, 1),
            &ev_deps_motion_est_tss, &ev_motion_est_tss);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue motion_est_tss kernel: {}", err);
        return MyErrCode::kFailed;
    }

    // calc_alpha
    ker_calc_alphab.setArg(0, buf_in);
    ker_calc_alphab.setArg(1, buf_comp_rgb);
    ker_calc_alphab.setArg(2, buf_out);
    ker_calc_alphab.setArg(3, buf_comp_alpha);
    ker_calc_alphab.setArg(4, frame_width);
    ker_calc_alphab.setArg(5, frame_height);
    ker_calc_alphab.setArg(6, sizeof(cl_float), nullptr);
    ker_calc_alphab.setArg(7, est_size);
    ker_calc_alphab.setArg(8, bweight);
    ker_calc_alphab.setArg(9, malpha);
    ker_calc_alphab.setArg(10, buf_last_rgb);

    cl::Event ev_calc_alpha;
    std::vector<cl::Event> ev_deps_calc_alpha = {ev_motion_est_tss, ev_copy_comp_edge};
    if (auto err = cmd_q.enqueueNDRangeKernel(
            ker_calc_alphab, cl::NullRange, cl::NDRange(frame_width, frame_height),
            cl::NDRange(est_size, est_size), &ev_deps_calc_alpha, &ev_calc_alpha);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue calc_alpha kernel: {}", err);
        return MyErrCode::kFailed;
    }

    ev_calc_alpha.wait();

    // read butter
    cl::Event ev_read_image;
    if (auto err = cmd_q.enqueueReadBuffer(buf_out, CL_FALSE, 0, frame_len, img_out_data.data(),
                                           nullptr, &ev_read_image);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue read buffer: {}", err);
        return MyErrCode::kFailed;
    }
    ev_read_image.wait();

    MY_TIMER_END

    // write image
    cv::Mat img_out(frame_height, frame_width, CV_8UC3, img_out_data.data());
    auto outfile =
        FSTR("{}-spatial_denoise-ocl{}", fpath.stem().string(), fpath.extension().string());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), img_out);

    return MyErrCode::kOk;
}

}  // namespace ocl