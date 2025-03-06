#include "./common.h"
#include "toolkit/toolkit.h"
#include "toolkit/timer.h"
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

XXD_DECLARE_RES(DATA_TXI_GUIDED_CL)

namespace ocl
{

MyErrCode txiGuided(int argc, char** argv)
{
    // param
    float eps = 1000;
    int radius = 8;
    float enhance_k = 1.5;
    float complex_k = 1.0;
    int output_mode = 1;

    // setup
    cl::Device dev;
    CHECK_ERR_RET(getDefaultDevice(dev));
    std::string src(XXD_GET_RES(DATA_TXI_GUIDED_CL));
    cl::Context ctx({dev});
    cl::Program prog;
    CHECK_ERR_RET(buildProgram(ctx, dev, src, prog));

    int frame_width = 1920;
    int frame_height = 1080;
    size_t frame_len = frame_width * frame_height * 3;

    cl::CommandQueue cmd_q(ctx, dev,
                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);

    cl::Kernel ker_separate(prog, "separate");
    cl::Kernel ker_linear_conv(prog, "linear_conv");
    cl::Kernel ker_calc_ab(prog, "calc_ab");

    size_t buf_len = frame_width * frame_height * sizeof(cl_float);
    cl::Buffer buf_in(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, frame_len);
    cl::Buffer buf_out(ctx, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, frame_len);
    cl::Buffer buf_r(ctx, CL_MEM_READ_WRITE, buf_len);
    cl::Buffer buf_g(ctx, CL_MEM_READ_WRITE, buf_len);
    cl::Buffer buf_b(ctx, CL_MEM_READ_WRITE, buf_len);
    cl::Buffer buf_pa(ctx, CL_MEM_READ_WRITE, buf_len);
    cl::Buffer buf_pb(ctx, CL_MEM_READ_WRITE, buf_len);

    // read image
    auto fpath = toolkit::getDataDir() / "hdr.jpg";
    cv::Mat img_file = cv::imread(fpath.string(), cv::IMREAD_COLOR);
    if (img_file.data == nullptr) {
        ELOG("failed to load image file: {}", fpath);
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

    // separate
    ker_separate.setArg(0, buf_in);
    ker_separate.setArg(1, buf_r);
    ker_separate.setArg(2, buf_g);
    ker_separate.setArg(3, buf_b);
    ker_separate.setArg(4, frame_width);
    ker_separate.setArg(5, frame_height);

    cl::Event ev_separate;
    if (auto err = cmd_q.enqueueNDRangeKernel(ker_separate, cl::NullRange,
                                              cl::NDRange(frame_width, frame_height), cl::NullRange,
                                              nullptr, &ev_separate);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue separate kernel: {}", err);
        return MyErrCode::kFailed;
    }

    ev_separate.wait();

    // handle channel
    std::vector<cl::Buffer*> channel_bufs{&buf_b, &buf_g, &buf_r};
    for (int color_idx = 0; color_idx < 3; ++color_idx) {
        // calc_ab
        ker_calc_ab.setArg(0, *channel_bufs[color_idx]);
        ker_calc_ab.setArg(1, buf_pa);
        ker_calc_ab.setArg(2, buf_pb);
        ker_calc_ab.setArg(3, frame_width);
        ker_calc_ab.setArg(4, frame_height);
        ker_calc_ab.setArg(5, radius);
        ker_calc_ab.setArg(6, eps);

        cl::Event ev_calc_ab;
        if (auto err = cmd_q.enqueueNDRangeKernel(ker_calc_ab, cl::NullRange,
                                                  cl::NDRange(frame_width, frame_height),
                                                  cl::NullRange, nullptr, &ev_calc_ab);
            err != CL_SUCCESS) {
            ELOG("failed to enqueue calc_ab kernel: {}", err);
            return MyErrCode::kFailed;
        }

        // linear_conv
        ker_linear_conv.setArg(0, *channel_bufs[color_idx]);
        ker_linear_conv.setArg(1, buf_pa);
        ker_linear_conv.setArg(2, buf_pb);
        ker_linear_conv.setArg(3, radius);
        ker_linear_conv.setArg(4, buf_out);
        ker_linear_conv.setArg(5, frame_width);
        ker_linear_conv.setArg(6, frame_height);
        ker_linear_conv.setArg(7, color_idx);
        ker_linear_conv.setArg(8, enhance_k);
        ker_linear_conv.setArg(9, complex_k);
        ker_linear_conv.setArg(10, output_mode);

        cl::Event ev_linear_conv;
        std::vector<cl::Event> ev_deps_linear_conv = {ev_calc_ab};
        if (auto err = cmd_q.enqueueNDRangeKernel(
                ker_linear_conv, cl::NullRange, cl::NDRange(frame_width, frame_height),
                cl::NullRange, &ev_deps_linear_conv, &ev_linear_conv);
            err != CL_SUCCESS) {
            ELOG("failed to enqueue linear_conv kernel: {}", err);
            return MyErrCode::kFailed;
        }

        ev_linear_conv.wait();
    }

    // read butter
    cl::Event ev_read_image;
    if (auto err = cmd_q.enqueueReadBuffer(buf_out, CL_FALSE, 0, frame_len, img_out_data.data(),
                                           nullptr, &ev_read_image);
        err != CL_SUCCESS) {
        ELOG("failed to enqueue read buffer: {}", err);
        return MyErrCode::kFailed;
    }
    ev_read_image.wait();

    MY_TIMER_END()

    // write image
    cv::Mat img_out(frame_height, frame_width, CV_8UC3, img_out_data.data());
    auto outfile = FSTR("{}-txi_guided-ocl{}", fpath.stem(), fpath.extension());
    cv::imwrite((toolkit::getTempDir() / outfile).string(), img_out);

    return MyErrCode::kOk;
}

}  // namespace ocl