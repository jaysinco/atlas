#pragma once
#include <torch/torch.h>
#include "toolkit/timer.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "fwd.h"

template <size_t D>
using IntArray = std::array<int64_t, D>;

struct InceptionOptions
{
    InceptionOptions(int64_t c0, int64_t c1, IntArray<2> c2, IntArray<2> c3, int64_t c4)
        : c0_(c0), c1_(c1), c2_(c2), c3_(c3), c4_(c4)
    {
    }

    // c0 is the number of input channels
    TORCH_ARG(int64_t, c0);

    // c1-c4 are the number of output channels for each branch
    TORCH_ARG(int64_t, c1);
    TORCH_ARG(IntArray<2>, c2);
    TORCH_ARG(IntArray<2>, c3);
    TORCH_ARG(int64_t, c4);
};

struct InceptionImpl: torch::nn::Module
{
    explicit InceptionImpl(InceptionOptions const& o);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d b1_1_ = nullptr;
    torch::nn::Conv2d b2_1_ = nullptr;
    torch::nn::Conv2d b2_2_ = nullptr;
    torch::nn::Conv2d b3_1_ = nullptr;
    torch::nn::Conv2d b3_2_ = nullptr;
    torch::nn::MaxPool2d b4_1_ = nullptr;
    torch::nn::Conv2d b4_2_ = nullptr;
};

TORCH_MODULE(Inception);
