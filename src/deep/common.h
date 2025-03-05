#pragma once
#include <torch/torch.h>
#include <fmt/ostream.h>
#include "toolkit/timer.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "fwd.h"

#define ADD_NN_MOD(name, type, opt) \
    torch::nn::type name = register_module(#name, torch::nn::type(torch::nn::opt))

template <>
struct fmt::formatter<torch::Tensor>: fmt::ostream_formatter
{
};
