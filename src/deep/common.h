#pragma once
#include <torch/torch.h>
#include <fmt/ostream.h>
#include "toolkit/timer.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "fwd.h"

template <>
struct fmt::formatter<torch::Tensor>: fmt::ostream_formatter
{
};
