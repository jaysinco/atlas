#include <fmt/ostream.h>
#include <torch/all.h>
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    ILOG("CUDA is available: {}", torch::cuda::is_available());
    torch::Tensor tensor = torch::rand({2, 3});
    ILOG("\n{}", fmt::streamed(tensor));
}