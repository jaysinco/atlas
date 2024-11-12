#include "./fwd.h"
#include "toolkit/args.h"

int main(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    // hello_world(argc, argv);
    // check_device(argc, argv);
    // sum_matrix(argc, argv);
    // reduce_integer(argc, argv);
    // nested_hello_world(argc, argv);
    // global_variable(argc, argv);
    // test_cufft(argc, argv);
    julia_set(argc, argv);
    // dot_product(argc, argv);
    // ray_tracing(argc, argv);
    // txi_gaussian(argc, argv);
    // txi_guided(argc, argv);
    // trt_mnist(argc, argv);
    return 0;
}