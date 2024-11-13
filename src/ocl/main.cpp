#include "./common.h"
#include "toolkit/args.h"

using namespace ocl;

int main(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    txiGuided(argc, argv);

    return 0;
}