#include "./fwd.h"
#include "toolkit/args.h"

int main(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();

    // helloWorld(argc, argv);
    // checkDevice(argc, argv);
    // sumMatrix(argc, argv);
    // reduceInteger(argc, argv);
    // nestedHelloWorld(argc, argv);
    // globalVariable(argc, argv);
    // cufftTest(argc, argv);
    // juliaSet(argc, argv);
    // dotProduct(argc, argv);
    // rayTracing(argc, argv);
    // txiGaussian(argc, argv);
    // txiGuided(argc, argv);
    // trtMnist(argc, argv);
    contrastLG(argc, argv);
    return 0;
}