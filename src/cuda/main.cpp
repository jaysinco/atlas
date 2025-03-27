#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "fwd.h"

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    MY_TRY
    // CHECK_ERR_RTI(helloWorld(argc, argv));
    // CHECK_ERR_RTI(checkDevice(argc, argv));
    // CHECK_ERR_RTI(sumMatrix(argc, argv));
    // CHECK_ERR_RTI(reduceInteger(argc, argv));
    // CHECK_ERR_RTI(nestedHelloWorld(argc, argv));
    // CHECK_ERR_RTI(globalVariable(argc, argv));
    // CHECK_ERR_RTI(cufftTest(argc, argv));
    // CHECK_ERR_RTI(juliaSet(argc, argv));
    // CHECK_ERR_RTI(dotProduct(argc, argv));
    // CHECK_ERR_RTI(rayTracing(argc, argv));
    // CHECK_ERR_RTI(txiGaussian(argc, argv));
    // CHECK_ERR_RTI(txiGuided(argc, argv));
    // CHECK_ERR_RTI(trtMnist(argc, argv));
    CHECK_ERR_RTI(contrastLG(argc, argv));
    MY_CATCH_RTI return 0;
}