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
    // CHECK_ERR_RET_INT(helloWorld(argc, argv));
    // CHECK_ERR_RET_INT(checkDevice(argc, argv));
    // CHECK_ERR_RET_INT(sumMatrix(argc, argv));
    // CHECK_ERR_RET_INT(reduceInteger(argc, argv));
    // CHECK_ERR_RET_INT(nestedHelloWorld(argc, argv));
    // CHECK_ERR_RET_INT(globalVariable(argc, argv));
    // CHECK_ERR_RET_INT(cufftTest(argc, argv));
    // CHECK_ERR_RET_INT(juliaSet(argc, argv));
    // CHECK_ERR_RET_INT(dotProduct(argc, argv));
    // CHECK_ERR_RET_INT(rayTracing(argc, argv));
    // CHECK_ERR_RET_INT(txiGaussian(argc, argv));
    // CHECK_ERR_RET_INT(txiGuided(argc, argv));
    // CHECK_ERR_RET_INT(trtMnist(argc, argv));
    CHECK_ERR_RET_INT(contrastLG(argc, argv));
    MY_CATCH_RET_INT
    return 0;
}