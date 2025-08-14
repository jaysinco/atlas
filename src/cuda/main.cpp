#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "fwd.h"

MY_MAIN
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    MY_TRY
    // CHECK_ERR_RET(helloWorld(argc, argv));
    // CHECK_ERR_RET(checkDevice(argc, argv));
    // CHECK_ERR_RET(sumMatrix(argc, argv));
    // CHECK_ERR_RET(reduceInteger(argc, argv));
    // CHECK_ERR_RET(nestedHelloWorld(argc, argv));
    // CHECK_ERR_RET(globalVariable(argc, argv));
    // CHECK_ERR_RET(cufftTest(argc, argv));
    // CHECK_ERR_RET(juliaSet(argc, argv));
    // CHECK_ERR_RET(dotProduct(argc, argv));
    // CHECK_ERR_RET(rayTracing(argc, argv));
    // CHECK_ERR_RET(txiGaussian(argc, argv));
    // CHECK_ERR_RET(txiGuided(argc, argv));
    // CHECK_ERR_RET(trtMnist(argc, argv));
    CHECK_ERR_RET(contrastLG(argc, argv));
    MY_CATCH_RET
    return MyErrCode::kOk;
}
