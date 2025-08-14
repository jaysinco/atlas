#include "./common.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"

using namespace ocl;

MY_MAIN
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    MY_TRY
    CHECK_ERR_RET(txiGuided(argc, argv));
    // CHECK_ERR_RET(spatialDenoise(argc, argv));
    return MyErrCode::kOk;
    MY_CATCH_RET
}
