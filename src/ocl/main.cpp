#include "./common.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "toolkit/logging.h"

using namespace ocl;

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    MY_TRY
    CHECK_ERR_RET_INT(txiGuided(argc, argv));
    // CHECK_ERR_RET_INT(spatialDenoise(argc, argv));
    MY_CATCH_RET_INT
    return 0;
}