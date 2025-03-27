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
    CHECK_ERR_RTI(txiGuided(argc, argv));
    // CHECK_ERR_RTI(spatialDenoise(argc, argv));
    MY_CATCH_RTI return 0;
}