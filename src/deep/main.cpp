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
    CHECK_ERR_RET_INT(linearRegression(argc, argv));
    MY_CATCH_RET_INT
    return 0;
}