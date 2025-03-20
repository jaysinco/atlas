#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "fwd.h"

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    MY_TRY
    // CHECK_ERR_RET_INT(linearRegression(argc, argv));
    // CHECK_ERR_RET_INT(fashionMnist(argc, argv));
    CHECK_ERR_RET_INT(poemGenerator(argc, argv));
    MY_CATCH_RET_INT
    return 0;
}