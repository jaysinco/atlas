#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "fwd.h"

MY_MAIN
{
    toolkit::installCrashHook();
    MY_TRY
    // CHECK_ERR_RET(linearRegression(argc, argv));
    // CHECK_ERR_RET(fashionMnist(argc, argv));
    // CHECK_ERR_RET(poemWriter(argc, argv));
    CHECK_ERR_RET(essayWriter(argc, argv));
    return MyErrCode::kOk;
    MY_CATCH_RET
}
