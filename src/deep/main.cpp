#include "toolkit/toolkit.h"
#include "toolkit/logging.h"
#include "fwd.h"

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    MY_TRY
    // CHECK_ERR_RTI(linearRegression(argc, argv));
    // CHECK_ERR_RTI(fashionMnist(argc, argv));
    CHECK_ERR_RTI(poemGenerator(argc, argv));
    MY_CATCH_RTI return 0;
}