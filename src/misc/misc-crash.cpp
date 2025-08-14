#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"

void func5()
{
    ILOG("about to crash...");
    int* p = nullptr;
    *p = 666;
}

void func4() { func5(); }

void func3() { func4(); }

void func2() { func3(); }

void func1() { func2(); }

MY_MAIN
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();
    func1();
    return MyErrCode::kOk;
}
