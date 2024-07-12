#include "toolkit/args.h"
#include "toolkit/logging.h"

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

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();
    func1();
}
