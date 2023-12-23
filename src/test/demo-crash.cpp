#include "toolkit/args.h"

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
    utils::installCrashHook();
    INIT_LOG(argc, argv);
    func1();
}
