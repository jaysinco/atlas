#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"

int addSub(int a, int b, int c, int d)
{
    int result;
    __asm__ volatile(
        R"(
            mov %1, %%eax
            add %2, %%eax
            add %3, %%eax
            sub %4, %%eax
            mov %%eax, %0
        )"
        : "=r"(result)
        : "r"(a), "r"(b), "r"(c), "r"(d)
        : "%eax");
    return result;
}

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();
    int a = 1;
    int b = 2;
    int c = 3;
    int d = 40;
    int result = addSub(a, b, c, d);
    ILOG("{}+{}+{}-{}={}", a, b, c, d, result);
}
