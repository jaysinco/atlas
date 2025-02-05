#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include "decl.h"

int main(int argc, char** argv)
{
    toolkit::installCrashHook();
    toolkit::Args args(argc, argv);
    args.parse();

    ILOG("BEGIN");
    ILOG("===========");
    asmCpuid();
    ILOG("===========");
    asmPrintf();
    ILOG("===========");
    ILOG("END");
    return 0;
}
