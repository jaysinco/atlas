#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include "decl.h"

MY_MAIN
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
    return MyErrCode::kOk;
}
