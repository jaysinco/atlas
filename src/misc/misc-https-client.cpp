#include "toolkit/args.h"
#include "toolkit/logging.h"
#include <cpr/cpr.h>

MY_MAIN
{
    toolkit::Args args(argc, argv);
    args.parse();
    cpr::Response r = cpr::Get(cpr::Url{"https://127.0.0.1:8080"}, cpr::VerifySsl{false});
    ILOG("status: {}", r.status_code);
    ILOG("type: {}", r.header["content-type"]);
    ILOG("text: {}", r.text);
    return MyErrCode::kOk;
}
