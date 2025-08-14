#include "toolkit/logging.h"
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

MY_MAIN
{
    CHECK_ERR_RET(toolkit::initLogger());
    Catch::Session session;
    int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return MyErrCode::kFailed;
    }
    auto& config = session.configData();
    config.shouldDebugBreak = true;
    int num_failed = session.run();
    if (num_failed != 0) {
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}
