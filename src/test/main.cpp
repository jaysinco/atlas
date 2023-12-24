#include "toolkit/logging.h"
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

int main(int argc, char** argv)
{
    toolkit::initLogger();
    Catch::Session session;
    int return_code = session.applyCommandLine(argc, argv);
    if (return_code != 0) {
        return return_code;
    }
    auto& config = session.configData();
    config.shouldDebugBreak = true;
    int num_failed = session.run();
    return num_failed;
}
