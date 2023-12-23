#include "utils/args.h"
#include <cpr/cpr.h>

int main(int argc, char** argv)
{
    INIT_LOG(argc, argv);
    cpr::Response r = cpr::Get(cpr::Url{"https://127.0.0.1:8080"}, cpr::VerifySsl{false});
    ILOG("status: {}", r.status_code);
    ILOG("type: {}", r.header["content-type"]);
    ILOG("text: {}", r.text);
    return 0;
}
