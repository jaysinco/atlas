#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.positional("mac", po::value<std::string>()->default_value(""), "mac address", 1);
    CHECK_ERR_RET_INT(args.parse());

    auto opt_mac = args.get<std::string>("mac");

    if (opt_mac.empty()) {
        ELOG("empty mac address, please input mac");
        return -1;
    }

    net::Mac mac(opt_mac);
    auto& apt = net::Adaptor::fit();
    void* handle;
    CHECK_ERR_RET_INT(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    net::Ip4 ip;
    if (net::Transport::mac2ip(handle, mac, ip) == MyErrCode::kOk) {
        ILOG("{} is at {}", mac, ip);
    } else {
        ILOG("{} is offline", mac);
    }
    MY_CATCH_RET_INT
}
