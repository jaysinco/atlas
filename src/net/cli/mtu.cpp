#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::runAsRoot(argc, argv);
    toolkit::Args args(argc, argv);
    args.positional("target", po::value<std::string>(), "ipv4 or host name", 1);
    args.optional("max", po::value<int>()->default_value(1500), "high bound for mtu");
    CHECK_ERR_RTI(args.parse());

    auto opt_target = args.get<std::string>("target");
    auto opt_max = args.get<int>("max");

    net::Ip4 target_ip;
    std::ostringstream ip_desc;
    if (net::Ip4::fromDottedDec(opt_target, &target_ip) == MyErrCode::kOk) {
        ip_desc << target_ip.toStr();
    } else {
        if (net::Ip4::fromDomain(opt_target, &target_ip) != MyErrCode::kOk) {
            ELOG("invalid ip or host name: {}", opt_target);
            return -1;
        }
        ip_desc << opt_target << " [" << target_ip.toStr() << "]";
    }
    ILOG("Ping {}", ip_desc.str());

    auto& apt = net::Adaptor::fit(net::Ip4::kZeros);
    void* handle;
    CHECK_ERR_RTI(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    int nbytes;
    CHECK_ERR_RTI(net::Transport::calcMtu(handle, apt, target_ip, nbytes, opt_max));
    ILOG("MTU={}", nbytes);

    MY_CATCH_RTI
}
