#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "protocol/ipv4.h"
#include <iostream>
#include <iomanip>

MY_MAIN
{
    MY_TRY
    toolkit::runAsRoot(argc, argv);
    toolkit::Args args(argc, argv);
    args.positional("target", po::value<std::string>(), "ipv4 or host name", 1);
    CHECK_ERR_RET(args.parse());

    auto opt_target = args.get<std::string>("target");

    net::Ip4 target_ip;
    std::ostringstream ip_desc;
    if (net::Ip4::fromDottedDec(opt_target, &target_ip) == MyErrCode::kOk) {
        ip_desc << target_ip.toStr();
    } else {
        if (net::Ip4::fromDomain(opt_target, &target_ip) != MyErrCode::kOk) {
            ELOG("invalid ip or host name: {}", opt_target);
            return MyErrCode::kFailed;
        }
        ip_desc << opt_target << " [" << target_ip.toStr() << "]";
    }
    std::cout << "Route traced to " << ip_desc.str() << std::endl;

    auto& apt = net::Adaptor::fit(net::Ip4::kZeros);
    void* handle;
    CHECK_ERR_RET(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    int ttl = 0;
    constexpr int kEpochCnt = 3;
    while (true) {
        ++ttl;
        net::Ip4 router_ip;
        int timeout_cnt = 0;
        std::cout << std::setw(2) << ttl << " " << std::flush;
        for (int i = 0; i < kEpochCnt; ++i) {
            std::cout << std::setw(6);
            net::Packet reply;
            int64_t cost_ms;
            if (net::Transport::ping(handle, apt, target_ip, reply, cost_ms, ttl, "hello,world",
                                     false, 3000) == MyErrCode::kOk) {
                if (reply.hasIcmpError()) {
                    auto& ih = dynamic_cast<net::Ipv4 const&>(*reply.getDetail().layers.at(1));
                    router_ip = ih.getDetail().sip;
                    std::cout << cost_ms << "ms" << std::flush;
                } else {
                    router_ip = target_ip;
                    std::cout << cost_ms << "ms" << std::flush;
                }
            } else {
                ++timeout_cnt;
                std::cout << "       *" << std::flush;
            }
        }
        if (timeout_cnt >= kEpochCnt) {
            std::cout << "  -- timeout --" << std::endl;
        } else {
            std::cout << "  " << router_ip.toStr() << std::endl;
            if (router_ip == target_ip) {
                std::cout << "Tracking is complete." << std::endl;
                break;
            }
        }
    }
    MY_CATCH_RET
    return MyErrCode::kOk;
}
