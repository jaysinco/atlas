#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include "protocol/ipv4.h"
#include <iostream>

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.positional("target", po::value<std::string>(), "ipv4 or host name", 1);
    args.optional("count,c", po::value<int>()->default_value(5), "send count");
    CHECK_ERR_RET_INT(args.parse());

    auto opt_target = args.get<std::string>("target");
    auto opt_count = args.get<int>("count");

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
    std::cout << "Ping " << ip_desc.str() << ":" << std::endl;

    auto& apt = net::Adaptor::fit(net::Ip4::kZeros);
    void* handle;
    CHECK_ERR_RET_INT(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    int recv_cnt = 0;
    int64_t sum_cost = 0;
    int64_t min_cost = std::numeric_limits<int64_t>::max();
    int64_t max_cost = std::numeric_limits<int64_t>::min();
    for (int i = 0; i < opt_count; ++i) {
        std::cout << "Reply from " << target_ip.toStr() << ": ";
        net::Packet reply;
        int64_t cost_ms;
        if (net::Transport::ping(handle, apt, target_ip, reply, cost_ms, 128, "greatjaysinco") ==
            MyErrCode::kOk) {
            if (reply.hasIcmpError()) {
                TLOG(reply.toVariant().toJsonStr(3));
                std::cout << "error" << std::endl;
                continue;
            }
            ++recv_cnt;
            min_cost = std::min(min_cost, cost_ms);
            max_cost = std::max(max_cost, cost_ms);
            sum_cost += cost_ms;
            auto& prot = *reply.getDetail().layers.at(1);
            int ttl = dynamic_cast<net::Ipv4 const&>(prot).getDetail().ttl;
            std::cout << "time" << (cost_ms == 0 ? "<1" : FSTR("={}", cost_ms)) << "ms"
                      << " ttl=" << ttl << std::endl;
        } else {
            std::cout << "timeout" << std::endl;
        }
    }
    std::cout << "\nStatistical information:" << std::endl;
    std::cout << "    packets: sent=" << opt_count << ", recv=" << recv_cnt
              << ", lost=" << (opt_count - recv_cnt) << " ("
              << static_cast<int>(static_cast<float>(opt_count - recv_cnt) / opt_count * 100)
              << "% lost)\n";
    if (sum_cost > 0) {
        std::cout << "Estimated time of round trip:" << std::endl;
        std::cout << "    min=" << min_cost << "ms, max=" << max_cost
                  << "ms, avg=" << (sum_cost) / opt_count << "ms\n";
    }
    MY_CATCH_RET_INT
}
