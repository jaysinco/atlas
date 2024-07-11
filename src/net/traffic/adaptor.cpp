#include "adaptor.h"
#include <algorithm>
#include <mutex>
#include "toolkit/logging.h"
#include <sys/types.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <pcap.h>

namespace net
{

toolkit::Variant Adaptor::toVariant() const
{
    toolkit::Variant j;
    j["name"] = name;
    j["desc"] = desc;
    j["mac"] = mac.toStr();
    j["ip"] = ip.toStr();
    j["mask"] = mask.toStr();
    j["gateway"] = gateway.toStr();
    return j;
}

Adaptor const& Adaptor::fit(Ip4 const& hint)
{
    auto it = std::find_if(all().begin(), all().end(), [&](Adaptor const& apt) {
        return apt.mask != Ip4::kZeros && apt.gateway != Ip4::kZeros &&
               (hint != Ip4::kZeros ? apt.ip.isLocal(hint, apt.mask) : true);
    });
    if (it == all().end()) {
        MY_THROW("no local adapter match {}", hint.toStr());
    }
    return *it;
}

bool Adaptor::isNative(Ip4 const& ip)
{
    return std::find_if(all().begin(), all().end(),
                        [&](Adaptor const& apt) { return ip == apt.ip; }) != all().end();
}

std::vector<Adaptor> const& Adaptor::all()
{
    static std::once_flag flag;
    static std::vector<Adaptor> adapters;

    std::call_once(flag, [&] {
        // struct ifaddrs* ifap;
        // if (getifaddrs(&ifap) == -1) {
        //     MY_THROW("failed to getifaddrs: {}", strerror(errno));
        // }
        // auto ifap_guard = toolkit::scopeExit([&] { freeifaddrs(ifap); });
        // for (ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
        //     if (ifa->ifa_addr == nullptr) {
        //         continue;
        //     }
        //     if (!(ifa->ifa_flags & IFF_UP) || (ifa->ifa_flags & IFF_LOOPBACK)) {
        //         continue;
        //     }
        //     Adaptor apt;
        //     apt.name = ifa->ifa_name;
        //     if (ifa->ifa_addr->sa_family == AF_INET) {
        //         apt.ip = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr;
        //     }
        //     if (ifa->ifa_addr->sa_family == AF_PACKET) {
        //         struct sockaddr_ll* sa_ll = (struct sockaddr_ll*)ifa->ifa_addr;
        //         std::stringstream mac_str;
        //         for (int i = 0; i < sa_ll->sll_halen; ++i) {
        //             mac_str << std::hex << std::setw(2) << std::setfill('0')
        //                     << (unsigned int)sa_ll->sll_addr[i];
        //             if (i < sa_ll->sll_halen - 1) mac_str << ":";
        //         }
        //         info.mac = mac_str.str();
        //     }
        // }

        pcap_if_t* alldevs;
        char errbuf[PCAP_ERRBUF_SIZE] = {0};
        if (pcap_findalldevs(&alldevs, errbuf) != 0) {
            MY_THROW("failed to find all devs: {}", errbuf);
        }
        auto alldevs_guard = toolkit::scopeExit([&] { pcap_freealldevs(alldevs); });
        for (pcap_if_t* d = alldevs; d != nullptr; d = d->next) {
            if (d->addresses == nullptr) {
                continue;
            }
            if (!(d->flags & PCAP_IF_UP) || (d->flags & PCAP_IF_LOOPBACK)) {
                continue;
            }
            Adaptor apt;
            apt.name = d->name;
            apt.desc = d->description;
            if (d->addresses->addr->sa_family == AF_INET) {
                apt.ip = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr)->sin_addr;
            }
        }
    });

    return adapters;
}

}  // namespace net