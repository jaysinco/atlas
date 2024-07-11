#include "adaptor.h"
#include <algorithm>
#include <mutex>
#include "toolkit/logging.h"
#include <sys/types.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <pcap.h>
#include <sys/ioctl.h>
#include <unistd.h>

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

MyErrCode Adaptor::getMacAddr(std::string const& if_name, Mac& mac)
{
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock == -1) {
        ELOG("failed to open socket: {}", strerror(errno));
        return MyErrCode::kFailed;
    }
    auto sock_guard = toolkit::scopeExit([&] { close(sock); });
    ifreq ifr;
    strncpy(ifr.ifr_name, if_name.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock, SIOCGIFHWADDR, &ifr) != 0) {
        ELOG("failed to ioctl SIOCGIFHWADDR: {}", strerror(errno));
        return MyErrCode::kFailed;
    }
    memcpy(reinterpret_cast<uint8_t*>(&mac), ifr.ifr_hwaddr.sa_data, 6);
    return MyErrCode::kOk;
}

MyErrCode Adaptor::getGateway(std::string const& if_name, Ip4& ip)
{
    char iface[IF_NAMESIZE]{0};
    char buf[4096]{0};

    FILE* fp = fopen("/proc/net/route", "r");
    if (!fp) {
        ELOG("failed to open /proc/net/route: {}", strerror(errno));
        return MyErrCode::kFailed;
    }
    auto fp_guard = toolkit::scopeExit([&] { fclose(fp); });

    int64_t destination, gateway;
    while (fgets(buf, sizeof(buf), fp)) {
        if (sscanf(buf, "%s %lx %lx", iface, &destination, &gateway) == 3) {
            if (destination == 0 && iface == if_name) {
                *reinterpret_cast<uint32_t*>(&ip) = gateway;
                return MyErrCode::kOk;
            }
        }
    }

    ELOG("failed to find gateway for {}", if_name);
    return MyErrCode::kFailed;
}

std::vector<Adaptor> const& Adaptor::all()
{
    static std::once_flag flag;
    static std::vector<Adaptor> adapters;

    std::call_once(flag, [&] {
        pcap_if_t* alldevs;
        char errbuf[PCAP_ERRBUF_SIZE] = {0};
        if (pcap_findalldevs(&alldevs, errbuf) != 0) {
            MY_THROW("failed to find all devs: {}", errbuf);
        }
        auto alldevs_guard = toolkit::scopeExit([&] { pcap_freealldevs(alldevs); });
        for (pcap_if_t* d = alldevs; d != nullptr; d = d->next) {
            if (!(d->flags & PCAP_IF_UP) || (d->flags & PCAP_IF_LOOPBACK)) {
                continue;
            }
            for (pcap_addr* a = d->addresses; a != nullptr; a = a->next) {
                if (a->addr->sa_family == AF_INET) {
                    Adaptor apt{};
                    apt.name = d->name;
                    if (d->description) {
                        apt.desc = d->description;
                    }
                    apt.ip = reinterpret_cast<sockaddr_in*>(a->addr)->sin_addr;
                    if (a->netmask && a->netmask->sa_family == AF_INET) {
                        apt.mask = reinterpret_cast<sockaddr_in*>(a->netmask)->sin_addr;
                    }
                    getMacAddr(apt.name, apt.mac);
                    getGateway(apt.name, apt.gateway);
                    adapters.push_back(std::move(apt));
                    break;
                }
            }
        }
    });

    return adapters;
}

}  // namespace net