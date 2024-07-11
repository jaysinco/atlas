#include "adaptor.h"
#include <iphlpapi.h>
#include <mutex>

json adaptor::to_json() const
{
    json j;
    j["name"] = name;
    j["desc"] = desc;
    j["mac"] = mac_.to_str();
    j["ip"] = ip.to_str();
    j["mask"] = mask.to_str();
    j["gateway"] = gateway.to_str();
    return j;
}

adaptor const& adaptor::fit(ip4 const& hint)
{
    auto it = std::find_if(all().begin(), all().end(), [&](adaptor const& apt) {
        return apt.mask != ip4::zeros && apt.gateway != ip4::zeros &&
               (hint != ip4::zeros ? apt.ip.is_local(hint, apt.mask) : true);
    });
    if (it == all().end()) {
        throw std::runtime_error("no local adapter match {}"_format(hint.to_str()));
    }
    return *it;
}

bool adaptor::is_native(ip4 const& ip)
{
    return std::find_if(all().begin(), all().end(),
                        [&](adaptor const& apt) { return ip == apt.ip; }) != all().end();
}

std::vector<adaptor> const& adaptor::all()
{
    static std::once_flag flag;
    static std::vector<adaptor> adapters;
    std::call_once(flag, [&] {
        uint64_t buflen = sizeof(IP_ADAPTER_INFO);
        auto plist = reinterpret_cast<IP_ADAPTER_INFO*>(malloc(sizeof(IP_ADAPTER_INFO)));
        std::shared_ptr<void> plist_guard(nullptr, [&](void*) { free(plist); });
        if (GetAdaptersInfo(plist, &buflen) == ERROR_BUFFER_OVERFLOW) {
            plist = reinterpret_cast<IP_ADAPTER_INFO*>(malloc(buflen));
            if (GetAdaptersInfo(plist, &buflen) != NO_ERROR) {
                throw std::runtime_error("failed to get adapters info");
            }
        }
        PIP_ADAPTER_INFO pinfo = plist;
        while (pinfo) {
            adaptor apt;
            ip4 ip(pinfo->IpAddressList.IpAddress.String);
            ip4 mask(pinfo->IpAddressList.IpMask.String);
            ip4 gateway(pinfo->GatewayList.IpAddress.String);
            if (ip != ip4::zeros) {
                apt.name = std::string("\\Device\\NPF_") + pinfo->AdapterName;
                apt.desc = pinfo->Description;
                apt.ip = ip;
                apt.mask = mask;
                apt.gateway = gateway;
                if (pinfo->AddressLength != sizeof(mac)) {
                    LOG(WARNING) << "wrong mac length: {}"_format(pinfo->AddressLength);
                } else {
                    auto c = reinterpret_cast<uint8_t*>(&apt.mac_);
                    for (unsigned i = 0; i < pinfo->AddressLength; ++i) {
                        c[i] = pinfo->Address[i];
                    }
                }
                adapters.push_back(apt);
            }
            pinfo = pinfo->Next;
        }
        if (adapters.size() <= 0) {
            throw std::runtime_error("failed to find any suitable adapter");
        }
    });
    return adapters;
}
