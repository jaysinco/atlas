#pragma once
#include "packet.h"
#include "adaptor.h"
#include "protocol/dns.h"
#include <functional>

namespace net
{

class Transport
{
public:
    static MyErrCode open(Adaptor const& apt, void*& handle, bool promisc = true,
                          int timeout_ms = 1000);
    static MyErrCode close(void* handle);
    static MyErrCode setFilter(void* handle, std::string const& filter, Ip4 const& mask);

    static MyErrCode recv(
        void* handle, std::function<bool(Packet const& p)> callback, int timeout_ms = -1,
        std::chrono::system_clock::time_point const& start_tm = std::chrono::system_clock::now());

    static MyErrCode send(void* handle, Packet const& pac);

    static MyErrCode request(void* handle, Packet const& req, Packet& reply, int timeout_ms = -1,
                             bool do_send = true);

    static MyErrCode ip2mac(void* handle, Ip4 const& ip, Mac& mac, bool use_cache = true,
                            int timeout_ms = 5000);

    static MyErrCode ping(void* handle, Adaptor const& apt, Ip4 const& ip, Packet& reply,
                          int64_t& cost_ms, int ttl = 128, std::string const& echo = "",
                          bool forbid_slice = false, int timeout_ms = 5000);

    static MyErrCode queryDns(Ip4 const& server, std::string const& domain, Dns& reply,
                              int timeout_ms = 5000);

    static MyErrCode calcMtu(void* handle, Adaptor const& apt, Ip4 const& ip, int& mtu,
                             int high_bound = 1500);
};

}  // namespace net