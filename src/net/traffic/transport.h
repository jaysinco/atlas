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
    static void* openAdaptor(Adaptor const& apt, int timeout_ms = 1000);

    static void setFilter(void* handle, std::string const& filter, Ip4 const& mask);

    static bool recv(
        void* handle, std::function<bool(Packet const& p)> callback, int timeout_ms = -1,
        std::chrono::system_clock::time_point const& start_tm = std::chrono::system_clock::now());

    static void send(void* handle, Packet const& pac);

    static bool request(void* handle, Packet const& req, Packet& reply, int timeout_ms = -1,
                        bool do_send = true);

    static bool ip2mac(void* handle, Ip4 const& ip, Mac& mac, bool use_cache = true,
                       int timeout_ms = 5000);

    static bool ping(void* handle, Adaptor const& apt, Ip4 const& ip, Packet& reply,
                     int64_t& cost_ms, int ttl = 128, std::string const& echo = "",
                     bool forbid_slice = false, int timeout_ms = 5000);

    static bool queryDns(Ip4 const& server, std::string const& domain, Dns& reply,
                         int timeout_ms = 5000);

    static int calcMtu(void* handle, Adaptor const& apt, Ip4 const& ip, int high_bound = 1500,
                       bool print_log = false);
};

}  // namespace net