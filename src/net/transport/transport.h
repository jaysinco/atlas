#pragma once
#include "packet.h"
#include "adaptor.h"
#include "protocol/dns.h"
#include <pcap.h>

class transport
{
public:
    static pcap_t *open_adaptor(const adaptor &apt, int timeout_ms = 1000);

    static void setfilter(pcap_t *handle, const std::string &filter, const ip4 &mask);

    static bool recv(
        pcap_t *handle, std::function<bool(const packet &p)> callback, int timeout_ms = -1,
        const std::chrono::system_clock::time_point &start_tm = std::chrono::system_clock::now());

    static void send(pcap_t *handle, const packet &pac);

    static bool request(pcap_t *handle, const packet &req, packet &reply, int timeout_ms = -1,
                        bool do_send = true);

    static bool ip2mac(pcap_t *handle, const ip4 &ip, mac &mac_, bool use_cache = true,
                       int timeout_ms = 5000);

    static bool ping(pcap_t *handle, const adaptor &apt, const ip4 &ip, packet &reply,
                     long &cost_ms, int ttl = 128, const std::string &echo = "",
                     bool forbid_slice = false, int timeout_ms = 5000);

    static bool query_dns(const ip4 &server, const std::string &domain, dns &reply,
                          int timeout_ms = 5000);

    static int calc_mtu(pcap_t *handle, const adaptor &apt, const ip4 &ip, int high_bound = 1500,
                        bool print_log = false);
};
