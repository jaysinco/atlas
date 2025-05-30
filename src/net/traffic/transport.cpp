#include "transport.h"
#include "protocol/arp.h"
#include "protocol/icmp.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <pcap.h>
#include <unistd.h>
#include <atomic>
#include <thread>

namespace net
{

MyErrCode Transport::open(Adaptor const& apt, void*& handle, bool promisc, int timeout_ms)
{
    pcap_t* hndl;
    char* errbuf = new char[PCAP_ERRBUF_SIZE];
    auto errbuf_guard = toolkit::scopeExit([&] { delete[] errbuf; });
    if (!(hndl = pcap_open_live(apt.name.c_str(), 65536, promisc, timeout_ms, errbuf))) {
        ELOG("failed to open adapter {}: {}", apt.name, errbuf);
        return MyErrCode::kFailed;
    }
    if (pcap_datalink(hndl) != DLT_EN10MB) {
        ELOG("link layer header is not ethernet");
        return MyErrCode::kFailed;
    }
    handle = hndl;
    return MyErrCode::kOk;
}

MyErrCode Transport::close(void* handle)
{
    auto hndl = reinterpret_cast<pcap_t*>(handle);
    pcap_close(hndl);
    return MyErrCode::kOk;
}

MyErrCode Transport::setFilter(void* handle, std::string const& filter, Ip4 const& mask)
{
    auto hndl = reinterpret_cast<pcap_t*>(handle);
    bpf_program fcode;
    if (pcap_compile(hndl, &fcode, filter.c_str(), 1, static_cast<uint32_t>(mask)) < 0) {
        ELOG(
            "failed to compile pcap filter: {}, please refer to "
            "https://nmap.org/npcap/guide/wpcap/pcap-filter.html",
            filter);
        return MyErrCode::kFailed;
    }
    if (pcap_setfilter(hndl, &fcode) < 0) {
        ELOG("failed to set pcap filter: {}", filter);
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode Transport::recv(void* handle, std::function<bool(Packet&& p)> callback, int timeout_ms,
                          std::chrono::system_clock::time_point const& start_tm)
{
    auto hndl = reinterpret_cast<pcap_t*>(handle);
    int res;
    pcap_pkthdr* info;
    uint8_t const* start;
    while ((res = pcap_next_ex(hndl, &info, &start)) >= 0) {
        if (timeout_ms > 0) {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_tm);
            if (duration.count() >= timeout_ms) {
                DLOG("packet receive timeout: {}ms >= {}ms", duration.count(), timeout_ms);
                return MyErrCode::kTimeout;
            }
        }
        if (res == 0) {
            continue;  // timeout elapsed
        }
        Packet pac;
        CHECK_ERR_RET(pac.decode(start, start + info->len, Protocol::kEthernet));
        pac.setTime(Packet::Time{std::chrono::seconds{info->ts.tv_sec} +
                                 std::chrono::microseconds{info->ts.tv_usec}});
        if (callback(std::move(pac))) {
            return MyErrCode::kOk;
        }
    }
    if (res == -1) {
        ELOG("failed to read packets: {}", pcap_geterr(hndl));
        return MyErrCode::kFailed;
    }
    ELOG("stop reading packets due to unexpected error: {}", res);
    return MyErrCode::kFailed;
}

MyErrCode Transport::send(void* handle, Packet const& pac)
{
    if (pac.getDetail().layers.empty()) {
        return MyErrCode::kOk;
    }
    auto hndl = reinterpret_cast<pcap_t*>(handle);
    std::vector<uint8_t> bytes;
    CHECK_ERR_RET(pac.encode(bytes));
    const_cast<Packet&>(pac).setTime(Packet::Clock::now());
    if (pcap_sendpacket(hndl, bytes.data(), bytes.size()) != 0) {
        ELOG("failed to send packet: {}", pcap_geterr(hndl));
        return MyErrCode::kFailed;
    }
    return MyErrCode::kOk;
}

MyErrCode Transport::request(void* handle, Packet const& req, Packet& reply, int timeout_ms,
                             bool do_send)
{
    auto start_tm = std::chrono::system_clock::now();
    if (do_send) {
        CHECK_ERR_RET(send(handle, req));
    }
    return recv(
        handle,
        [&](Packet&& p) {
            if (req.linkTo(p)) {
                reply = std::move(p);
                return true;
            }
            return false;
        },
        timeout_ms, start_tm);
}

MyErrCode Transport::ip2mac(void* handle, Ip4 const& ip, Mac& mac, bool use_cache, int timeout_ms)
{
    static std::map<Ip4, std::pair<Mac, std::chrono::system_clock::time_point>> cached;
    auto start_tm = std::chrono::system_clock::now();
    if (use_cache) {
        auto it = cached.find(ip);
        if (it != cached.end()) {
            if (start_tm - it->second.second < std::chrono::seconds(30)) {
                DLOG("use cached mac for {}", ip);
                mac = it->second.first;
                return MyErrCode::kOk;
            } else {
                DLOG("cached mac for {} expired, send arp to update", ip);
            }
        }
    }
    Adaptor apt = Adaptor::fit(ip);
    if (ip == apt.ip) {
        mac = apt.mac;
        cached[ip] = std::make_pair(mac, std::chrono::system_clock::now());
        return MyErrCode::kOk;
    }
    std::atomic<bool> over = false;
    Packet req = Packet::arp(apt.mac, apt.ip, Mac::kZeros, ip);
    std::thread send_loop([&] {
        while (!over) {
            send(handle, req);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    Packet reply;
    auto ok = Transport::request(handle, req, reply, timeout_ms, false);
    if (ok == MyErrCode::kOk) {
        Protocol const& prot = *reply.getDetail().layers.back();
        mac = dynamic_cast<Arp const&>(prot).getDetail().smac;
        cached[ip] = std::make_pair(mac, std::chrono::system_clock::now());
    }
    over = true;
    send_loop.join();
    return ok;
}

MyErrCode Transport::mac2ip(Mac const& mac, Ip4& ip)
{
    std::string ipstr;
    CHECK_ERR_RET(toolkit::execsh(
        FSTR("arp-scan --localnet | grep {} | awk '{{print $1}}'", mac.toStr(true)), ipstr));
    if (ipstr.empty()) {
        ELOG("failed to find ip for {}", mac);
        return MyErrCode::kFailed;
    }
    ip = Ip4(ipstr.substr(0, ipstr.size() - 1));
    return MyErrCode::kOk;
}

MyErrCode Transport::ping(void* handle, Adaptor const& apt, Ip4 const& ip, Packet& reply,
                          int64_t& cost_ms, int ttl, std::string const& echo, bool forbid_slice,
                          int timeout_ms)
{
    Ip4 dip = apt.ip.isLocal(ip, apt.mask) ? ip : apt.gateway;
    Mac dmac;
    CHECK_ERR_RET(ip2mac(handle, dip, dmac));
    Packet req = Packet::ping(apt.mac, apt.ip, dmac, ip, ttl, echo, forbid_slice);
    auto ok = Transport::request(handle, req, reply, timeout_ms);
    if (ok == MyErrCode::kOk) {
        cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(reply.getDetail().time -
                                                                        req.getDetail().time)
                      .count();
    }
    return ok;
}

MyErrCode Transport::queryDns(Ip4 const& server, std::string const& domain, Dns& reply,
                              int timeout_ms)
{
    int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) {
        ELOG("failed to create udp socket");
        return MyErrCode::kFailed;
    }
    auto socket_guard = toolkit::scopeExit([&] {
        DLOG("socket closed");
        if (shutdown(s, SHUT_RDWR) != 0) {
            DLOG("failed to shutdown socket: {}", strerror(errno));
        };
        if (::close(s) != 0) {
            ELOG("failed to close socket: {}", strerror(errno));
        };
    });
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(53);
    addr.sin_addr = static_cast<in_addr>(server);
    Dns query(domain);
    std::vector<uint8_t> packet;
    CHECK_ERR_RET(query.encode(packet));
    if (::sendto(s, reinterpret_cast<char const*>(packet.data()), static_cast<int>(packet.size()),
                 0, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in)) == -1) {
        ELOG("failed to send dns data: {}", strerror(errno));
        return MyErrCode::kFailed;
    }
    int64_t timeout_us = timeout_ms * 1000;
    struct timeval timeout;
    timeout.tv_sec = timeout_us / 1'000'000;
    timeout.tv_usec = timeout_us % 1'000'000;
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0) {
        ELOG("failed to set socket receive timeout to {}ms: {}", timeout_ms, strerror(errno));
        return MyErrCode::kFailed;
    }
    char buf[1024] = {0};
    sockaddr_in from;
    socklen_t from_len = sizeof(sockaddr_in);
    int recv_len =
        ::recvfrom(s, buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
    if (recv_len == -1) {
        ELOG("failed to receive dns data: {}", strerror(errno));
        return MyErrCode::kFailed;
    }
    uint8_t const* start = reinterpret_cast<uint8_t*>(buf);
    uint8_t const* end = start + recv_len;
    CHECK_ERR_RET(reply.decode(start, end, nullptr));
    return MyErrCode::kOk;
}

MyErrCode Transport::calcMtu(void* handle, Adaptor const& apt, Ip4 const& ip, int& mtu,
                             int high_bound)
{
    int const offset = sizeof(Ipv4::Detail) + sizeof(Icmp::Detail);
    int low = 0;
    int high = high_bound - offset;
    Packet reply;
    int64_t cost_ms;
    CHECK_ERR_RET(ping(handle, apt, ip, reply, cost_ms, 128, std::string(high, '*'), true));
    if (!reply.hasIcmpError()) {
        ELOG("even highest-bound={} can't generate ICMP error", high_bound);
        return MyErrCode::kFailed;
    }

    while (low < high - 1) {
        int vtest = (high + low) / 2;
        CHECK_ERR_RET(ping(handle, apt, ip, reply, cost_ms, 128, std::string(vtest, '*'), true));
        if (!reply.hasIcmpError()) {
            DLOG("- {:5d}", vtest + offset);
            low = vtest;
        } else {
            auto& p = dynamic_cast<Icmp const&>(*reply.getDetail().layers.back());
            if (p.getDetail().type == 3 && p.getDetail().code == 4) {
                DLOG("+ {:5d}", vtest + offset);
                high = vtest;
            } else {
                ELOG("get unexpected ICMP error: {}", p.toVariant().toJsonStr());
                return MyErrCode::kFailed;
            }
        }
    }
    mtu = low + offset;
    return MyErrCode::kOk;
}

}  // namespace net