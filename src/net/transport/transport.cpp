#include "transport.h"
#include "protocol/arp.h"
#include "protocol/icmp.h"
#include <atomic>
#include <thread>

pcap_t* transport::open_adaptor(adaptor const& apt, int timeout_ms)
{
    pcap_t* handle;
    char* errbuf = new char[PCAP_ERRBUF_SIZE];
    if (!(handle = pcap_open(apt.name.c_str(), 65536, PCAP_OPENFLAG_PROMISCUOUS, timeout_ms, NULL,
                             errbuf))) {
        throw std::runtime_error("failed to open adapter: {}"_format(apt.name));
    }
    return handle;
}

void transport::setfilter(pcap_t* handle, std::string const& filter, ip4 const& mask)
{
    bpf_program fcode;
    if (pcap_compile(handle, &fcode, filter.c_str(), 1, static_cast<uint32_t>(mask)) < 0) {
        throw std::runtime_error(
            "failed to compile pcap filter: {}, please refer to "
            "https://nmap.org/npcap/guide/wpcap/pcap-filter.html"_format(filter));
    }
    if (pcap_setfilter(handle, &fcode) < 0) {
        throw std::runtime_error("failed to set pcap filter: {}"_format(filter));
    }
}

bool transport::recv(pcap_t* handle, std::function<bool(packet const& p)> callback, int timeout_ms,
                     std::chrono::system_clock::time_point const& start_tm)
{
    int res;
    pcap_pkthdr* info;
    uint8_t const* start;
    while ((res = pcap_next_ex(handle, &info, &start)) >= 0) {
        if (timeout_ms > 0) {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_tm);
            if (duration.count() >= timeout_ms) {
                return false;
            }
        }
        if (res == 0) {
            continue;  // timeout elapsed
        }
        if (callback(packet(start, start + info->len, info->ts))) {
            return true;
        }
    }
    if (res == -1) {
        throw std::runtime_error("failed to read packets: {}"_format(pcap_geterr(handle)));
    }
    throw std::runtime_error("stop reading packets due to unexpected error");
}

void transport::send(pcap_t* handle, packet const& pac)
{
    std::vector<uint8_t> bytes;
    pac.to_bytes(bytes);
    const_cast<packet&>(pac).set_time(packet::gettimeofday());
    if (pcap_sendpacket(handle, bytes.data(), bytes.size()) != 0) {
        throw std::runtime_error("failed to send packet: {}"_format(pcap_geterr(handle)));
    }
}

bool transport::request(pcap_t* handle, packet const& req, packet& reply, int timeout_ms,
                        bool do_send)
{
    auto start_tm = std::chrono::system_clock::now();
    if (do_send) {
        send(handle, req);
    }
    return recv(
        handle,
        [&](packet const& p) {
            if (req.link_to(p)) {
                reply = p;
                return true;
            }
            return false;
        },
        timeout_ms, start_tm);
}

bool transport::ip2mac(pcap_t* handle, ip4 const& ip, mac& mac_, bool use_cache, int timeout_ms)
{
    static std::map<ip4, std::pair<mac, std::chrono::system_clock::time_point>> cached;
    auto start_tm = std::chrono::system_clock::now();
    if (use_cache) {
        auto it = cached.find(ip);
        if (it != cached.end()) {
            if (start_tm - it->second.second < 30s) {
                VLOG(3) << "use cached mac for {}"_format(ip.to_str());
                mac_ = it->second.first;
                return true;
            } else {
                VLOG(3) << "cached mac for {} expired, send arp to update"_format(ip.to_str());
            }
        }
    }
    adaptor apt = adaptor::fit(ip);
    if (ip == apt.ip) {
        mac_ = adaptor::fit(ip).mac_;
        cached[ip] = std::make_pair(mac_, std::chrono::system_clock::now());
        return true;
    }
    std::atomic<bool> over = false;
    packet req = packet::arp(apt.mac_, apt.ip, mac::zeros, ip);
    std::thread send_loop([&] {
        while (!over) {
            send(handle, req);
            std::this_thread::sleep_for(500ms);
        }
    });
    packet reply;
    bool ok = transport::request(handle, req, reply, timeout_ms, false);
    if (ok) {
        protocol const& prot = *reply.get_detail().layers.back();
        mac_ = dynamic_cast<arp const&>(prot).get_detail().smac;
        cached[ip] = std::make_pair(mac_, std::chrono::system_clock::now());
    }
    over = true;
    send_loop.join();
    return ok;
}

static long operator-(timeval const& tv1, timeval const& tv2)
{
    long diff_sec = tv1.tv_sec - tv2.tv_sec;
    long diff_ms = (tv1.tv_usec - tv2.tv_usec) / 1000;
    return (diff_sec * 1000 + diff_ms);
}

bool transport::ping(pcap_t* handle, adaptor const& apt, ip4 const& ip, packet& reply,
                     long& cost_ms, int ttl, std::string const& echo, bool forbid_slice,
                     int timeout_ms)
{
    mac dmac;
    ip4 dip = apt.ip.is_local(ip, apt.mask) ? ip : apt.gateway;
    if (!ip2mac(handle, dip, dmac)) {
        VLOG(1) << "can't resolve mac address of {}"_format(dip.to_str());
        return false;
    }
    packet req = packet::ping(apt.mac_, apt.ip, dmac, ip, ttl, echo, forbid_slice);
    bool ok = transport::request(handle, req, reply, timeout_ms);
    if (ok) {
        cost_ms = reply.get_detail().time - req.get_detail().time;
    }
    return ok;
}

bool transport::query_dns(ip4 const& server, std::string const& domain, dns& reply, int timeout_ms)
{
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) {
        LOG(ERROR) << "failed to create udp socket";
        return false;
    }
    std::shared_ptr<void*> socket_guard(nullptr, [&](void*) {
        VLOG(1) << "socket closed";
        if (closesocket(s) == SOCKET_ERROR) {
            LOG(ERROR) << "failed to close socket";
        };
    });
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(53);
    addr.sin_addr = static_cast<in_addr>(server);
    dns query(domain);
    std::vector<uint8_t> packet;
    query.to_bytes(packet);
    if (sendto(s, reinterpret_cast<char const*>(packet.data()), static_cast<int>(packet.size()), 0,
               reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in)) == SOCKET_ERROR) {
        LOG(ERROR) << "failed to send dns data: {}"_format(WSAGetLastError());
        return false;
    }
    DWORD timeout = timeout_ms;
    if (setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<char const*>(&timeout),
                   sizeof(DWORD)) == SOCKET_ERROR) {
        LOG(ERROR) << "failed to set socket receive timeout: {}"_format(WSAGetLastError());
        return false;
    }
    char buf[1024] = {0};
    sockaddr_in from;
    int from_len = sizeof(sockaddr_in);
    int recv_len = recvfrom(s, buf, sizeof(buf), 0, reinterpret_cast<sockaddr*>(&from), &from_len);
    if (recv_len == SOCKET_ERROR) {
        LOG(ERROR) << "failed to receive dns data: {}"_format(WSAGetLastError());
        return false;
    }
    uint8_t const* start = reinterpret_cast<uint8_t*>(buf);
    uint8_t const* end = start + recv_len;
    reply = dns(start, end);
    return true;
}

int transport::calc_mtu(pcap_t* handle, adaptor const& apt, ip4 const& ip, int high_bound,
                        bool print_log)
{
    int const offset = sizeof(ipv4::detail) + sizeof(icmp::detail);
    int low = 0;
    int high = high_bound - offset;
    packet reply;
    long cost_ms;
    if (ping(handle, apt, ip, reply, cost_ms, 128, std::string(high, '*'), true)) {
        if (!reply.is_error()) {
            throw std::runtime_error(
                "even highest-bound={} can't generate ICMP error"_format(high_bound));
        }
    } else {
        throw std::runtime_error("failed to call ping routine");
    }
    while (low < high - 1) {
        int vtest = (high + low) / 2;
        int ret = ping(handle, apt, ip, reply, cost_ms, 128, std::string(vtest, '*'), true);
        if (!ret) {
            throw std::runtime_error("failed to call ping routine");
        }
        if (!reply.is_error()) {
            LOG_IF(INFO, print_log) << "- {:5d}"_format(vtest + offset);
            low = vtest;
        } else {
            auto& p = dynamic_cast<icmp const&>(*reply.get_detail().layers.back());
            if (p.get_detail().type == 3 && p.get_detail().code == 4) {
                LOG_IF(INFO, print_log) << "+ {:5d}"_format(vtest + offset);
                high = vtest;
            } else {
                throw std::runtime_error(
                    "get unexpected ICMP error: {}"_format(p.to_json().dump(3)));
            }
        }
    }
    return low + offset;
}
