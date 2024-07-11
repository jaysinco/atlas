#include "packet.h"
#include "port-table.h"
#include "protocol/ethernet.h"
#include "protocol/arp.h"
#include "protocol/ipv4.h"
#include "protocol/icmp.h"
#include "protocol/udp.h"
#include "protocol/tcp.h"
#include "protocol/dns.h"
#include "protocol/http.h"

std::map<std::string, packet::decoder> packet::decoder_dict = {
    {Protocol_Type_Ethernet, packet::decode<::ethernet>},
    {Protocol_Type_ARP, packet::decode<::arp>},
    {Protocol_Type_RARP, packet::decode<::arp>},
    {Protocol_Type_IPv4, packet::decode<::ipv4>},
    {Protocol_Type_ICMP, packet::decode<::icmp>},
    {Protocol_Type_UDP, packet::decode<::udp>},
    {Protocol_Type_TCP, packet::decode<::tcp>},
    {Protocol_Type_DNS, packet::decode<::dns>},
    {Protocol_Type_HTTP, packet::decode<::http>},
};

packet::packet() { d.time = gettimeofday(); }

packet::packet(uint8_t const* const start, uint8_t const* const end, timeval const& tv)
{
    uint8_t const* pstart = start;
    std::string type = Protocol_Type_Ethernet;
    while (type != Protocol_Type_Void && pstart < end) {
        if (decoder_dict.count(type) <= 0) {
            VLOG_IF(3, protocol::is_specific(type))
                << "unimplemented protocol: {} -> {}"_format(d.layers.back()->type(), type);
            break;
        }
        uint8_t const* pend = end;
        std::shared_ptr<protocol> prot =
            decoder_dict.at(type)(pstart, pend, d.layers.size() > 0 ? &*d.layers.back() : nullptr);
        if (pend > end) {
            throw std::runtime_error("exceed data boundary after {}"_format(type));
        }
        d.layers.push_back(prot);
        pstart = pend;
        type = prot->succ_type();
    }
    d.time = tv;
    d.owner = get_owner();
}

timeval packet::gettimeofday()
{
    SYSTEMTIME st;
    GetLocalTime(&st);
    tm t = {0};
    t.tm_year = st.wYear - 1900;
    t.tm_mon = st.wMonth - 1;
    t.tm_mday = st.wDay;
    t.tm_hour = st.wHour;
    t.tm_min = st.wMinute;
    t.tm_sec = st.wSecond;
    t.tm_isdst = -1;
    time_t clock = mktime(&t);
    timeval tv;
    tv.tv_sec = clock;
    tv.tv_usec = st.wMilliseconds * 1000;
    return tv;
}

void packet::to_bytes(std::vector<uint8_t>& bytes) const
{
    for (auto it = d.layers.crbegin(); it != d.layers.crend(); ++it) {
        (*it)->to_bytes(bytes);
    }
}

static std::string tv2s(timeval const& tv)
{
    tm local;
    time_t timestamp = tv.tv_sec;
    localtime_s(&local, &timestamp);
    char timestr[16] = {0};
    strftime(timestr, sizeof(timestr), "%H:%M:%S", &local);
    return "{}.{:03d}"_format(timestr, tv.tv_usec / 1000);
}

json const& packet::to_json() const
{
    if (!j_cached) {
        json j;
        j["layers"] = json::array();
        for (auto it = d.layers.cbegin(); it != d.layers.cend(); ++it) {
            std::string type = (*it)->type();
            j[type] = (*it)->to_json();
            j["layers"].push_back(type);
        }
        j["time"] = tv2s(d.time);
        j["owner"] = d.owner;
        const_cast<packet&>(*this).j_cached = j;
    }
    return *j_cached;
}

bool packet::link_to(packet const& rhs) const
{
    if (d.layers.size() != rhs.d.layers.size()) {
        return false;
    }
    if (d.time.tv_sec > rhs.d.time.tv_sec) {
        return false;
    }
    if (rhs.d.layers.size() > 2 && rhs.d.layers.at(2)->type() == Protocol_Type_ICMP) {
        auto& ch = dynamic_cast<icmp const&>(*rhs.d.layers.at(2));
        if (ch.icmp_type() == "error" && d.layers.size() > 1 &&
            d.layers.at(1)->type() == Protocol_Type_IPv4) {
            auto& ih = dynamic_cast<ipv4 const&>(*d.layers.at(1));
            if (ch.get_extra().eip == ih) {
                return true;
            }
        }
    }
    for (int i = 0; i < d.layers.size(); ++i) {
        if (!d.layers.at(i)->link_to(*rhs.d.layers.at(i))) {
            return false;
        }
    }
    return true;
}

packet::detail const& packet::get_detail() const { return d; }

void packet::set_time(timeval const& tv) { d.time = tv; }

bool packet::is_error() const
{
    return std::find_if(d.layers.cbegin(), d.layers.cend(),
                        [](std::shared_ptr<protocol> const& pt) {
                            if (pt->type() == Protocol_Type_ICMP) {
                                auto& ch = dynamic_cast<const icmp&>(*pt);
                                return ch.icmp_type() == "error";
                            }
                            return false;
                        }) != d.layers.cend();
}

bool packet::has_type(std::string const& type) const
{
    return std::find_if(d.layers.cbegin(), d.layers.cend(),
                        [&](std::shared_ptr<protocol> const& pt) { return pt->type() == type; }) !=
           d.layers.cend();
}

std::string packet::get_owner() const
{
    auto output = [](std::string const& src, std::string const& dest) -> std::string {
        if (src == dest) {
            return src;
        }
        if (src.size() > 0 && dest.size() > 0) {
            return "{} > {}"_format(src, dest);
        }
        return std::max(src, dest);
    };
    if (has_type(Protocol_Type_UDP)) {
        auto const& id = dynamic_cast<ipv4 const&>(*d.layers[1]).get_detail();
        auto const& ud = dynamic_cast<udp const&>(*d.layers[2]).get_detail();
        return output(port_table::lookup(std::make_tuple("udp", id.sip, ud.sport)),
                      port_table::lookup(std::make_tuple("udp", id.dip, ud.dport)));
    } else if (has_type(Protocol_Type_TCP)) {
        auto const& id = dynamic_cast<ipv4 const&>(*d.layers[1]).get_detail();
        auto const& td = dynamic_cast<tcp const&>(*d.layers[2]).get_detail();
        return output(port_table::lookup(std::make_tuple("tcp", id.sip, td.sport)),
                      port_table::lookup(std::make_tuple("tcp", id.dip, td.dport)));
    }
    return "";
}

packet packet::arp(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip, bool reply,
                   bool reverse)
{
    packet p;
    p.d.layers.push_back(std::make_shared<ethernet>(
        smac, mac::broadcast, reverse ? Protocol_Type_RARP : Protocol_Type_ARP));
    p.d.layers.push_back(std::make_shared<::arp>(smac, sip, dmac, dip, reply, reverse));
    return p;
}

packet packet::ping(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip, uint8_t ttl,
                    std::string const& echo, bool forbid_slice)
{
    packet p;
    p.d.layers.push_back(std::make_shared<ethernet>(smac, dmac, Protocol_Type_IPv4));
    p.d.layers.push_back(std::make_shared<ipv4>(sip, dip, ttl, Protocol_Type_ICMP, forbid_slice));
    p.d.layers.push_back(std::make_shared<icmp>(echo));
    return p;
}
