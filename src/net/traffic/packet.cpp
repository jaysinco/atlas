#include "packet.h"
#include "toolkit/logging.h"
#include "protocol/ethernet.h"
#include "protocol/arp.h"
#include "protocol/ipv4.h"
#include "protocol/icmp.h"
#include "protocol/udp.h"
#include "protocol/tcp.h"
#include "protocol/dns.h"
#include "protocol/http.h"
#include <algorithm>

namespace net
{

Packet::Packet() { d_.time = std::chrono::system_clock::now(); }

MyErrCode Packet::encode(std::vector<uint8_t>& bytes) const
{
    for (auto it = d_.layers.crbegin(); it != d_.layers.crend(); ++it) {
        CHECK_ERR_RET((*it)->encode(bytes));
    }
    return MyErrCode::kOk;
}

MyErrCode Packet::decode(uint8_t const* const start, uint8_t const* const end,
                         Protocol::Type start_type)
{
    if (!d_.layers.empty()) {
        ELOG("dirty protocol stack");
        return MyErrCode::kFailed;
    }
    Protocol::Type type = start_type;
    uint8_t const* pstart = start;
    uint8_t const* pend = end;
    while (type != Protocol::kEmpty && type != Protocol::kUnknown && pstart < end) {
        Protocol const* prev = d_.layers.empty() ? nullptr : (&*d_.layers.back());
        Protocol::Ptr pt;
        CHECK_ERR_RET(decodeLayer(type, pstart, pend, prev, pt));
        if (pend > end) {
            ELOG("exceed data boundary after {}", type);
            return MyErrCode::kFailed;
        }
        d_.layers.push_back(pt);
        type = pt->succType();
        pstart = pend;
        pend = end;
    }
    return MyErrCode::kOk;
}

Variant const& Packet::toVariant() const
{
    if (!j_cached_) {
        Variant j;
        Variant::Vec layers;
        for (auto& p: d_.layers) {
            layers.push_back(p->toVariant());
        }
        j["layers"] = layers;
        j["time"] = TOSTR(d_.time);
        const_cast<Packet&>(*this).j_cached_ = std::move(j);
    }
    return *j_cached_;
}

bool Packet::linkTo(Packet const& rhs) const
{
    if (d_.layers.size() != rhs.d_.layers.size()) {
        return false;
    }
    if (d_.time > rhs.d_.time) {
        return false;
    }
    if (rhs.d_.layers.size() > 2 && rhs.d_.layers.at(2)->type() == Protocol::kICMP) {
        auto& ch = dynamic_cast<Icmp const&>(*rhs.d_.layers.at(2));
        if (ch.icmpType() == "error" && d_.layers.size() > 1 &&
            d_.layers.at(1)->type() == Protocol::kIPv4) {
            auto& ih = dynamic_cast<Ipv4 const&>(*d_.layers.at(1));
            if (ch.getExtra().eip == ih) {
                return true;
            }
        }
    }
    for (int i = 0; i < d_.layers.size(); ++i) {
        if (!d_.layers.at(i)->linkTo(*rhs.d_.layers.at(i))) {
            return false;
        }
    }
    return true;
}

bool Packet::hasType(Protocol::Type type) const
{
    return std::find_if(d_.layers.cbegin(), d_.layers.cend(), [&](Protocol::Ptr const& pt) {
               return pt->type() == type;
           }) != d_.layers.cend();
}

bool Packet::hasIcmpError() const
{
    return std::find_if(d_.layers.cbegin(), d_.layers.cend(), [](Protocol::Ptr const& pt) {
               if (pt->type() == Protocol::kICMP) {
                   auto& ch = dynamic_cast<Icmp const&>(*pt);
                   return ch.icmpType() == "error";
               }
               return false;
           }) != d_.layers.cend();
}

void Packet::setLayers(Protocol::Stack const& st) { d_.layers = st; }

void Packet::setLayers(Protocol::Stack&& st) { d_.layers = std::move(st); }

void Packet::setTime(Time const& tm) { d_.time = tm; }

Packet::Detail const& Packet::getDetail() const { return d_; }

MyErrCode Packet::decodeLayer(Protocol::Type type, uint8_t const* const start, uint8_t const*& end,
                              Protocol const* prev, Protocol::Ptr& pt)
{
    switch (type) {
        case Protocol::kEthernet:
            pt = std::make_shared<Ethernet>();
            break;
        case Protocol::kIPv4:
            pt = std::make_shared<Ipv4>();
            break;
        case Protocol::kARP:
        case Protocol::kRARP:
            pt = std::make_shared<Arp>();
            break;
        case Protocol::kICMP:
            pt = std::make_shared<Icmp>();
            break;
        case Protocol::kTCP:
            pt = std::make_shared<Tcp>();
            break;
        case Protocol::kUDP:
            pt = std::make_shared<Udp>();
            break;
        case Protocol::kDNS:
            pt = std::make_shared<Dns>();
            break;
        case Protocol::kHTTP:
            pt = std::make_shared<Http>();
            break;
        case Protocol::kIPv6:
        case Protocol::kHTTPS:
        case Protocol::kSSH:
        case Protocol::kTELNET:
        case Protocol::kRDP:
        case Protocol::kEmpty:
        case Protocol::kUnknown:
        default:
            ELOG("decode '{}' not support", TOSTR(type));
            return MyErrCode::kFailed;
    }
    CHECK_ERR_RET(pt->decode(start, end, prev));
    return MyErrCode::kOk;
}

Packet Packet::arp(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip, bool reply,
                   bool reverse)
{
    Packet p;
    p.d_.layers.push_back(std::make_shared<Ethernet>(smac, Mac::kBroadcast,
                                                     reverse ? Protocol::kRARP : Protocol::kARP));
    p.d_.layers.push_back(std::make_shared<Arp>(smac, sip, dmac, dip, reply, reverse));
    return p;
}

Packet Packet::ping(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip, uint8_t ttl,
                    std::string const& echo, bool forbid_slice)
{
    Packet p;
    p.d_.layers.push_back(std::make_shared<Ethernet>(smac, dmac, Protocol::kIPv4));
    p.d_.layers.push_back(std::make_shared<Ipv4>(sip, dip, ttl, Protocol::kICMP, forbid_slice));
    p.d_.layers.push_back(std::make_shared<Icmp>(echo));
    return p;
}

}  // namespace net