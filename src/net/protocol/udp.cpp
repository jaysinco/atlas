#include "udp.h"
#include "ipv4.h"

namespace net
{

MyErrCode Udp::encode(std::vector<uint8_t>& bytes) const { return MyErrCode::kUnimplemented; }

MyErrCode Udp::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    end = start + sizeof(Detail);
    auto& ipdt = dynamic_cast<Ipv4 const*>(prev)->getDetail();
    PseudoHeader ph;
    ph.sip = ipdt.sip;
    ph.dip = ipdt.dip;
    ph.type = ipdt.type;
    ph.zero_pad = 0;
    ph.len = htons(d_.len);
    size_t tlen = sizeof(PseudoHeader) + d_.len;
    uint8_t* buf = new uint8_t[tlen];
    std::memcpy(buf, &ph, sizeof(PseudoHeader));
    std::memcpy(buf + sizeof(PseudoHeader), start, d_.len);
    extra_.crc = calcChecksum(buf, tlen);
    delete[] buf;
    return MyErrCode::kOk;
}

Variant Udp::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["udp-type"] = TOSTR(succType());
    j["source-port"] = d_.sport;
    j["dest-port"] = d_.dport;
    j["total-size"] = d_.len;
    j["checksum"] = extra_.crc;
    return j;
}

Protocol::Type Udp::type() const { return kUDP; }

Protocol::Type Udp::succType() const
{
    auto dtype = guessProtocolByPort(d_.dport, kUDP);
    if (dtype != kUnknown) {
        return dtype;
    }
    return guessProtocolByPort(d_.sport, kUDP);
}

bool Udp::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Udp const&>(rhs);
        return d_.sport == p.d_.dport && d_.dport == p.d_.sport;
    }
    return false;
}

Udp::Detail const& Udp::getDetail() const { return d_; }

Udp::ExtraDetail const& Udp::getExtra() const { return extra_; }

Udp::Detail Udp::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.sport, !reverse, s);
    ntohx(dt.dport, !reverse, s);
    ntohx(dt.len, !reverse, s);
    return dt;
}

Udp::Detail Udp::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net