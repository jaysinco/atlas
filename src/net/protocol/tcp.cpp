#include "tcp.h"
#include "ipv4.h"
#include "toolkit/logging.h"
#include <boost/algorithm/string/join.hpp>

namespace net
{

MyErrCode Tcp::encode(std::vector<uint8_t>& bytes) const { return MyErrCode::kUnimplemented; }

MyErrCode Tcp::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    end = start + 4 * (d_.hl_flags >> 12 & 0xf);
    auto& ipdt = dynamic_cast<Ipv4 const*>(prev)->getDetail();
    extra_.len = ipdt.tlen - 4 * (ipdt.ver_hl & 0xf);
    PseudoHeader ph;
    ph.sip = ipdt.sip;
    ph.dip = ipdt.dip;
    ph.type = ipdt.type;
    ph.zero_pad = 0;
    ph.len = htons(extra_.len);
    size_t tlen = sizeof(PseudoHeader) + extra_.len;
    uint8_t* buf = new uint8_t[tlen];
    std::memcpy(buf, &ph, sizeof(PseudoHeader));
    std::memcpy(buf + sizeof(PseudoHeader), start, extra_.len);
    extra_.crc = calcChecksum(buf, tlen);
    delete[] buf;
    return MyErrCode::kOk;
}

Variant Tcp::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["tcp-type"] = succType();
    j["source-port"] = d_.sport;
    j["dest-port"] = d_.dport;
    size_t header_size = 4 * (d_.hl_flags >> 12 & 0xf);
    j["header-size"] = header_size;
    j["total-size"] = extra_.len;
    j["sequence-no"] = d_.sn;
    j["acknowledge-no"] = d_.an;
    std::vector<std::string> flags;
    if (d_.hl_flags >> 8 & 0x1) {
        flags.push_back("ns");
    }
    if (d_.hl_flags >> 7 & 0x1) {
        flags.push_back("cwr");
    }
    if (d_.hl_flags >> 6 & 0x1) {
        flags.push_back("ece");
    }
    if (d_.hl_flags >> 5 & 0x1) {
        flags.push_back("urg");
    }
    if (d_.hl_flags >> 4 & 0x1) {
        flags.push_back("ack");
    }
    if (d_.hl_flags >> 3 & 0x1) {
        flags.push_back("psh");
    }
    if (d_.hl_flags >> 2 & 0x1) {
        flags.push_back("rst");
    }
    if (d_.hl_flags >> 1 & 0x1) {
        flags.push_back("syn");
    }
    if (d_.hl_flags & 0x1) {
        flags.push_back("fin");
    }
    j["flags"] = boost::algorithm::join(flags, ";");
    j["window-size"] = d_.wlen;
    j["checksum"] = extra_.crc;
    j["urgent-pointer"] = d_.urp;
    return j;
}

Protocol::Type Tcp::type() const { return kTCP; }

Protocol::Type Tcp::succType() const
{
    auto dtype = guessProtocolByPort(d_.dport, kTCP);
    if (dtype != kUnknown) {
        return dtype;
    }
    return guessProtocolByPort(d_.sport, kTCP);
}

bool Tcp::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Tcp const&>(rhs);
        return d_.sport == p.d_.dport && d_.dport == p.d_.sport;
    }
    return false;
}

Tcp::Detail const& Tcp::getDetail() const { return d_; }

Tcp::Detail Tcp::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.sport, !reverse, s);
    ntohx(dt.dport, !reverse, s);
    ntohx(dt.sn, !reverse, l);
    ntohx(dt.an, !reverse, l);
    ntohx(dt.hl_flags, !reverse, s);
    ntohx(dt.wlen, !reverse, s);
    ntohx(dt.urp, !reverse, s);
    return dt;
}

Tcp::Detail Tcp::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net