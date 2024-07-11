#include "arp.h"
#include "toolkit/logging.h"

namespace net
{

Arp::Arp(Mac const& smac, Ip4 const& sip, Mac const& dmac, Ip4 const& dip, bool reply, bool reverse)
{
    d_.hw_type = 1;
    d_.prot_type = 0x0800;
    d_.hw_len = 6;
    d_.prot_len = 4;
    d_.op = reverse ? (reply ? 4 : 3) : (reply ? 2 : 1);
    d_.smac = smac;
    d_.sip = sip;
    d_.dmac = dmac;
    d_.dip = dip;
}

MyErrCode Arp::encode(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d_);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(Detail));
    return MyErrCode::kOk;
}

MyErrCode Arp::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    if (end != start + sizeof(Detail)) {
        ELOG("abnormal arp length: expected={}, got={}", sizeof(Detail), end - start);
        return MyErrCode::kFailed;
    }
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    return MyErrCode::kOk;
}

Variant Arp::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["hardware-type"] = d_.hw_type;
    j["protocol-type"] = d_.prot_type;
    j["hardware-addr-len"] = d_.hw_len;
    j["protocol-addr-len"] = d_.prot_len;
    j["operate"] = (d_.op == 1 || d_.op == 3)   ? "request"
                   : (d_.op == 2 || d_.op == 4) ? "reply"
                                                : FSTR("unknown({})", d_.op);
    j["source-mac"] = d_.smac.toStr();
    j["source-ip"] = d_.sip.toStr();
    j["dest-mac"] = d_.dmac.toStr();
    j["dest-ip"] = d_.dip.toStr();
    return j;
}

Protocol::Type Arp::type() const
{
    return (d_.op == 1 || d_.op == 2) ? kARP : (d_.op == 3 || d_.op == 4) ? kRARP : kUnknow;
}

Protocol::Type Arp::succType() const { return kEmpty; }

bool Arp::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Arp const&>(rhs);
        return (d_.op == 1 || d_.op == 3) && (p.d_.op == 2 || p.d_.op == 4) && (d_.dip == p.d_.sip);
    }
    return false;
}

Arp::Detail const& Arp::getDetail() const { return d_; }

Arp::Detail Arp::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.hw_type, !reverse, s);
    ntohx(dt.prot_type, !reverse, s);
    ntohx(dt.op, !reverse, s);
    return dt;
}

Arp::Detail Arp::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net