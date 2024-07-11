#include "ethernet.h"
#include "toolkit/logging.h"

namespace net
{

std::map<uint16_t, Protocol::Type> Ethernet::type_dict = {
    {0x0800, kIPv4},
    {0x86dd, kIPv6},
    {0x0806, kARP},
    {0x8035, kRARP},
};

Ethernet::Ethernet(Mac const& smac, Mac const& dmac, Type type)
{
    bool found = false;
    for (auto it: type_dict) {
        if (it.second == type) {
            found = true;
            d_.type = it.first;
            break;
        }
    }
    if (!found) {
        MY_THROW("unknown ethernet type: {}", type);
    }
    d_.dmac = dmac;
    d_.smac = smac;
}

MyErrCode Ethernet::encode(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d_);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(Detail));
    return MyErrCode::kOk;
}

MyErrCode Ethernet::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    end = start + sizeof(Detail);
    return MyErrCode::kOk;
}

Variant Ethernet::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["ethernet-type"] = TOSTR(succType());
    j["source-mac"] = d_.smac.toStr();
    j["dest-mac"] = d_.dmac.toStr();
    return j;
}

Protocol::Type Ethernet::type() const { return kEthernet; }

Protocol::Type Ethernet::succType() const
{
    if (type_dict.count(d_.type) != 0) {
        return type_dict[d_.type];
    }
    return kUnknown;
}

bool Ethernet::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Ethernet const&>(rhs);
        return p.d_.dmac == Mac::kBroadcast || d_.smac == p.d_.dmac;
    }
    return false;
}

Ethernet::Detail const& Ethernet::getDetail() const { return d_; }

Ethernet::Detail Ethernet::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.type, !reverse, s);
    return dt;
}

Ethernet::Detail Ethernet::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net