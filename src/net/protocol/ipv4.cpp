#include "ipv4.h"
#include "toolkit/logging.h"

namespace net
{

std::map<uint8_t, Protocol::Type> Ipv4::type_dict = {
    {1, kICMP},
    {6, kTCP},
    {17, kUDP},
};

Ipv4::Ipv4(Ip4 const& sip, Ip4 const& dip, uint8_t ttl, Type type, bool forbid_slice)
{
    bool found = false;
    for (auto it = type_dict.cbegin(); it != type_dict.cend(); ++it) {
        if (it->second == type) {
            found = true;
            d_.type = it->first;
            break;
        }
    }
    if (!found) {
        MY_THROW("unknown ipv4 type: {}", type);
    }
    d_.ver_hl = (4 << 4) | (sizeof(Detail) / 4);
    d_.id = randUint16();
    d_.flags_fo = forbid_slice ? 0x4000 : 0;
    d_.ttl = ttl;
    d_.sip = sip;
    d_.dip = dip;
}

MyErrCode Ipv4::encode(std::vector<uint8_t>& bytes) const
{
    auto pt = const_cast<Ipv4*>(this);
    pt->d_.ver_hl = (4 << 4) | (sizeof(Detail) / 4);
    pt->d_.tlen = sizeof(Detail) + bytes.size();

    auto dt = hton(d_);
    dt.crc = calcChecksum(&dt, sizeof(Detail));
    pt->d_.crc = dt.crc;
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(Detail));
    return MyErrCode::kOk;
}

MyErrCode Ipv4::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    if (d_.tlen != end - start) {
        ELOG("abnormal ipv4 length: expected={}, got={}", d_.tlen, end - start);
        return MyErrCode::kFailed;
    }
    end = start + 4 * (d_.ver_hl & 0xf);
    return MyErrCode::kOk;
}

Variant Ipv4::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["ipv4-type"] = TOSTR(succType());
    j["version"] = d_.ver_hl >> 4;
    j["tos"] = d_.tos;
    size_t header_size = 4 * (d_.ver_hl & 0xf);
    j["header-size"] = header_size;
    int checksum = -1;
    if (header_size == sizeof(Detail)) {
        auto dt = hton(d_);
        checksum = calcChecksum(&dt, header_size);
    }
    j["header-checksum"] = checksum;
    j["total-size"] = d_.tlen;
    j["id"] = d_.id;
    j["more-fragment"] = d_.flags_fo & 0x2000 ? true : false;
    j["forbid-slice"] = d_.flags_fo & 0x4000 ? true : false;
    j["fragment-offset"] = (d_.flags_fo & 0x1fff) * 8;
    j["ttl"] = static_cast<int>(d_.ttl);
    j["source-ip"] = d_.sip.toStr();
    j["dest-ip"] = d_.dip.toStr();
    return j;
}

Protocol::Type Ipv4::type() const { return kIPv4; }

Protocol::Type Ipv4::succType() const
{
    if (type_dict.count(d_.type) != 0) {
        return type_dict[d_.type];
    }
    return kUnknown;
}

bool Ipv4::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Ipv4 const&>(rhs);
        return d_.sip == p.d_.dip;
    }
    return false;
}

Ipv4::Detail const& Ipv4::getDetail() const { return d_; }

uint16_t Ipv4::payloadSize() const { return d_.tlen - 4 * (d_.ver_hl & 0xf); }

Ipv4::Detail Ipv4::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.tlen, !reverse, s);
    ntohx(dt.id, !reverse, s);
    ntohx(dt.flags_fo, !reverse, s);
    return dt;
}

Ipv4::Detail Ipv4::hton(Detail const& d) { return ntoh(d, true); }

bool Ipv4::operator==(Ipv4 const& rhs) const
{
    return d_.ver_hl == rhs.d_.ver_hl && d_.id == rhs.d_.id && d_.flags_fo == rhs.d_.flags_fo &&
           d_.type == rhs.d_.type && d_.sip == rhs.d_.sip && d_.dip == rhs.d_.dip;
}

}  // namespace net
