#include "ipv4.h"

std::map<uint8_t, std::string> ipv4::type_dict = {
    {1, Protocol_Type_ICMP},
    {6, Protocol_Type_TCP},
    {17, Protocol_Type_UDP},
};

ipv4::ipv4(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    d = ntoh(*reinterpret_cast<detail const*>(start));
    if (d.tlen != end - start) {
        VLOG(3) << "abnormal ipv4 length: expected={}, got={}"_format(d.tlen, end - start);
    }
    end = start + 4 * (d.ver_hl & 0xf);
}

ipv4::ipv4(ip4 const& sip, ip4 const& dip, uint8_t ttl, std::string const& type, bool forbid_slice)
{
    bool found = false;
    for (auto it = type_dict.cbegin(); it != type_dict.cend(); ++it) {
        if (it->second == type) {
            found = true;
            d.type = it->first;
            break;
        }
    }
    if (!found) {
        MY_THROW("unknow ipv4 type: {}"_format(type));
    }
    d.ver_hl = (4 << 4) | (sizeof(detail) / 4);
    d.id = rand_ushort();
    d.flags_fo = forbid_slice ? 0x4000 : 0;
    d.ttl = ttl;
    d.sip = sip;
    d.dip = dip;
}

void ipv4::to_bytes(std::vector<uint8_t>& bytes) const
{
    auto pt = const_cast<ipv4*>(this);
    pt->d.ver_hl = (4 << 4) | (sizeof(detail) / 4);
    pt->d.tlen = sizeof(detail) + bytes.size();

    auto dt = hton(d);
    dt.crc = calc_checksum(&dt, sizeof(detail));
    pt->d.crc = dt.crc;
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(detail));
}

json ipv4::to_json() const
{
    json j;
    j["type"] = type();
    j["ipv4-type"] = succ_type();
    j["version"] = d.ver_hl >> 4;
    j["tos"] = d.tos;
    size_t header_size = 4 * (d.ver_hl & 0xf);
    j["header-size"] = header_size;
    int checksum = -1;
    if (header_size == sizeof(detail)) {
        auto dt = hton(d);
        checksum = calc_checksum(&dt, header_size);
    }
    j["header-checksum"] = checksum;
    j["total-size"] = d.tlen;
    j["id"] = d.id;
    j["more-fragment"] = d.flags_fo & 0x2000 ? true : false;
    j["forbid-slice"] = d.flags_fo & 0x4000 ? true : false;
    j["fragment-offset"] = (d.flags_fo & 0x1fff) * 8;
    j["ttl"] = static_cast<int>(d.ttl);
    j["source-ip"] = d.sip.to_str();
    j["dest-ip"] = d.dip.to_str();
    return j;
}

std::string ipv4::type() const { return Protocol_Type_IPv4; }

std::string ipv4::succ_type() const
{
    if (type_dict.count(d.type) != 0) {
        return type_dict[d.type];
    }
    return Protocol_Type_Unknow(d.type);
}

bool ipv4::link_to(protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<ipv4 const&>(rhs);
        return d.sip == p.d.dip;
    }
    return false;
}

ipv4::detail const& ipv4::get_detail() const { return d; }

uint16_t ipv4::payload_size() const { return d.tlen - 4 * (d.ver_hl & 0xf); }

ipv4::detail ipv4::ntoh(detail const& d, bool reverse)
{
    detail dt = d;
    ntohx(dt.tlen, !reverse, s);
    ntohx(dt.id, !reverse, s);
    ntohx(dt.flags_fo, !reverse, s);
    return dt;
}

ipv4::detail ipv4::hton(detail const& d) { return ntoh(d, true); }

bool ipv4::operator==(ipv4 const& rhs) const
{
    return d.ver_hl == rhs.d.ver_hl && d.id == rhs.d.id && d.flags_fo == rhs.d.flags_fo &&
           d.type == rhs.d.type && d.sip == rhs.d.sip && d.dip == rhs.d.dip;
}
