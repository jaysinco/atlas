#include "ethernet.h"

std::map<uint16_t, std::string> ethernet::type_dict = {
    {0x0800, Protocol_Type_IPv4},
    {0x86dd, Protocol_Type_IPv6},
    {0x0806, Protocol_Type_ARP},
    {0x8035, Protocol_Type_RARP},
};

ethernet::ethernet(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    d = ntoh(*reinterpret_cast<detail const*>(start));
    end = start + sizeof(detail);
}

ethernet::ethernet(mac const& smac, mac const& dmac, std::string const& type)
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
        throw std::runtime_error("unknow ethernet type: {}"_format(type));
    }
    d.dmac = dmac;
    d.smac = smac;
}

void ethernet::to_bytes(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(detail));
}

json ethernet::to_json() const
{
    json j;
    j["type"] = type();
    j["ethernet-type"] = succ_type();
    j["source-mac"] = d.smac.to_str();
    j["dest-mac"] = d.dmac.to_str();
    return j;
}

std::string ethernet::type() const { return Protocol_Type_Ethernet; }

std::string ethernet::succ_type() const
{
    if (type_dict.count(d.type) != 0) {
        return type_dict[d.type];
    }
    return Protocol_Type_Unknow(d.type);
}

bool ethernet::link_to(protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<ethernet const&>(rhs);
        return p.d.dmac == mac::broadcast || d.smac == p.d.dmac;
    }
    return false;
}

ethernet::detail const& ethernet::get_detail() const { return d; }

ethernet::detail ethernet::ntoh(detail const& d, bool reverse)
{
    detail dt = d;
    ntohx(dt.type, !reverse, s);
    return dt;
}

ethernet::detail ethernet::hton(detail const& d) { return ntoh(d, true); }
