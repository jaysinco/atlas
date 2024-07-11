#include "arp.h"

arp::arp(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    d = ntoh(*reinterpret_cast<detail const*>(start));
    if (end != start + sizeof(detail)) {
        VLOG(3) << "abnormal arp length: expected={}, got={}"_format(sizeof(detail), end - start);
    }
}

arp::arp(mac const& smac, ip4 const& sip, mac const& dmac, ip4 const& dip, bool reply, bool reverse)
{
    d.hw_type = 1;
    d.prot_type = 0x0800;
    d.hw_len = 6;
    d.prot_len = 4;
    d.op = reverse ? (reply ? 4 : 3) : (reply ? 2 : 1);
    d.smac = smac;
    d.sip = sip;
    d.dmac = dmac;
    d.dip = dip;
}

void arp::to_bytes(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(detail));
}

json arp::to_json() const
{
    json j;
    j["type"] = type();
    j["hardware-type"] = d.hw_type;
    j["protocol-type"] = d.prot_type;
    j["hardware-addr-len"] = d.hw_len;
    j["protocol-addr-len"] = d.prot_len;
    j["operate"] = (d.op == 1 || d.op == 3)   ? "request"
                   : (d.op == 2 || d.op == 4) ? "reply"
                                              : Protocol_Type_Unknow(d.op);
    j["source-mac"] = d.smac.to_str();
    j["source-ip"] = d.sip.to_str();
    j["dest-mac"] = d.dmac.to_str();
    j["dest-ip"] = d.dip.to_str();
    return j;
}

std::string arp::type() const
{
    return (d.op == 1 || d.op == 2)   ? Protocol_Type_ARP
           : (d.op == 3 || d.op == 4) ? Protocol_Type_RARP
                                      : Protocol_Type_Unknow(d.op);
}

std::string arp::succ_type() const { return Protocol_Type_Void; }

bool arp::link_to(protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<arp const&>(rhs);
        return (d.op == 1 || d.op == 3) && (p.d.op == 2 || p.d.op == 4) && (d.dip == p.d.sip);
    }
    return false;
}

arp::detail const& arp::get_detail() const { return d; }

arp::detail arp::ntoh(detail const& d, bool reverse)
{
    detail dt = d;
    ntohx(dt.hw_type, !reverse, s);
    ntohx(dt.prot_type, !reverse, s);
    ntohx(dt.op, !reverse, s);
    return dt;
}

arp::detail arp::hton(detail const& d) { return ntoh(d, true); }
