#include "icmp.h"

std::map<uint8_t, std::pair<std::string, std::map<uint8_t, std::string>>> icmp::type_dict = {
    {
        0,
        {"ping-reply", {}},
    },
    {
        3,
        {"error",
         {
             {0, "network unreachable"},
             {1, "host unreachable"},
             {2, "protocol unreachable"},
             {3, "port unreachable"},
             {4, "fragmentation needed but forbid-slice bit set"},
             {5, "source routing failed"},
             {6, "destination network unknown"},
             {7, "destination host unknown"},
             {8, "source host isolated (obsolete)"},
             {9, "destination network administratively prohibited"},
             {10, "destination host administratively prohibited"},
             {11, "network unreachable for TOS"},
             {12, "host unreachable for TOS"},
             {13, "communication administratively prohibited by filtering"},
             {14, "host precedence violation"},
             {15, "Pprecedence cutoff in effect"},
         }},
    },
    {
        4,
        {"error",
         {
             {0, "source quench"},
         }},
    },
    {
        5,
        {"error",
         {
             {0, "redirect for network"},
             {1, "redirect for host"},
             {2, "redirect for TOS and network"},
             {3, "redirect for TOS and host"},
         }},
    },
    {
        8,
        {"ping-ask", {}},
    },
    {
        9,
        {"router-notice", {}},
    },
    {
        10,
        {"router-request", {}},
    },
    {
        11,
        {"error",
         {
             {0, "ttl equals 0 during transit"},
             {1, "ttl equals 0 during reassembly"},
         }},
    },
    {
        12,
        {"error",
         {
             {0, "ip header bad (catch-all error)"},
             {1, "required options missing"},
         }},
    },
    {
        17,
        {"netmask-ask", {}},
    },
    {
        18,
        {"netmask-reply", {}},
    },
};

icmp::icmp(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    d = ntoh(*reinterpret_cast<detail const*>(start));
    auto ip = dynamic_cast<ipv4 const*>(prev);
    extra.raw = std::string(start + sizeof(detail), start + ip->payload_size());
    if (icmp_type() == "error") {
        uint8_t const* pend = end;
        extra.eip = ipv4(start + sizeof(detail), pend);
        std::memcpy(&extra.buf, pend, 8);
    }
}

icmp::icmp(std::string const& ping_echo)
{
    d.type = 8;
    d.code = 0;
    d.u.s.id = rand_ushort();
    d.u.s.sn = rand_ushort();
    extra.raw = ping_echo;
}

void icmp::to_bytes(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d);
    size_t tlen = sizeof(detail) + extra.raw.size();
    uint8_t* buf = new uint8_t[tlen];
    std::memcpy(buf, &dt, sizeof(detail));
    std::memcpy(buf + sizeof(detail), extra.raw.data(), extra.raw.size());
    dt.crc = calc_checksum(buf, tlen);
    const_cast<icmp&>(*this).d.crc = dt.crc;
    std::memcpy(buf, &dt, sizeof(detail));
    bytes.insert(bytes.cbegin(), buf, buf + tlen);
    delete[] buf;
}

json icmp::to_json() const
{
    json j;
    j["type"] = type();
    std::string tp = icmp_type();
    j["icmp-type"] = tp;
    if (type_dict.count(d.type) > 0) {
        auto& code_dict = type_dict.at(d.type).second;
        if (code_dict.count(d.code) > 0) {
            j["desc"] = code_dict.at(d.code);
        }
    }
    j["id"] = d.u.s.id;
    j["serial-no"] = d.u.s.sn;
    size_t tlen = sizeof(detail) + extra.raw.size();
    uint8_t* buf = new uint8_t[tlen];
    auto dt = hton(d);
    std::memcpy(buf, &dt, sizeof(detail));
    std::memcpy(buf + sizeof(detail), extra.raw.data(), extra.raw.size());
    j["checksum"] = calc_checksum(buf, tlen);
    delete[] buf;

    if (tp == "ping-reply" || tp == "ping-ask") {
        j["echo"] = extra.raw;
    }
    if (tp == "error") {
        json ep;
        ep["ipv4"] = extra.eip.to_json();
        auto error_type = extra.eip.succ_type();
        if (error_type == Protocol_Type_TCP || error_type == Protocol_Type_UDP) {
            ep["source-port"] = ntohs(*reinterpret_cast<uint16_t const*>(&extra.buf[0]));
            ep["dest-port"] =
                ntohs(*reinterpret_cast<uint16_t const*>(&extra.buf[0] + sizeof(uint16_t)));
        }
        j["error-header"] = ep;
    }
    return j;
}

std::string icmp::type() const { return Protocol_Type_ICMP; }

std::string icmp::succ_type() const { return Protocol_Type_Void; }

bool icmp::link_to(protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<icmp const&>(rhs);
        return d.u.s.id == p.d.u.s.id && d.u.s.sn == p.d.u.s.sn;
    }
    return false;
}

icmp::detail const& icmp::get_detail() const { return d; }

icmp::extra_detail const& icmp::get_extra() const { return extra; }

std::string icmp::icmp_type() const
{
    return type_dict.count(d.type) > 0 ? type_dict.at(d.type).first : Protocol_Type_Unknow(d.type);
}

icmp::detail icmp::ntoh(detail const& d, bool reverse)
{
    detail dt = d;
    ntohx(dt.u.s.id, !reverse, s);
    ntohx(dt.u.s.sn, !reverse, s);
    return dt;
}

icmp::detail icmp::hton(detail const& d) { return ntoh(d, true); }
