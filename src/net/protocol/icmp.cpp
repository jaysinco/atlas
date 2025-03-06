#include "icmp.h"

namespace net
{

std::map<uint8_t, std::pair<std::string, std::map<uint8_t, std::string>>> Icmp::type_dict = {
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

Icmp::Icmp(std::string const& ping_echo)
{
    d_.type = 8;
    d_.code = 0;
    d_.u.s.id = randUint16();
    d_.u.s.sn = randUint16();
    extra_.raw = ping_echo;
}

MyErrCode Icmp::encode(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d_);
    size_t tlen = sizeof(Detail) + extra_.raw.size();
    uint8_t* buf = new uint8_t[tlen];
    std::memcpy(buf, &dt, sizeof(Detail));
    std::memcpy(buf + sizeof(Detail), extra_.raw.data(), extra_.raw.size());
    dt.crc = calcChecksum(buf, tlen);
    const_cast<Icmp&>(*this).d_.crc = dt.crc;
    std::memcpy(buf, &dt, sizeof(Detail));
    bytes.insert(bytes.cbegin(), buf, buf + tlen);
    delete[] buf;
    return MyErrCode::kOk;
}

MyErrCode Icmp::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    auto ip = dynamic_cast<Ipv4 const*>(prev);
    extra_.raw = std::string(start + sizeof(Detail), start + ip->payloadSize());
    if (icmpType() == "error") {
        uint8_t const* pend = end;
        CHECK_ERR_RET(extra_.eip.decode(start + sizeof(Detail), pend, nullptr));
        std::memcpy(&extra_.buf, pend, 8);
    }
    return MyErrCode::kOk;
}

Variant Icmp::toVariant() const
{
    Variant j;
    j["type"] = FSTR(type());
    std::string tp = icmpType();
    j["icmp-type"] = tp;
    if (type_dict.count(d_.type) > 0) {
        auto& code_dict = type_dict.at(d_.type).second;
        if (code_dict.count(d_.code) > 0) {
            j["desc"] = code_dict.at(d_.code);
        }
    }
    j["id"] = d_.u.s.id;
    j["serial-no"] = d_.u.s.sn;
    size_t tlen = sizeof(Detail) + extra_.raw.size();
    uint8_t* buf = new uint8_t[tlen];
    auto dt = hton(d_);
    std::memcpy(buf, &dt, sizeof(Detail));
    std::memcpy(buf + sizeof(Detail), extra_.raw.data(), extra_.raw.size());
    j["checksum"] = calcChecksum(buf, tlen);
    delete[] buf;

    if (tp == "ping-reply" || tp == "ping-ask") {
        j["echo"] = extra_.raw;
    }
    if (tp == "error") {
        Variant ep;
        ep["ipv4"] = extra_.eip.toVariant();
        auto error_type = extra_.eip.succType();
        if (error_type == kTCP || error_type == kUDP) {
            ep["source-port"] = ntohs(*reinterpret_cast<uint16_t const*>(&extra_.buf[0]));
            ep["dest-port"] =
                ntohs(*reinterpret_cast<uint16_t const*>(&extra_.buf[0] + sizeof(uint16_t)));
        }
        j["error-header"] = ep;
    }
    return j;
}

Protocol::Type Icmp::type() const { return kICMP; }

Protocol::Type Icmp::succType() const { return kEmpty; }

bool Icmp::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Icmp const&>(rhs);
        return d_.u.s.id == p.d_.u.s.id && d_.u.s.sn == p.d_.u.s.sn;
    }
    return false;
}

Icmp::Detail const& Icmp::getDetail() const { return d_; }

Icmp::ExtraDetail const& Icmp::getExtra() const { return extra_; }

std::string Icmp::icmpType() const
{
    return type_dict.count(d_.type) > 0 ? type_dict.at(d_.type).first : "unknown";
}

Icmp::Detail Icmp::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.u.s.id, !reverse, s);
    ntohx(dt.u.s.sn, !reverse, s);
    return dt;
}

Icmp::Detail Icmp::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net