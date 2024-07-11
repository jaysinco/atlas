#include "dns.h"
#include <boost/algorithm/string.hpp>
#include "addr.h"
#include "toolkit/logging.h"

namespace net
{

Dns::Dns(std::string const& query_domain)
{
    d_.id = randUint16();
    d_.flags |= 0 << 15;          // qr
    d_.flags |= (0 & 0xf) << 11;  // opcode
    d_.flags |= 0 << 10;          // authoritative answer
    d_.flags |= 1 << 9;           // truncated
    d_.flags |= 1 << 8;           // recursion desired
    d_.flags |= 0 << 7;           // recursion available
    d_.flags |= (0 & 0xf);        // rcode
    d_.qrn = 1;
    QueryDetail qr;
    qr.domain = query_domain;
    qr.type = 1;  // A
    qr.cls = 1;   // internet address
    extra_.query.push_back(qr);
}

MyErrCode Dns::encode(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d_);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(Detail));
    for (auto const& qr: extra_.query) {
        std::string name = encodeDomain(qr.domain);
        bytes.insert(bytes.end(), name.cbegin(), name.cend());
        uint16_t type = htons(qr.type);
        auto pt = reinterpret_cast<uint8_t*>(&type);
        bytes.insert(bytes.end(), pt, pt + sizeof(uint16_t));
        uint16_t cls = htons(qr.cls);
        auto pc = reinterpret_cast<uint8_t*>(&cls);
        bytes.insert(bytes.end(), pc, pc + sizeof(uint16_t));
    }
    auto encode_res = [&](std::vector<ResDetail> const& rd) {
        for (const auto& rr: rd) {
            std::string name = encodeDomain(rr.domain);
            bytes.insert(bytes.end(), name.cbegin(), name.cend());
            uint16_t type = htons(rr.type);
            auto pt = reinterpret_cast<uint8_t*>(&type);
            bytes.insert(bytes.end(), pt, pt + sizeof(uint16_t));
            uint16_t cls = htons(rr.cls);
            auto pc = reinterpret_cast<uint8_t*>(&cls);
            bytes.insert(bytes.end(), pc, pc + sizeof(uint16_t));
            uint32_t ttl = htonl(rr.ttl);
            auto pl = reinterpret_cast<uint8_t*>(&ttl);
            bytes.insert(bytes.end(), pl, pl + sizeof(uint32_t));
            uint16_t dlen = htons(rr.dlen);
            auto pd = reinterpret_cast<uint8_t*>(&dlen);
            bytes.insert(bytes.end(), pd, pd + sizeof(uint16_t));
            bytes.insert(bytes.end(), rr.data.data(), rr.data.data() + rr.data.size());
        }
    };
    encode_res(extra_.reply);
    encode_res(extra_.auth);
    encode_res(extra_.extra);
    return MyErrCode::kOk;
}

MyErrCode Dns::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    d_ = ntoh(*reinterpret_cast<Detail const*>(start));
    auto it = start + sizeof(Detail);
    for (int i = 0; i < d_.qrn; ++i) {
        QueryDetail qr;
        qr.domain = decodeDomain(start, end, it);
        qr.type = ntohs(*reinterpret_cast<uint16_t const*>(it));
        it += sizeof(uint16_t);
        qr.cls = ntohs(*reinterpret_cast<uint16_t const*>(it));
        it += sizeof(uint16_t);
        extra_.query.push_back(qr);
    }
    auto parse_res = [&](std::vector<ResDetail>& vec, uint16_t count) {
        for (int i = 0; i < count; ++i) {
            ResDetail rr;
            rr.domain = decodeDomain(start, end, it);
            rr.type = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            rr.cls = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            rr.ttl = ntohl(*reinterpret_cast<const uint32_t*>(it));
            it += sizeof(uint32_t);
            rr.dlen = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            if (rr.type == 5) {  // CNAME
                rr.data = decodeDomain(start, end, it);
            } else {
                rr.data = std::string(it, it + rr.dlen);
                it += rr.dlen;
            }
            vec.push_back(rr);
        }
    };
    parse_res(extra_.reply, d_.rrn);
    parse_res(extra_.auth, d_.arn);
    parse_res(extra_.extra, d_.ern);
    return MyErrCode::kOk;
}

Variant Dns::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["id"] = d_.id;
    j["dns-type"] = d_.flags & 0x8000 ? "reply" : "query";
    j["opcode"] = (d_.flags >> 11) & 0xf;
    j["authoritative-answer"] = d_.flags & 0x400 ? true : false;
    j["truncated"] = d_.flags & 0x200 ? true : false;
    j["recursion-desired"] = d_.flags & 0x100 ? true : false;
    j["recursion-available"] = d_.flags & 0x80 ? true : false;
    j["rcode"] = d_.flags & 0xf;
    j["query-no"] = d_.qrn;
    j["reply-no"] = d_.rrn;
    j["author-no"] = d_.arn;
    j["extra-no"] = d_.ern;
    if (d_.qrn > 0) {
        Variant::Vec query;
        for (auto const& qr: extra_.query) {
            Variant r;
            r["domain"] = qr.domain;
            r["query-type"] = qr.type;
            r["query-class"] = qr.cls;
            query.push_back(r);
        }
        j["query"] = query;
    }
    auto jsonify_res = [](std::vector<ResDetail> const& rd) -> Variant {
        Variant::Vec res;
        for (const auto& rr: rd) {
            Variant r;
            r["domain"] = rr.domain;
            r["query-type"] = rr.type;
            r["query-class"] = rr.cls;
            r["ttl"] = static_cast<uint64_t>(rr.ttl);
            r["data-size"] = rr.dlen;
            if (rr.type == 1) {  // A
                r["data"] = reinterpret_cast<const Ip4*>(rr.data.data())->toStr();
            } else if (rr.type == 5) {  // CNAME
                r["data"] = rr.data;
            }
            res.push_back(r);
        }
        return res;
    };
    if (d_.rrn > 0) {
        j["reply"] = jsonify_res(extra_.reply);
    }
    if (d_.arn > 0) {
        j["author"] = jsonify_res(extra_.auth);
    }
    if (d_.ern > 0) {
        j["extra"] = jsonify_res(extra_.extra);
    }
    return j;
}

Protocol::Type Dns::type() const { return kDNS; }

Protocol::Type Dns::succType() const { return kEmpty; }

bool Dns::linkTo(Protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<Dns const&>(rhs);
        return d_.id == p.d_.id;
    }
    return false;
}

Dns::Detail const& Dns::getDetail() const { return d_; }

Dns::ExtraDetail const& Dns::getExtra() const { return extra_; }

std::string Dns::encodeDomain(std::string const& domain)
{
    std::string bytes;
    std::vector<std::string> svec;
    boost::split(svec, domain, boost::is_any_of("."));
    for (auto const& s: svec) {
        if (s.size() > 63) {
            MY_THROW("segment of domain exceed 63: {}", s);
        }
        bytes.push_back(static_cast<uint8_t>(s.size()));
        bytes.insert(bytes.end(), s.cbegin(), s.cend());
    }
    bytes.push_back(0);
    return bytes;
}

std::string Dns::decodeDomain(uint8_t const* const pstart, uint8_t const* const pend,
                              uint8_t const*& it)
{
    std::vector<std::string> domain_vec;
    bool compressed = false;
    for (; it < pend && *it != 0;) {
        size_t cnt = *it;
        if ((cnt & 0xc0) != 0xc0) {
            domain_vec.emplace_back(it + 1, it + cnt + 1);
            it += cnt + 1;
        } else {
            compressed = true;
            uint16_t index = ((cnt & 0x3f) << 8) | it[1];
            auto new_it = pstart + index;
            domain_vec.push_back(decodeDomain(pstart, pend, new_it));
            it += 2;
            break;
        }
    }
    if (!compressed) {
        ++it;
    }
    return boost::join(domain_vec, ".");
}

Dns::Detail Dns::ntoh(Detail const& d, bool reverse)
{
    Detail dt = d;
    ntohx(dt.id, !reverse, s);
    ntohx(dt.flags, !reverse, s);
    ntohx(dt.qrn, !reverse, s);
    ntohx(dt.rrn, !reverse, s);
    ntohx(dt.arn, !reverse, s);
    ntohx(dt.ern, !reverse, s);
    return dt;
}

Dns::Detail Dns::hton(Detail const& d) { return ntoh(d, true); }

}  // namespace net