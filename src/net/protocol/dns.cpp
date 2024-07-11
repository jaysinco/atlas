#include "dns.h"
#include <boost/algorithm/string.hpp>

dns::dns(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    d = ntoh(*reinterpret_cast<detail const*>(start));
    auto it = start + sizeof(detail);
    for (int i = 0; i < d.qrn; ++i) {
        query_detail qr;
        qr.domain = decode_domain(start, end, it);
        qr.type = ntohs(*reinterpret_cast<uint16_t const*>(it));
        it += sizeof(uint16_t);
        qr.cls = ntohs(*reinterpret_cast<uint16_t const*>(it));
        it += sizeof(uint16_t);
        extra.query.push_back(qr);
    }
    auto parse_res = [&](std::vector<res_detail>& vec, uint16_t count) {
        for (int i = 0; i < count; ++i) {
            res_detail rr;
            rr.domain = decode_domain(start, end, it);
            rr.type = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            rr.cls = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            rr.ttl = ntohl(*reinterpret_cast<const uint32_t*>(it));
            it += sizeof(uint32_t);
            rr.dlen = ntohs(*reinterpret_cast<const uint16_t*>(it));
            it += sizeof(uint16_t);
            if (rr.type == 5) {  // CNAME
                rr.data = decode_domain(start, end, it);
            } else {
                rr.data = std::string(it, it + rr.dlen);
                it += rr.dlen;
            }
            vec.push_back(rr);
        }
    };
    parse_res(extra.reply, d.rrn);
    parse_res(extra.auth, d.arn);
    parse_res(extra.extra, d.ern);
}

dns::dns(std::string const& query_domain)
{
    d.id = rand_ushort();
    d.flags |= 0 << 15;          // qr
    d.flags |= (0 & 0xf) << 11;  // opcode
    d.flags |= 0 << 10;          // authoritative answer
    d.flags |= 1 << 9;           // truncated
    d.flags |= 1 << 8;           // recursion desired
    d.flags |= 0 << 7;           // recursion available
    d.flags |= (0 & 0xf);        // rcode
    d.qrn = 1;
    query_detail qr;
    qr.domain = query_domain;
    qr.type = 1;  // A
    qr.cls = 1;   // internet address
    extra.query.push_back(qr);
}

void dns::to_bytes(std::vector<uint8_t>& bytes) const
{
    auto dt = hton(d);
    auto it = reinterpret_cast<uint8_t const*>(&dt);
    bytes.insert(bytes.cbegin(), it, it + sizeof(detail));
    for (auto const& qr: extra.query) {
        std::string name = encode_domain(qr.domain);
        bytes.insert(bytes.end(), name.cbegin(), name.cend());
        uint16_t type = htons(qr.type);
        auto pt = reinterpret_cast<uint8_t*>(&type);
        bytes.insert(bytes.end(), pt, pt + sizeof(uint16_t));
        uint16_t cls = htons(qr.cls);
        auto pc = reinterpret_cast<uint8_t*>(&cls);
        bytes.insert(bytes.end(), pc, pc + sizeof(uint16_t));
    }
    auto encode_res = [&](std::vector<res_detail> const& rd) {
        for (const auto& rr: rd) {
            std::string name = encode_domain(rr.domain);
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
    encode_res(extra.reply);
    encode_res(extra.auth);
    encode_res(extra.extra);
}

json dns::to_json() const
{
    json j;
    j["type"] = type();
    j["id"] = d.id;
    j["dns-type"] = d.flags & 0x8000 ? "reply" : "query";
    j["opcode"] = (d.flags >> 11) & 0xf;
    j["authoritative-answer"] = d.flags & 0x400 ? true : false;
    j["truncated"] = d.flags & 0x200 ? true : false;
    j["recursion-desired"] = d.flags & 0x100 ? true : false;
    j["recursion-available"] = d.flags & 0x80 ? true : false;
    j["rcode"] = d.flags & 0xf;
    j["query-no"] = d.qrn;
    j["reply-no"] = d.rrn;
    j["author-no"] = d.arn;
    j["extra-no"] = d.ern;
    if (d.qrn > 0) {
        json query;
        for (auto const& qr: extra.query) {
            json r;
            r["domain"] = qr.domain;
            r["query-type"] = qr.type;
            r["query-class"] = qr.cls;
            query.push_back(r);
        }
        j["query"] = query;
    }
    auto jsonify_res = [](std::vector<res_detail> const& rd) -> json {
        json res;
        for (const auto& rr: rd) {
            json r;
            r["domain"] = rr.domain;
            r["query-type"] = rr.type;
            r["query-class"] = rr.cls;
            r["ttl"] = rr.ttl;
            r["data-size"] = rr.dlen;
            if (rr.type == 1) {  // A
                r["data"] = reinterpret_cast<const ip4*>(rr.data.data())->to_str();
            } else if (rr.type == 5) {  // CNAME
                r["data"] = rr.data;
            }
            res.push_back(r);
        }
        return res;
    };
    if (d.rrn > 0) j["reply"] = jsonify_res(extra.reply);
    if (d.arn > 0) j["author"] = jsonify_res(extra.auth);
    if (d.ern > 0) j["extra"] = jsonify_res(extra.extra);
    return j;
}

std::string dns::type() const { return Protocol_Type_DNS; }

std::string dns::succ_type() const { return Protocol_Type_Void; }

bool dns::link_to(protocol const& rhs) const
{
    if (type() == rhs.type()) {
        auto p = dynamic_cast<dns const&>(rhs);
        return d.id == p.d.id;
    }
    return false;
}

dns::detail const& dns::get_detail() const { return d; }

dns::extra_detail const& dns::get_extra() const { return extra; }

std::string dns::encode_domain(std::string const& domain)
{
    std::string bytes;
    std::vector<std::string> svec;
    boost::split(svec, domain, boost::is_any_of("."));
    for (auto const& s: svec) {
        if (s.size() > 63) {
            throw std::runtime_error("segment of domain exceed 63: {}"_format(s));
        }
        bytes.push_back(static_cast<uint8_t>(s.size()));
        bytes.insert(bytes.end(), s.cbegin(), s.cend());
    }
    bytes.push_back(0);
    return bytes;
}

std::string dns::decode_domain(uint8_t const* const pstart, uint8_t const* const pend,
                               uint8_t const*& it)
{
    std::vector<std::string> domain_vec;
    bool compressed = false;
    for (; it < pend && *it != 0;) {
        size_t cnt = *it;
        if ((cnt & 0xc0) != 0xc0) {
            domain_vec.push_back(std::string(it + 1, it + cnt + 1));
            it += cnt + 1;
        } else {
            compressed = true;
            uint16_t index = ((cnt & 0x3f) << 8) | it[1];
            auto new_it = pstart + index;
            domain_vec.push_back(decode_domain(pstart, pend, new_it));
            it += 2;
            break;
        }
    }
    if (!compressed) {
        ++it;
    }
    return boost::join(domain_vec, ".");
}

dns::detail dns::ntoh(detail const& d, bool reverse)
{
    detail dt = d;
    ntohx(dt.id, !reverse, s);
    ntohx(dt.flags, !reverse, s);
    ntohx(dt.qrn, !reverse, s);
    ntohx(dt.rrn, !reverse, s);
    ntohx(dt.arn, !reverse, s);
    ntohx(dt.ern, !reverse, s);
    return dt;
}

dns::detail dns::hton(detail const& d) { return ntoh(d, true); }
