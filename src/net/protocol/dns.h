#pragma once
#include "protocol.h"

namespace net
{

class Dns: public Protocol
{
public:
    struct Detail
    {
        uint16_t id;     // Identification
        uint16_t flags;  // Flags
        uint16_t qrn;    // Query number
        uint16_t rrn;    // Reply resource record number
        uint16_t arn;    // Auth resource record number
        uint16_t ern;    // Extra resource record number
    };

    struct QueryDetail
    {
        std::string domain;  // Query domain
        uint16_t type;       // Query type
        uint16_t cls;        // Query class
    };

    struct ResDetail
    {
        std::string domain;  // Domain
        uint16_t type;       // Query type
        uint16_t cls;        // Query class
        uint32_t ttl;        // Time to live
        uint16_t dlen;       // Resource data length
        std::string data;    // Resource data
    };

    struct ExtraDetail
    {
        std::vector<QueryDetail> query;  // Query
        std::vector<ResDetail> reply;    // Reply
        std::vector<ResDetail> auth;     // Auth
        std::vector<ResDetail> extra;    // Extra
    };

    Dns() = default;
    Dns(std::string const& query_domain);

    ~Dns() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;
    ExtraDetail const& getExtra() const;

private:
    Detail d_{0};
    ExtraDetail extra_;

    static std::string encodeDomain(std::string const& domain);
    static std::string decodeDomain(uint8_t const* const pstart, uint8_t const* const pend,
                                    uint8_t const*& it);

    static Detail ntoh(Detail const& d, bool reverse = false);
    static Detail hton(Detail const& d);
};

}  // namespace net