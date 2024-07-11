#pragma once
#include "protocol.h"

class dns: public protocol
{
public:
    struct detail
    {
        uint16_t id;     // Identification
        uint16_t flags;  // Flags
        uint16_t qrn;    // Query number
        uint16_t rrn;    // Reply resource record number
        uint16_t arn;    // Auth resource record number
        uint16_t ern;    // Extra resource record number
    };

    struct query_detail
    {
        std::string domain;  // Query domain
        uint16_t type;       // Query type
        uint16_t cls;        // Query class
    };

    struct res_detail
    {
        std::string domain;  // Domain
        uint16_t type;       // Query type
        uint16_t cls;        // Query class
        uint32_t ttl;        // Time to live
        uint16_t dlen;       // Resource data length
        std::string data;    // Resource data
    };

    struct extra_detail
    {
        std::vector<query_detail> query;  // Query
        std::vector<res_detail> reply;    // Reply
        std::vector<res_detail> auth;     // Auth
        std::vector<res_detail> extra;    // Extra
    };

    dns() = default;

    dns(uint8_t const* const start, uint8_t const*& end, protocol const* prev = nullptr);

    dns(std::string const& query_domain);

    virtual ~dns() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

    extra_detail const& get_extra() const;

private:
    detail d{0};

    extra_detail extra;

    static std::string encode_domain(std::string const& domain);

    static std::string decode_domain(uint8_t const* const pstart, uint8_t const* const pend,
                                     uint8_t const*& it);

    static detail ntoh(detail const& d, bool reverse = false);

    static detail hton(detail const& d);
};
