#pragma once
#include "protocol.h"

class http: public protocol
{
public:
    struct detail
    {
        std::string op;                             // Request or Response
        std::string ver;                            // Protocol version
        std::string method;                         // Request method
        std::string url;                            // Request url
        int status;                                 // Response status
        std::string msg;                            // Response message
        std::map<std::string, std::string> header;  // Key-value header
        std::string body;                           // Content body
    };

    http() = default;

    http(uint8_t const* const start, uint8_t const*& end, protocol const* prev = nullptr);

    virtual ~http() = default;

    virtual void to_bytes(std::vector<uint8_t>& bytes) const override;

    virtual json to_json() const override;

    virtual std::string type() const override;

    virtual std::string succ_type() const override;

    virtual bool link_to(protocol const& rhs) const override;

    detail const& get_detail() const;

private:
    detail d{};
};
