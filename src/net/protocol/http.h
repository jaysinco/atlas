#pragma once
#include "protocol.h"

namespace net
{

class Http: public Protocol
{
public:
    struct Detail
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

    Http() = default;

    ~Http() override = default;
    MyErrCode encode(std::vector<uint8_t>& bytes) const override;
    MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                     Protocol const* prev) override;
    Variant toVariant() const override;
    Type type() const override;
    Type succType() const override;
    bool linkTo(Protocol const& rhs) const override;

    Detail const& getDetail() const;

private:
    Detail d_{};
};

}  // namespace net