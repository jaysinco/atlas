#pragma once
#include "toolkit/variant.h"
#include "toolkit/error.h"
#include "toolkit/format.h"

#define ntohx(field, reverse, suffix) field = ((reverse) ? ntoh##suffix : hton##suffix)(field);

namespace net
{

using Variant = toolkit::Variant;

class Protocol
{
public:
    using Ptr = std::shared_ptr<Protocol>;
    using Stack = std::vector<Ptr>;

    enum Type
    {
        kEmpty,
        kUnknown,
        kEthernet,
        kIPv4,
        kIPv6,
        kARP,
        kRARP,
        kICMP,
        kTCP,
        kUDP,
        kDNS,
        kHTTP,
        kHTTPS,
        kSSH,
        kTELNET,
        kRDP,
    };

    // destructor should be virtual
    virtual ~Protocol() = default;

    // serialize current protocol layer
    virtual MyErrCode encode(std::vector<uint8_t>& bytes) const = 0;

    // deserialize current protocol layer
    virtual MyErrCode decode(uint8_t const* const start, uint8_t const*& end,
                             Protocol const* prev = nullptr) = 0;

    // convert protocol detail to variant
    virtual Variant toVariant() const = 0;

    // protocol type
    virtual Type type() const = 0;

    // successor protocol type that follows
    virtual Type succType() const = 0;

    // whether this is request, and rhs is the response
    virtual bool linkTo(Protocol const& rhs) const = 0;

    // format to str
    std::string toStr() const;

protected:
    static uint16_t randUint16();
    static uint16_t calcChecksum(void const* data, size_t tlen);
    static Type guessProtocolByPort(uint16_t port, Type type = kTCP);

private:
    static std::map<Type, std::map<uint16_t, Type>> protocol_ports;
};

std::string toString(Protocol::Type type);

}  // namespace net
