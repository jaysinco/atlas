#pragma once
#include "protocol/addr.h"
#include "toolkit/variant.h"

namespace net
{

struct Adaptor
{
public:
    std::string name;
    std::string desc;
    Ip4 ip;
    Ip4 mask;
    Ip4 gateway;
    Mac mac;

    toolkit::Variant toVariant() const;

    static Adaptor const& fit(Ip4 const& hint = Ip4::kZeros);
    static bool isNative(Ip4 const& ip);
    static std::vector<Adaptor> const& all();

private:
    static MyErrCode getMacAddr(std::string const& if_name, Mac& mac);
    static MyErrCode getGateway(std::string const& if_name, Ip4& ip);
};

}  // namespace net

DEFINE_FORMATTER(net::Adaptor, item.toVariant().toJsonStr());
