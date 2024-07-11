#pragma once
#include "protocol/type.h"

struct adaptor
{
    std::string name;
    std::string desc;
    ip4 ip;
    ip4 mask;
    ip4 gateway;
    mac mac_;

public:
    json to_json() const;

    static const adaptor &fit(const ip4 &hint = ip4::zeros);

    static bool is_native(const ip4 &ip);

    static const std::vector<adaptor> &all();
};
