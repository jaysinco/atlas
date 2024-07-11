#pragma once
#include "protocol/type.h"
#include <map>
#include <tuple>
#include <mutex>

class port_table
{
public:
    using key_type = std::tuple<std::string, ip4, uint16_t>;

    using storage_t = std::map<key_type, std::string>;

    using image_cache_t =
        std::map<uint32_t, std::pair<std::string, std::chrono::system_clock::time_point>>;

    static void update();

    static void clear();

    static std::string lookup(key_type const& key);

private:
    static std::string pid_to_image(uint32_t pid);

    static storage_t tcp();

    static storage_t udp();

    static std::mutex lk_map;
    static storage_t map;

    static std::mutex lk_image;
    static image_cache_t image_cache;
};
