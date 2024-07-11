#include "port-table.h"
#include <iphlpapi.h>
#include <filesystem>

std::mutex port_table::lk_map;
port_table::storage_t port_table::map;
std::mutex port_table::lk_image;
port_table::image_cache_t port_table::image_cache;

std::string port_table::pid_to_image(uint32_t pid)
{
    std::lock_guard<std::mutex> lk(lk_image);
    auto start_tm = std::chrono::system_clock::now();
    auto it = image_cache.find(pid);
    if (it != image_cache.end()) {
        if (start_tm - it->second.second < 60s) {
            VLOG(3) << "use cached image for pid={}"_format(pid);
            return it->second.first;
        } else {
            VLOG(3) << "cached image for pid={} expired, call winapi to update"_format(pid);
        }
    }
    std::string s_default = "pid({})"_format(pid);
    HANDLE handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (handle == NULL) {
        image_cache[pid] = std::make_pair(s_default, std::chrono::system_clock::now());
        return s_default;
    }
    std::shared_ptr<void> handle_guard(nullptr, [&](void*) { CloseHandle(handle); });
    char buf[1024];
    DWORD size = sizeof(buf);
    if (!QueryFullProcessImageNameA(handle, 0, buf, &size)) {
        image_cache[pid] = std::make_pair(s_default, std::chrono::system_clock::now());
        return s_default;
    }
    std::filesystem::path fp(std::string(buf, size));
    std::string image = fp.filename().string();
    image_cache[pid] = std::make_pair(image, std::chrono::system_clock::now());
    return image;
}

port_table::storage_t port_table::tcp()
{
    ULONG size = sizeof(MIB_TCPTABLE);
    PMIB_TCPTABLE2 ptable = reinterpret_cast<MIB_TCPTABLE2*>(malloc(size));
    std::shared_ptr<void> ptable_guard(nullptr, [&](void*) { free(ptable); });
    DWORD ret = 0;
    if ((ret = GetTcpTable2(ptable, &size, FALSE)) == ERROR_INSUFFICIENT_BUFFER) {
        free(ptable);
        ptable = reinterpret_cast<MIB_TCPTABLE2*>(malloc(size));
        if (ptable == nullptr) {
            throw std::runtime_error("failed to allocate memory, size={}"_format(size));
        }
    }
    ret = GetTcpTable2(ptable, &size, FALSE);
    if (ret != NO_ERROR) {
        throw std::runtime_error("failed to get tcp port-pid table, ret={}"_format(ret));
    }
    port_table::storage_t map;
    for (int i = 0; i < ptable->dwNumEntries; ++i) {
        in_addr addr;
        addr.S_un.S_addr = ptable->table[i].dwLocalAddr;
        ip4 ip(addr);
        uint16_t port = ntohs(ptable->table[i].dwLocalPort);
        uint32_t pid = ptable->table[i].dwOwningPid;
        if (pid != 0) {
            map[std::make_tuple("tcp", ip, port)] = pid_to_image(pid);
        }
    }
    return map;
}

port_table::storage_t port_table::udp()
{
    ULONG size = sizeof(MIB_UDPTABLE_OWNER_PID);
    PMIB_UDPTABLE_OWNER_PID ptable = reinterpret_cast<MIB_UDPTABLE_OWNER_PID*>(malloc(size));
    std::shared_ptr<void> ptable_guard(nullptr, [&](void*) { free(ptable); });
    DWORD ret = 0;
    if ((ret = GetExtendedUdpTable(ptable, &size, FALSE, AF_INET, UDP_TABLE_OWNER_PID, 0)) ==
        ERROR_INSUFFICIENT_BUFFER) {
        free(ptable);
        ptable = reinterpret_cast<MIB_UDPTABLE_OWNER_PID*>(malloc(size));
        if (ptable == nullptr) {
            throw std::runtime_error("failed to allocate memory, size={}"_format(size));
        }
    }
    ret = GetExtendedUdpTable(ptable, &size, FALSE, AF_INET, UDP_TABLE_OWNER_PID, 0);
    if (ret != NO_ERROR) {
        throw std::runtime_error("failed to get udp port-pid table, ret={}"_format(ret));
    }
    port_table::storage_t map;
    for (int i = 0; i < ptable->dwNumEntries; ++i) {
        in_addr addr;
        addr.S_un.S_addr = ptable->table[i].dwLocalAddr;
        ip4 ip(addr);
        uint16_t port = ntohs(ptable->table[i].dwLocalPort);
        uint32_t pid = ptable->table[i].dwOwningPid;
        if (pid != 0) {
            map[std::make_tuple("udp", ip, port)] = pid_to_image(pid);
        }
    }
    return map;
}

void port_table::update()
{
    storage_t map_tcp(tcp());
    storage_t map_udp(udp());
    std::lock_guard<std::mutex> lk(lk_map);
    map.insert(map_tcp.begin(), map_tcp.end());
    map.insert(map_udp.begin(), map_udp.end());
}

void port_table::clear()
{
    {
        std::lock_guard<std::mutex> lk_i(lk_image);
        image_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lk_m(lk_map);
        map.clear();
    }
}

std::string port_table::lookup(key_type const& key)
{
    std::lock_guard<std::mutex> lk(lk_map);
    auto it = map.find(key);
    if (it == map.end()) {
        return "";
    }
    return it->second;
}
