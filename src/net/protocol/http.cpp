#include "http.h"
#include <algorithm>
#include <regex>

http::http(uint8_t const* const start, uint8_t const*& end, protocol const* prev)
{
    static std::regex req_r("(\\w+?) (\\S+?) HTTP/(\\d\\.\\d)");
    static std::regex resp_r("HTTP/(\\d\\.\\d) (\\d+?) (.+?)");

    std::string delim = "\r\n";
    auto fl_end = std::search(start, end, delim.cbegin(), delim.cend());
    if (fl_end == end) {
        d.op = "fragments";
        return;
    }
    std::string fl(start, fl_end);
    std::smatch res_m;
    if (std::regex_match(fl, res_m, req_r)) {
        d.op = "request";
        d.method = res_m.str(1);
        d.url = res_m.str(2);
        d.ver = res_m.str(3);
    } else if (std::regex_match(fl, res_m, resp_r)) {
        d.op = "response";
        d.ver = res_m.str(1);
        d.status = std::stoi(res_m.str(2));
        d.msg = res_m.str(3);
    } else {
        d.op = "fragments";
        return;
    }
    auto nl_start = fl_end + 2;
    while (nl_start < end) {
        auto nl_end = std::search(nl_start, end, delim.cbegin(), delim.cend());
        if (nl_end == end) {
            return;
        } else if (nl_end == nl_start) {
            nl_start = nl_end + 2;
            break;
        } else {
            std::string line(nl_start, nl_end);
            size_t idx = line.find_first_of(':');
            if (idx == std::string::npos || idx == line.size() - 1) {
                return;
            }
            d.header[line.substr(0, idx)] = line.substr(idx + 1);
        }
        nl_start = nl_end + 2;
    }
    d.body = std::string(nl_start, end);
}

void http::to_bytes(std::vector<uint8_t>& bytes) const { MY_THROW("unimplemented method"); }

json http::to_json() const
{
    json j;
    j["type"] = type();
    j["op"] = d.op;
    if (d.op == "request") {
        j["version"] = d.ver;
        j["method"] = "{} {}"_format(d.method, d.url);
    } else if (d.op == "response") {
        j["version"] = d.ver;
        j["status"] = "{} {}"_format(d.status, d.msg);
    } else {
        return j;
    }
    json header;
    for (auto const& kv: d.header) {
        header[kv.first] = kv.second;
    }
    if (!header.is_null()) {
        j["header"] = header;
    }
    j["body"] = d.body;
    return j;
}

std::string http::type() const { return Protocol_Type_HTTP; }

std::string http::succ_type() const { return Protocol_Type_Void; }

bool http::link_to(protocol const& rhs) const { MY_THROW("unimplemented method"); }

http::detail const& http::get_detail() const { return d; }
