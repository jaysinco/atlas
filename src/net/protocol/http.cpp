#include "http.h"
#include <algorithm>
#include <regex>
#include "toolkit/logging.h"

namespace net
{

MyErrCode Http::encode(std::vector<uint8_t>& bytes) const { return MyErrCode::kUnimplemented; }

MyErrCode Http::decode(uint8_t const* const start, uint8_t const*& end, Protocol const* prev)
{
    static std::regex req_r("(\\w+?) (\\S+?) HTTP/(\\d\\.\\d)");
    static std::regex resp_r("HTTP/(\\d\\.\\d) (\\d+?) (.+?)");

    std::string delim = "\r\n";
    auto fl_end = std::search(start, end, delim.cbegin(), delim.cend());
    if (fl_end == end) {
        d_.op = "fragments";
        return MyErrCode::kOk;
    }
    std::string fl(start, fl_end);
    std::smatch res_m;
    if (std::regex_match(fl, res_m, req_r)) {
        d_.op = "request";
        d_.method = res_m.str(1);
        d_.url = res_m.str(2);
        d_.ver = res_m.str(3);
    } else if (std::regex_match(fl, res_m, resp_r)) {
        d_.op = "response";
        d_.ver = res_m.str(1);
        d_.status = std::stoi(res_m.str(2));
        d_.msg = res_m.str(3);
    } else {
        d_.op = "fragments";
        return MyErrCode::kOk;
    }
    auto nl_start = fl_end + 2;
    while (nl_start < end) {
        auto nl_end = std::search(nl_start, end, delim.cbegin(), delim.cend());
        if (nl_end == end) {
            return MyErrCode::kOk;
        } else if (nl_end == nl_start) {
            nl_start = nl_end + 2;
            break;
        } else {
            std::string line(nl_start, nl_end);
            size_t idx = line.find_first_of(':');
            if (idx == std::string::npos || idx == line.size() - 1) {
                return MyErrCode::kOk;
            }
            d_.header[line.substr(0, idx)] = line.substr(idx + 1);
        }
        nl_start = nl_end + 2;
    }
    d_.body = std::string(nl_start, end);
    return MyErrCode::kOk;
}

Variant Http::toVariant() const
{
    Variant j;
    j["type"] = TOSTR(type());
    j["op"] = d_.op;
    if (d_.op == "request") {
        j["version"] = d_.ver;
        j["method"] = FSTR("{} {}", d_.method, d_.url);
    } else if (d_.op == "response") {
        j["version"] = d_.ver;
        j["status"] = FSTR("{} {}", d_.status, d_.msg);
    } else {
        return j;
    }
    Variant::Map header;
    for (auto const& kv: d_.header) {
        header[kv.first] = kv.second;
    }
    if (!header.empty()) {
        j["header"] = header;
    }
    j["body"] = d_.body;
    return j;
}

Protocol::Type Http::type() const { return kHTTP; }

Protocol::Type Http::succType() const { return kEmpty; }

bool Http::linkTo(Protocol const& rhs) const { MY_THROW("{}", "unimplemented method"); }

Http::Detail const& Http::getDetail() const { return d_; }

}  // namespace net