#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.optional("ip,i", po::value<std::string>()->default_value(""),
                  "ipv4 address used to choose adapter");
    args.optional("filter,f", po::value<std::string>()->default_value(""),
                  "capture filter applied to adapter driver");
    args.parse();

    auto opt_ip = args.get<std::string>("ip");
    auto opt_filter = args.get<std::string>("filter");

    auto& apt = net::Adaptor::fit(!opt_ip.empty() ? net::Ip4(opt_ip) : net::Ip4::kZeros);
    void* handle;
    CHECK_ERR_RET_INT(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });
    ILOG(apt.toVariant().toJsonStr(3));

    if (!opt_filter.empty()) {
        ILOG("set filter \"{}\", mask={}", opt_filter, apt.mask);
        CHECK_ERR_RET_INT(net::Transport::setFilter(handle, opt_filter, apt.mask));
    }

    ILOG("begin to sniff...");
    CHECK_ERR_RET_INT(net::Transport::recv(handle, [&](net::Packet const& p) {
        ILOG(p.toVariant()["layers"].toJsonStr(3));
        return false;
    }));
    MY_CATCH
}