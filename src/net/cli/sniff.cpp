#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"

int main(int argc, char* argv[])
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.optional("ip", po::value<std::string>()->default_value(""),
                  "ipv4 address used to choose adapter, select first if empty");
    args.optional("filter", po::value<std::string>()->default_value(""),
                  "capture filter applied to adapter driver");
    args.parse();

    auto& apt = net::Adaptor::fit(args.has("ip") ? net::Ip4(args.get<std::string>("ip"))
                                                 : net::Ip4::kZeros);

    void* handle;
    CHECK_ERR_RET_INT(net::Transport::open(apt, handle));
    auto handle_guard = toolkit::scopeExit([&] { net::Transport::close(handle); });
    ILOG(apt.toVariant().toJsonStr(3));

    if (args.has("filter")) {
        std::string filter = args.get<std::string>("filter");
        ILOG("set filter \"{}\", mask={}", filter, apt.mask);
        CHECK_ERR_RET_INT(net::Transport::setFilter(handle, filter, apt.mask));
    }

    ILOG("begin to sniff...");
    CHECK_ERR_RET_INT(net::Transport::recv(handle, [&](net::Packet const& p) {
        ILOG(p.toVariant()["layers"].toJsonStr(3));
        return false;
    }));
    MY_CATCH
}