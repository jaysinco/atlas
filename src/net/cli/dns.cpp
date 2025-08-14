#include "traffic/transport.h"
#include "toolkit/logging.h"
#include "toolkit/args.h"

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.positional("domain", po::value<std::string>(), "query domain", 1);
    args.optional("server,s", po::value<std::string>()->default_value("8.8.8.8"), "dns server ip");
    CHECK_ERR_RET(args.parse());

    auto opt_domain = args.get<std::string>("domain");
    auto opt_server = args.get<std::string>("server");

    net::Dns reply;
    CHECK_ERR_RET(net::Transport::queryDns(net::Ip4(opt_server), opt_domain, reply));
    ILOG(reply.toVariant().toJsonStr(3));
    MY_CATCH_RET
    return MyErrCode::kOk;
}
