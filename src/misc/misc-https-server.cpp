#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/toolkit.h"
#include <uwebsockets/App.h>

MY_MAIN
{
    toolkit::Args args(argc, argv);
    args.parse();
    auto key_file = (toolkit::getDataDir() / "key.pem").generic_string();
    auto cert_file = (toolkit::getDataDir() / "cert.pem").generic_string();

    uWS::SocketContextOptions opt;
    opt.key_file_name = key_file.c_str();
    opt.cert_file_name = cert_file.c_str();

    uWS::SSLApp(opt)
        .get("/*",
             [](auto* res, auto* req) {
                 res->writeHeader("Content-Type", "text/html; charset=utf-8");
                 res->end("Hello world!");
             })
        .listen(8080,
                [](auto* listen_socket) {
                    if (listen_socket) {
                        ILOG(
                            "listening on port {}",
                            us_socket_local_port(1, reinterpret_cast<us_socket_t*>(listen_socket)));
                    }
                })
        .run();
    return MyErrCode::kOk;
}
