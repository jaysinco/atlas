#include "transport/transport.h"

DEFINE_string(ip, "8.8.8.8", "dns server ip");

int main(int argc, char *argv[])
{
    NT_TRY
    INIT_LOG(argc, argv);
    if (argc < 2) {
        LOG(ERROR) << "empty domain name, please input domain name";
        return -1;
    }

    dns reply;
    if (transport::query_dns(ip4(FLAGS_ip), argv[1], reply)) {
        LOG(INFO) << reply.to_json().dump(3);
    }
    NT_CATCH
}
