#include "transport/adaptor.h"

int main(int argc, char *argv[])
{
    NT_TRY
    INIT_LOG(argc, argv);
    json j;
    for (const auto &apt : adaptor::all()) {
        j.push_back(apt.to_json());
    }
    LOG(INFO) << j.dump(3);
    NT_CATCH
}
