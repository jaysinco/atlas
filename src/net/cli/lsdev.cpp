#include "traffic/adaptor.h"
#include "toolkit/args.h"
#include "toolkit/variant.h"

int main(int argc, char** argv)
{
    MY_TRY
    INIT_LOG(argc, argv);
    toolkit::Variant::Vec j;
    for (auto const& apt: net::Adaptor::all()) {
        j.push_back(apt.toVariant());
    }
    ILOG(toolkit::Variant(j).toJsonStr(3));
    MY_CATCH
}
