#include "traffic/adaptor.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"
#include "toolkit/variant.h"

MY_MAIN
{
    MY_TRY
    toolkit::Args args(argc, argv);
    args.parse();
    toolkit::Variant::Vec j;
    for (auto const& apt: net::Adaptor::all()) {
        j.push_back(apt.toVariant());
    }
    ILOG(toolkit::Variant(j).toJsonStr(3));
    MY_CATCH_RET
    return MyErrCode::kOk;
}
