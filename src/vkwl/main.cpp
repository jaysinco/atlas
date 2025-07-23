
#include "app.h"
#include "toolkit/args.h"
#include "toolkit/logging.h"

int main(int argc, char** argv)
{
    toolkit::Args args(argc, argv);
    args.parse();
    Application::run("vkwl", "vkwl");
    ILOG("end!");
    return 0;
}
