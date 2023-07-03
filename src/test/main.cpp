#include "utils/args.h"

int main(int argc, char** argv)
{
    INIT_LOG(argc, argv);
    std::string world = "world";
    ILOG("hello, {}", world);
}
