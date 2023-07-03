#include "utils/args.h"
#include <bshoshany/BS_thread_pool.hpp>

int main(int argc, char** argv)
{
    INIT_LOG(argc, argv);
    BS::thread_pool pool;
    std::future<int> my_future = pool.submit([] { return 42; });
    ILOG(my_future.get());
}
