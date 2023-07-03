#include "fs.h"
#include "encoding.h"
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <fstream>
#include <sstream>

namespace utils
{

std::filesystem::path const& sourceRepo()
{
    static std::filesystem::path repo(std::getenv("MY_SOURCE_REPO"));
    return repo;
}

std::filesystem::path const& projectRoot()
{
    static auto root = currentExeDir().parent_path();
    return root;
}

static std::filesystem::path currentExeDirImpl()
{
    char cep[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", cep, PATH_MAX);
    return std::filesystem::path(cep).parent_path();
}

std::filesystem::path const& currentExeDir()
{
    static auto ced = currentExeDirImpl();
    return ced;
}

std::filesystem::path defaultLoggingDir() { return currentExeDir() / "logs"; }

Expected<std::string> readFile(std::filesystem::path const& path)
{
    std::ifstream in_file(path);
    if (!in_file) {
        return unexpected("failed to open file");
    }
    std::stringstream ss;
    ss << in_file.rdbuf();
    return ss.str();
}

}  // namespace utils
