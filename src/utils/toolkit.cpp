#include "toolkit.h"
#include "encoding.h"
#include <stdlib.h>
#ifdef __linux__
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#elif _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <fstream>
#include <sstream>

namespace utils
{

std::filesystem::path const& projectRoot()
{
    static auto root = currentExeDir().parent_path();
    return root;
}

static std::filesystem::path currentExeDirImpl()
{
#ifdef __linux__
    char cep[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", cep, PATH_MAX);
    return s2ws(dirname(cep));
#elif _WIN32
    wchar_t buf[MAX_PATH + 1] = {0};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    (wcsrchr(buf, L'\\'))[0] = 0;
    return buf;
#endif
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

bool setenv(char const* var, char const* value)
{
#ifdef __linux__
    return setenv(var, value, 1) == 0;
#elif _WIN32
    return _putenv_s(var, value) == 0;
#endif
}

}  // namespace utils
