#include "base.h"
#include "encoding.h"
#include "logging.h"
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
    static auto root = currentExeDir().parent_path().parent_path();
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
        ELOG("failed to open file: {}", path.string());
        return unexpected(MyErrCode::kUnknown);
    }
    std::stringstream ss;
    ss << in_file.rdbuf();
    return ss.str();
}

MyErrCode setEnv(char const* varname, char const* value)
{
#ifdef __linux__
    return ::setenv(varname, value, 1) == 0 ? MyErrCode::kOk : MyErrCode::kUnknown;
#elif _WIN32
    return _putenv_s(varname, value) == 0 ? MyErrCode::kOk : MyErrCode::kUnknown;
#endif
}

}  // namespace utils
