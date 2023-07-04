#include "base.h"
#include "encoding.h"
#include "logging.h"
#include <stdlib.h>
#include <signal.h>
#ifdef __linux__
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <execinfo.h>
#elif _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <fstream>
#include <sstream>

namespace utils
{

std::filesystem::path projectRoot() { return currentExeDir().parent_path().parent_path(); }

std::filesystem::path currentExeDir() { return currentExePath().parent_path(); }

std::filesystem::path currentExePath()
{
#ifdef __linux__
    char cep[PATH_MAX] = {0};
    int n = readlink("/proc/self/exe", cep, PATH_MAX);
    return std::string_view(cep, n);
#elif _WIN32
    wchar_t buf[MAX_PATH + 1] = {0};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return buf;
#endif
}

std::string currentExeName() { return currentExePath().filename().string(); }

std::filesystem::path defaultLoggingDir() { return currentExeDir() / "logs"; }

Expected<std::string> readFile(std::filesystem::path const& path)
{
    std::ifstream in_file(path);
    if (!in_file) {
        ELOG("failed to open file: {}", path.string());
        return unexpected(MyErrCode::kFailed);
    }
    std::stringstream ss;
    ss << in_file.rdbuf();
    return ss.str();
}

MyErrCode setEnv(char const* varname, char const* value)
{
#ifdef __linux__
    return ::setenv(varname, value, 1) == 0 ? MyErrCode::kOk : MyErrCode::kFailed;
#elif _WIN32
    return _putenv_s(varname, value) == 0 ? MyErrCode::kOk : MyErrCode::kFailed;
#endif
}

void handleSIGSEGV(int sig)
{
    fprintf(stderr, "signal SIGSEGV(%d) received!\n", sig);
#ifdef __linux__
    constexpr int kAddrsLen = 20;
    void* addrs[kAddrsLen] = {nullptr};
    int size = backtrace(addrs, kAddrsLen);
    backtrace_symbols_fd(addrs, size, STDERR_FILENO);
#elif _WIN32
    // TODO: https://github.com/JochenKalmbach/StackWalker
#endif
    std::abort();
}

MyErrCode installCrashHook()
{
    signal(SIGSEGV, handleSIGSEGV);
    return MyErrCode::kOk;
}

}  // namespace utils
