#include "toolkit.h"
#include "logging.h"
#include <signal.h>
#ifdef __linux__
#include <unistd.h>
#include <libgen.h>
#include <execinfo.h>
#include <sys/wait.h>
#elif _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include <limits.h>
#include <fstream>
#include <sstream>

namespace toolkit
{

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
    return std::filesystem::path(buf);
#endif
}

std::string currentExeName() { return currentExePath().filename().string(); }

std::filesystem::path getLoggingDir() { return currentExeDir() / "logs"; }

std::filesystem::path getDataDir() { return currentExeDir() / "data"; }

std::filesystem::path getTempDir() { return currentExeDir() / "temp"; }

MyErrCode readFile(std::filesystem::path const& path, std::string& content)
{
    std::ifstream in_file(path);
    if (!in_file) {
        ELOG("failed to open file: {}", path.string());
        return MyErrCode::kFailed;
    }
    std::stringstream ss;
    ss << in_file.rdbuf();
    content = ss.str();
    return MyErrCode::kOk;
}

MyErrCode setEnv(char const* varname, char const* value)
{
#ifdef __linux__
    return ::setenv(varname, value, 1) == 0 ? MyErrCode::kOk : MyErrCode::kFailed;
#elif _WIN32
    return _putenv_s(varname, value) == 0 ? MyErrCode::kOk : MyErrCode::kFailed;
#endif
}

#ifdef __linux__
void handleSIGSEGV(int sig)
{
    fprintf(stderr, "signal SIGSEGV(%d) received!\n", sig);
    constexpr int kAddrsLen = 20;
    void* addrs[kAddrsLen] = {nullptr};
    int size = backtrace(addrs, kAddrsLen);
    backtrace_symbols_fd(addrs, size, STDERR_FILENO);
    std::abort();
}
#endif

MyErrCode installCrashHook()
{
#ifdef __linux__
    signal(SIGSEGV, handleSIGSEGV);
    return MyErrCode::kOk;
#elif _WIN32
    return MyErrCode::kUnimplemented;
#endif
}

MyErrCode runAsRoot(int argc, char* argv[])
{
#ifdef __linux__
    if (getuid() == 0) {
        return MyErrCode::kOk;
    }
    std::string cep = currentExePath().string();
    std::vector<char*> args;
    std::string sudo = "sudo";
    args.push_back(sudo.data());
    args.push_back(cep.data());
    for (int i = 1; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    args.push_back(nullptr);
    execvp("sudo", args.data());
    ELOG("failed to exec 'sudo {}': {}", cep, strerror(errno));
    exit(1);
    return MyErrCode::kOk;
#elif _WIN32
    return MyErrCode::kUnimplemented;
#endif
}

}  // namespace toolkit
