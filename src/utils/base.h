#pragma once
#include "error.h"
#include <filesystem>

#define CURR_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace utils
{

template <typename T>
struct ScopeExit
{
    explicit ScopeExit(T&& t): t(std::move(t)) {}

    ~ScopeExit() { t(); }

    T t;
};

template <typename T>
ScopeExit<T> scopeExit(T&& t)
{
    return ScopeExit<T>(std::move(t));
}

std::string currentExeName();
std::filesystem::path projectRoot();
std::filesystem::path currentExeDir();
std::filesystem::path currentExePath();
std::filesystem::path defaultLoggingDir();

MyErrCode setEnv(char const* varname, char const* value);
Expected<std::string> readFile(std::filesystem::path const& path);

}  // namespace utils
