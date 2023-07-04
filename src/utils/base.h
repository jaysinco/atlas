#pragma once
#include "error.h"
#include <filesystem>

#define CURR_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace utils
{

std::filesystem::path const& projectRoot();
std::filesystem::path const& currentExeDir();
std::filesystem::path defaultLoggingDir();
Expected<std::string> readFile(std::filesystem::path const& path);
MyErrCode setEnv(char const* varname, char const* value);

}  // namespace utils
