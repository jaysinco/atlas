#pragma once
#include "error.h"
#include <filesystem>
#define CURR_FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace utils
{

std::filesystem::path const& sourceRepo();
std::filesystem::path const& projectRoot();
std::filesystem::path const& currentExeDir();
std::filesystem::path defaultLoggingDir();
Expected<std::string> readFile(std::filesystem::path const& path);

}  // namespace utils
