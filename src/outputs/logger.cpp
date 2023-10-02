#include "logger.hpp"
#include <filesystem>
#include <iostream>

namespace rgbd_slam::outputs {

void log(const std::string_view& message, const std::source_location& location) noexcept
{
    std::cout << "[INF] " << std::filesystem::path(location.file_name()).filename().string() << "(" << location.line()
              << ":" << location.column()
              << ") "
              //<< location.function_name() << " | "
              << message << std::endl;
}
void log_warning(const std::string_view& message, const std::source_location& location) noexcept
{
    std::cerr << "[WARN] " << std::filesystem::path(location.file_name()).filename().string() << "(" << location.line()
              << ":" << location.column()
              << ") "
              //<< location.function_name() << " | "
              << message << std::endl;
}
void log_error(const std::string_view& message, const std::source_location& location) noexcept
{
    std::cerr << "[ERR] " << std::filesystem::path(location.file_name()).filename().string() << "(" << location.line()
              << ":" << location.column()
              << ") "
              //<< location.function_name() << " | "
              << message << std::endl;
}

} // namespace rgbd_slam::outputs
