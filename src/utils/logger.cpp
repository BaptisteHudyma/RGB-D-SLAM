#include "logger.hpp"

#include <filesystem>
#include <iostream>

namespace rgbd_slam {
    namespace utils {

        void log(std::string_view message, const std::source_location& location)
        {
            std::cout << "[INF] "
                << std::filesystem::path(location.file_name()).filename().string() << "("
                << location.line() << ":" << location.column() <<  ") "
                //<< location.function_name() << " | "
                << message << std::endl;
        }
        void log_error(std::string_view message, const std::source_location& location)
        {
            std::cerr << "[ERR] "
                << std::filesystem::path(location.file_name()).filename().string() << "("
                << location.line() << ":" << location.column() <<  ") "
                //<< location.function_name() << " | "
                << message << std::endl;
        }

    }
}
