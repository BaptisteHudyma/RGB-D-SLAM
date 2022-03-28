#ifndef RGBDSLAM_UTILS_LOGGER_HPP
#define RGBDSLAM_UTILS_LOGGER_HPP

#include <string_view>
#include <source_location>


namespace rgbd_slam {
    namespace utils {

        /**
         * Log an error line
         */
        void log(const std::string_view& message, const std::source_location& location = std::source_location::current());
        void log_error(const std::string_view& message, const std::source_location& location = std::source_location::current());

    }   // utils
}   // rgbd_slam

#endif
