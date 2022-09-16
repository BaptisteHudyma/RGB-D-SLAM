#ifndef RGBDSLAM_OUTPUTS_LOGGER_HPP
#define RGBDSLAM_OUTPUTS_LOGGER_HPP

#include <string_view>
#include <source_location>


namespace rgbd_slam {
    namespace outputs {

        /**
         * Log an error line
         */
        void log(const std::string_view& message, const std::source_location& location = std::source_location::current());
        void log_error(const std::string_view& message, const std::source_location& location = std::source_location::current());

    }   // outputs
}   // rgbd_slam

#endif
