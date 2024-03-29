#ifndef RGBDSLAM_OUTPUTS_LOGGER_HPP
#define RGBDSLAM_OUTPUTS_LOGGER_HPP

#include <source_location>
#include <string_view>

namespace rgbd_slam::outputs {

/**
 * Log an error line
 */
void log(const std::string_view& message,
         const std::source_location& location = std::source_location::current()) noexcept;
void log_warning(const std::string_view& message,
                 const std::source_location& location = std::source_location::current()) noexcept;
void log_error(const std::string_view& message,
               const std::source_location& location = std::source_location::current()) noexcept;

} // namespace rgbd_slam::outputs

#endif
