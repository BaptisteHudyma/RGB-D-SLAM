#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include "types.hpp"
#include <limits>

namespace rgbd_slam::utils {

/**
 * \brief compute a distance bewteen two angles in radian
 * \param[in] angleA
 * \param[in] angleB
 */
[[nodiscard]] double angle_distance(const double angleA, const double angleB) noexcept;

/**
 * \brief Check an equality between two doubles, with an epsilon: TODO move this to a more sensible location
 * \param[in] a
 * \param[in] b
 * \param[in] epsilon
 * \return true if the two values are equal at +- epsilon
 */
[[nodiscard]] bool double_equal(const double a,
                                const double b,
                                const double epsilon = std::numeric_limits<double>::epsilon()) noexcept;

/**
 * \brief compute the distance between two lines.
 * This is the distance between the two points closest to each others on each line
 * \return a signed distance, in the same unit as the points
 */
[[nodiscard]] vector3 signed_line_distance(const vector3& line1point,
                                           const vector3& line1normal,
                                           const vector3& line2point,
                                           const vector3& line2normal) noexcept;

} // namespace rgbd_slam::utils

#endif
