#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include <cmath>
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

} // namespace rgbd_slam::utils

#endif
