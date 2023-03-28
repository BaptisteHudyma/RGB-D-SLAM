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
double angle_distance(const double angleA, const double angleB);

/**
 * \brief Check an equality between two doubles, with an epsilon
 * \param[in] epsilon
 */
bool double_equal(const double a, const double b, const double epsilon = std::numeric_limits<double>::epsilon());

} // namespace rgbd_slam::utils

#endif
