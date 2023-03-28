#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include <cmath>

namespace rgbd_slam::utils {

/**
 * \brief compute a distance bewteen two angles in radian
 * \param[in] angleA
 * \param[in] angleB
 */
double angle_distance(const double angleA, const double angleB)
{
    return atan2(sin(angleA - angleB), cos(angleA - angleB));
}

} // namespace rgbd_slam::utils

#endif
