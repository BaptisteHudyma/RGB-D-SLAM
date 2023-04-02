#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Compute the convex hull for a set of points
 * \param[in] points The points to compute a convex hull for
 * \return The ordered points defining a convex hull of points
 */
std::vector<vector2> compute_convex_hull(const std::vector<vector2>& points);

} // namespace rgbd_slam::utils

#endif