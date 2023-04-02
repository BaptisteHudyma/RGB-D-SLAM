#ifndef RGBDSLAM_UTILS_POLYGON_UTILS_HPP
#define RGBDSLAM_UTILS_POLYGON_UTILS_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Find the best fitting polygon to a set of points
 */
std::vector<vector2> get_best_fitting_polygon(const std::vector<vector2>& points);

} // namespace rgbd_slam::utils

#endif