#include "distance_utils.hpp"

namespace rgbd_slam::utils {

double angle_distance(const double angleA, const double angleB)
{
    return atan2(sin(angleA - angleB), cos(angleA - angleB));
}

bool double_equal(const double a, const double b, const double epsilon) { return std::abs(a - b) <= epsilon; }

} // namespace rgbd_slam::utils