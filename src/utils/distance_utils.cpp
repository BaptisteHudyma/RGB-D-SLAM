#include "distance_utils.hpp"
#include <cmath>

namespace rgbd_slam::utils {

double angle_distance(const double angleA, const double angleB) noexcept
{
    return atan2(sin(angleA - angleB), cos(angleA - angleB));
}

bool double_equal(const double a, const double b, const double epsilon) noexcept { return std::abs(a - b) <= epsilon; }

} // namespace rgbd_slam::utils