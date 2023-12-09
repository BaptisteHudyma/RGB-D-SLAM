#include "distance_utils.hpp"
#include <cmath>

namespace rgbd_slam::utils {

double angle_distance(const double angleA, const double angleB) noexcept
{
    return atan2(sin(angleA - angleB), cos(angleA - angleB));
}

bool double_equal(const double a, const double b, const double epsilon) noexcept { return std::abs(a - b) <= epsilon; }

vector3 signed_line_distance(const vector3& p1, const vector3& d1, const vector3& p2, const vector3& d2) noexcept
{
    // we want to find a point that the two lines pass by, define by the origin of the line and the normal :
    // P = p1 + t1 * d1
    // P = p2 + t2 * d2

    const vector3 n = d1.cross(d2);
    // normals are parallels, distance is the distance between points
    if (n.isApproxToConstant(0.0))
    {
        // parallel line distance
        return d1.cross(p1 - p2);
    }

    const vector3& n1 = d1.cross(n);
    const vector3& n2 = d2.cross(n);

    const double t1 = (p2 - p1).dot(n2) / d1.dot(n2);
    const double t2 = (p1 - p2).dot(n1) / d2.dot(n1);

    const vector3 c1 = p1 + t1 * d1; // closest point to line 2 on line 1
    const vector3 c2 = p2 + t2 * d2; // closest point to line 1 on line 2
    return c1 - c2;
}

} // namespace rgbd_slam::utils