#include "polygon.hpp"
#include "types.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <tuple>
#include "logger.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Rotate the given vetor by 90 degrees
 */
vector2 rotate90(const vector2& other) { return vector2(-other.y(), other.x()); }

std::vector<vector2> compute_convex_hull(const std::vector<vector2>& pointsIn)
{
    if (pointsIn.size() < 3)
    {
        outputs::log_warning("Could not find boundary for plane patch");
        return std::vector<vector2>();
    }
    std::vector<vector2> sortedPoints(pointsIn);

    const vector2 first_point(*std::ranges::min_element(sortedPoints, [](const vector2& left, const vector2& right) {
        return std::make_tuple(left.y(), left.x()) < std::make_tuple(right.y(), right.x());
    })); // Find the lowest and leftmost point

    std::ranges::sort(sortedPoints, [&](const vector2& left, const vector2& right) {
        if (left.isApprox(first_point))
        {
            return right != first_point;
        }
        else if (right.isApprox(first_point))
        {
            return false;
        }
        const double dir = (rotate90(left - first_point)).dot(right - first_point);
        if (abs(dir) <= 0.01)
        { // If the points are on a line with first point, sort by distance (manhattan is equivalent here)
            return (left - first_point).lpNorm<2>() < (right - first_point).lpNorm<2>();
        }
        return dir > 0;
    }); // Sort the points by angle to the chosen first point

    std::vector<vector2> result;
    for (const vector2& point: sortedPoints)
    {
        // For as long as the last 3 points cause the hull to be non-convex, discard the middle one
        while (result.size() >= 2 && (rotate90(result[result.size() - 1] - result[result.size() - 2]))
                                                     .dot(point - result[result.size() - 1]) <= 0)
        {
            result.pop_back();
        }
        result.push_back(point);
    }
    return result;
}

} // namespace rgbd_slam::utils
