#include "polygon.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <tuple>
#include <utility>
#include "logger.hpp"

namespace rgbd_slam::utils {

std::pair<vector3, vector3> get_plane_coordinate_system(const vector3& normal)
{
    const vector3 r = vector3(-normal.z(), normal.x(), normal.y()).normalized();
    assert(abs(r.dot(normal)) > 0.00001);

    const vector3 u = normal.cross(r).normalized();
    const vector3 v = normal.cross(u).normalized();

    return std::make_pair(u, v);
}

Polygon::Polygon(const std::vector<vector2>& points,
                 const vector2& lowerLeftBoundary,
                 const vector2& upperRightBoundary) :
    _boundaryPoints(points),
    _lowerLeftBoundary(lowerLeftBoundary),
    _upperRightBoundary(upperRightBoundary)
{
}

/**
 * \brief Rotate the given vetor by 90 degrees
 */
vector2 rotate90(const vector2& other) { return vector2(-other.y(), other.x()); }

Polygon Polygon::compute_convex_hull(const std::vector<vector2>& pointsIn)
{
    if (pointsIn.size() < 3)
    {
        outputs::log_warning("Cannot compute a polygon with less than 3 sides");
        return Polygon();
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

    vector2 lowerLeftBoundary = vector2::Zero();
    vector2 upperRightBoundary = vector2::Zero();
    std::vector<vector2> boundary;
    for (const vector2& point: sortedPoints)
    {
        // For as long as the last 3 points cause the hull to be non-convex, discard the middle one
        while (boundary.size() >= 2 && (rotate90(boundary[boundary.size() - 1] - boundary[boundary.size() - 2]))
                                                       .dot(point - boundary[boundary.size() - 1]) <= 0)
        {
            boundary.pop_back();
        }
        boundary.push_back(point);

        lowerLeftBoundary.x() = std::min(lowerLeftBoundary.x(), point.x());
        lowerLeftBoundary.y() = std::min(lowerLeftBoundary.y(), point.y());
        upperRightBoundary.x() = std::max(upperRightBoundary.x(), point.x());
        upperRightBoundary.y() = std::max(upperRightBoundary.y(), point.y());
    }
    return Polygon(boundary, lowerLeftBoundary, upperRightBoundary);
}

bool Polygon::contains(const vector2& point) const
{
    // Source: Wm. Randolph Franklin

    const double pointX = point.x();
    const double pointY = point.y();

    // Check that the point is inside the coarse boundary for efficiency
    /*if (p.x() < _Xmin or p.x() > _Xmax or p.y() < _Ymin or p.y() > _Ymax)
        return false;
    */

    bool res = false;
    // i form 0 to n-1, j = i-1
    for (uint i = 0, j = _boundaryPoints.size() - 1; i < _boundaryPoints.size(); j = i++)
    {
        const vector2& vertex1 = _boundaryPoints[i];
        const vector2& vertex2 = _boundaryPoints[j];

        if (((vertex1.y() > pointY) != (vertex2.y() > pointY)) and
            (pointX < (vertex2.x() - vertex1.x()) * (pointY - vertex1.y()) / (vertex2.y() - vertex1.y()) + vertex1.x()))
            res = !res;
    }
    return res;
}

void Polygon::merge(const Polygon& other)
{
    // TODO
    _boundaryPoints = other._boundaryPoints;
    _lowerLeftBoundary = other._lowerLeftBoundary;
    _upperRightBoundary = other._upperRightBoundary;
}

} // namespace rgbd_slam::utils
