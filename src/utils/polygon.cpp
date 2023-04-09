#include "polygon.hpp"
#include "coordinates.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <boost/geometry/algorithms/area.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
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

Polygon::Polygon(const std::vector<vector2>& points)
{
    // set boundary in reverse (clockwise)
    std::vector<point_2d> boundaryPoints;
    boundaryPoints.reserve(points.size());
    std::ranges::transform(points.rbegin(), points.rend(), std::back_inserter(boundaryPoints), [](const vector2& c) {
        return point_2d(c.x(), c.y());
    });

    boost::geometry::assign_points(_polygon, boundaryPoints);
    boost::geometry::correct(_polygon);
    // simplify the input mesh
    simplify();
}
Polygon::Polygon(const std::vector<point_2d>& boundaryPoints)
{
    // set boundary in reverse (clockwise)
    boost::geometry::assign_points(_polygon, boundaryPoints);
    boost::geometry::correct(_polygon);
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
    }
    // close shape
    boundary.emplace_back(boundary[0]);
    return Polygon(boundary);
}

bool Polygon::contains(const vector2& point) const
{
    return boost::geometry::within(boost::geometry::make<point_2d>(point.x(), point.y()), _polygon);
}

void Polygon::merge(const Polygon& other)
{
    const polygon& res = union_one(other);
    if (res.outer().empty())
    {
        outputs::log_warning("Merge of two polygons produces no overlaps, returning without merge operation");
        return;
    }
    _polygon = res;
    // simplify the final mesh
    simplify();
}

double Polygon::area() const
{
    if (_polygon.outer().size() < 3)
    {
        outputs::log_error("Cannot compute the area of an empty polygon");
        return 0;
    }
    return boost::geometry::area(_polygon);
}

Polygon::polygon Polygon::union_one(const Polygon& other) const
{
    multi_polygon res;
    boost::geometry::union_(_polygon, other._polygon, res);
    if (res.empty() or res.size() > 1)
        return polygon(); // empty polygon : union produces more than one poly
    return res.front();
}

Polygon::polygon Polygon::inter_one(const Polygon& other) const
{
    multi_polygon res;
    boost::geometry::intersection(_polygon, other._polygon, res);
    if (res.empty() or res.size() > 1)
        return polygon(); // empty polygon : union produces more than one poly
    return res.front();
}

double Polygon::inter_over_union(const Polygon& other) const
{
    const polygon& un = union_one(other);
    if (un.outer().size() < 3)
        return 0.0;

    const double finalUnion = boost::geometry::area(un);
    const double finalInter = boost::geometry::area(inter_one(other));

    if (finalInter <= 0 or finalUnion <= 0)
        return 0.0;
    return finalInter / finalUnion;
}

Polygon Polygon::project(const vector3& currentCenter,
                         const vector3& currentUVec,
                         const vector3& currentVVec,
                         const vector3& nextCenter,
                         const vector3& nextUVec,
                         const vector3& nextVVec) const
{
    std::vector<point_2d> newPolygonBoundary;
    newPolygonBoundary.reserve(_polygon.outer().size());

    for (const point_2d& point: _polygon.outer())
    {
        // project to world
        const vector3& worldPoint = utils::get_point_from_plane_coordinates(
                vector2(point.x(), point.y()), currentCenter, currentUVec, currentVVec);

        // project back to plane space
        const vector2& projected = utils::get_projected_plan_coordinates(worldPoint, nextCenter, nextUVec, nextVVec);

        newPolygonBoundary.emplace_back(point_2d(projected.x(), projected.y()));
    }
    return Polygon(newPolygonBoundary);
}

void Polygon::display(const vector3& center,
                      const vector3& uVec,
                      const vector3& vVec,
                      const cv::Scalar& color,
                      cv::Mat& debugImage) const
{
    cv::Point previousPoint;
    bool isPreviousPointSet = false;
    for (const point_2d& p: _polygon.outer())
    {
        const utils::CameraCoordinate cameraPoint(
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), center, uVec, vVec));
        utils::ScreenCoordinate screenPoint;
        if (cameraPoint.to_screen_coordinates(screenPoint))
        {
            const cv::Point newPoint(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y()));
            if (isPreviousPointSet)
            {
                cv::line(debugImage, previousPoint, newPoint, color, 2);
            }
            cv::circle(debugImage,
                       cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                       3,
                       color,
                       -1);
            previousPoint = newPoint;
            isPreviousPointSet = true;
        }
    }
}

void Polygon::simplify(const double distanceThreshold)
{
    // use temporary object to prevent segfault
    polygon out;
    boost::geometry::simplify(_polygon, out, distanceThreshold);
    _polygon = out;
}

} // namespace rgbd_slam::utils
