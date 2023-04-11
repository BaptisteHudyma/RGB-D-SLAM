#include "polygon.hpp"
#include "coordinates.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <boost/geometry/algorithms/area.hpp>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "logger.hpp"

namespace rgbd_slam::utils {

std::pair<vector3, vector3> get_plane_coordinate_system(const vector3& normal)
{
    // define a vector orthogonal to the normal (r.dot normal should be close to 1)
    const vector3 r = vector3(normal.z(), normal.x(), normal.y()).normalized();

    // get two vectors that will span the plane
    const vector3 u = normal.cross(r).normalized();
    const vector3 v = normal.cross(u).normalized();

    // check that angles between vectors is close to 0
    assert(abs(u.dot(normal)) <= .01);
    assert(abs(v.dot(u)) <= .01);
    assert(abs(v.dot(normal)) <= .01);

    return std::make_pair(u, v);
}

Polygon::Polygon(const std::vector<vector2>& points,
                 const vector3& center,
                 const vector3& xAxis,
                 const vector3& yAxis) :
    _center(center),
    _xAxis(xAxis),
    _yAxis(yAxis)
{
    // set boundary in reverse (clockwise)
    std::vector<point_2d> boundaryPoints;
    boundaryPoints.reserve(points.size());
    std::ranges::transform(points.rbegin(), points.rend(), std::back_inserter(boundaryPoints), [](const vector2& c) {
        return boost::geometry::make<point_2d>(c.x(), c.y());
    });

    boost::geometry::assign_points(_polygon, boundaryPoints);
    boost::geometry::correct(_polygon);
    // simplify the input mesh
    simplify();
}
Polygon::Polygon(const std::vector<point_2d>& boundaryPoints,
                 const vector3& center,
                 const vector3& xAxis,
                 const vector3& yAxis) :
    _center(center),
    _xAxis(xAxis),
    _yAxis(yAxis)
{
    // set boundary in reverse (clockwise)
    boost::geometry::assign_points(_polygon, boundaryPoints);
    boost::geometry::correct(_polygon);
}

/**
 * \brief Rotate the given vetor by 90 degrees
 */
vector2 rotate90(const vector2& other) { return vector2(-other.y(), other.x()); }

std::vector<Polygon::point_2d> Polygon::compute_convex_hull(const std::vector<vector2>& pointsIn)
{
    if (pointsIn.size() < 3)
    {
        outputs::log_warning("Cannot compute a polygon with less than 3 sides");
        return std::vector<point_2d>();
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

    std::vector<point_2d> res;
    res.reserve(boundary.size());
    for (const vector2& point: boundary)
    {
        res.emplace_back(point.x(), point.y());
    }

    return res;
}

bool Polygon::contains(const vector2& point) const
{
    return boost::geometry::within(boost::geometry::make<point_2d>(point.x(), point.y()), _polygon);
}

void Polygon::merge(const Polygon& other)
{
    const polygon& res = union_one(other.project(_center, _xAxis, _yAxis));
    if (res.outer().empty())
    {
        outputs::log_warning("Merge of two polygons produces no overlaps, returning without merge operation");
        return;
    }
    _polygon = res;
    // simplify the final mesh
    simplify();
}

Polygon Polygon::project(const vector3& nextCenter, const vector3& nextXAxis, const vector3& nextYAxis) const
{
    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    for (const auto& p: _polygon.outer())
    {
        const vector3& retroProjected =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);

        const vector2& projected =
                utils::get_projected_plan_coordinates(retroProjected, nextCenter, nextXAxis, nextYAxis);
        newBoundary.emplace_back(projected.x(), projected.y());
    }

    return Polygon(newBoundary, nextCenter, nextXAxis, nextYAxis);
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
    boost::geometry::union_(_polygon, other.project(_center, _xAxis, _yAxis)._polygon, res);
    if (res.empty() or res.size() > 1)
        return polygon(); // empty polygon : union produces more than one poly
    return res.front();
}

Polygon::polygon Polygon::inter_one(const Polygon& other) const
{
    multi_polygon res;
    boost::geometry::intersection(_polygon, other.project(_center, _xAxis, _yAxis)._polygon, res);
    if (res.empty() or res.size() > 1)
        return polygon(); // empty polygon : union produces more than one poly
    return res.front();
}

double Polygon::inter_over_union(const Polygon& other) const
{
    const Polygon& projectedOther = other.project(_center, _xAxis, _yAxis);
    const polygon& un = union_one(projectedOther);
    if (un.outer().size() < 3)
        return 0.0;

    const double finalUnion = boost::geometry::area(un);
    const double finalInter = boost::geometry::area(inter_one(projectedOther));

    if (finalInter <= 0 or finalUnion <= 0)
        return 0.0;
    return finalInter / finalUnion;
}

double Polygon::inter_area(const Polygon& other) const
{
    const polygon& inter = inter_one(other.project(_center, _xAxis, _yAxis));
    if (inter.outer().size() < 3)
        return 0.0;
    return boost::geometry::area(inter);
}

void Polygon::simplify(const double distanceThreshold)
{
    // use temporary object to prevent segfault
    polygon out;
    boost::geometry::simplify(_polygon, out, distanceThreshold);
    _polygon = out;
}

std::vector<vector2> Polygon::get_boundary() const
{
    std::vector<vector2> boundaryPoints;
    boundaryPoints.reserve(_polygon.outer().size());
    std::ranges::transform(_polygon.outer().rbegin(),
                           _polygon.outer().rend(),
                           std::back_inserter(boundaryPoints),
                           [](const point_2d& c) {
                               return vector2(c.x(), c.y());
                           });
    return boundaryPoints;
}

std::vector<vector2> Polygon::get_envelop() const
{
    box_2d box;
    boost::geometry::envelope(_polygon, box);

    std::vector<vector2> vectorBox;

    vectorBox.emplace_back(vector2(box.min_corner().x(), box.min_corner().y()));
    vectorBox.emplace_back(vector2(box.max_corner().x(), box.max_corner().y()));

    return vectorBox;
}

/**
 *
 * CAMERA POLYGON
 *
 */

void CameraPolygon::display(const cv::Scalar& color, cv::Mat& debugImage) const
{
    cv::Point previousPoint;
    bool isPreviousPointSet = false;
    for (const ScreenCoordinate& screenPoint: get_screen_points())
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

WorldPolygon CameraPolygon::project(const CameraToWorldMatrix& c2w) const
{
    const WorldCoordinate& newCenter = CameraCoordinate(_center).to_world_coordinates(c2w);

    // rotate axis
    const matrix33& rotationMatrix = c2w.block(0, 0, 3, 3);
    const vector3 newXAxis = rotationMatrix * _xAxis;
    const vector3 newYAxis = rotationMatrix * _yAxis;

    assert(abs(newXAxis.dot(newYAxis)) <= .01);

    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const CameraCoordinate cameraPoint(
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis));
        const WorldCoordinate& w = cameraPoint.to_world_coordinates(c2w);

        const vector2& newPolygonPoint = utils::get_projected_plan_coordinates(w, newCenter, newXAxis, newYAxis);
        newBoundary.emplace_back(point_2d(newPolygonPoint.x(), newPolygonPoint.y()));
    }

    return WorldPolygon(newBoundary, newCenter, newXAxis, newYAxis);
}

std::vector<ScreenCoordinate> CameraPolygon::get_screen_points() const
{
    std::vector<ScreenCoordinate> screenBoundary;
    screenBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const CameraCoordinate& projectedPoint =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);

        ScreenCoordinate s;
        if (not projectedPoint.to_screen_coordinates(s))
            std::cout << "Could not transform plane to screen coordinates" << std::endl;
        else
            screenBoundary.emplace_back(s);
    }
    return screenBoundary;
}

/**
 *
 * WORLD POLYGON
 *
 */

CameraPolygon WorldPolygon::project(const WorldToCameraMatrix& w2c) const
{
    const CameraCoordinate& newCenter = WorldCoordinate(_center).to_camera_coordinates(w2c);

    // rotate axis
    const matrix33& rotationMatrix = w2c.block(0, 0, 3, 3);
    const vector3 newXAxis = rotationMatrix * _xAxis;
    const vector3 newYAxis = rotationMatrix * _yAxis;

    assert(abs(newXAxis.dot(newYAxis)) <= .01);

    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const WorldCoordinate& worldPoint =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);
        const CameraCoordinate& c = worldPoint.to_camera_coordinates(w2c);

        const vector2& newPolygonPoint = utils::get_projected_plan_coordinates(c, newCenter, newXAxis, newYAxis);
        newBoundary.emplace_back(point_2d(newPolygonPoint.x(), newPolygonPoint.y()));
    }

    return CameraPolygon(newBoundary, newCenter, newXAxis, newYAxis);
}

void WorldPolygon::merge(const WorldPolygon& other)
{
    Polygon::merge(other.Polygon::project(_center, _xAxis, _yAxis));
};

} // namespace rgbd_slam::utils
