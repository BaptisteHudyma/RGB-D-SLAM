#include "polygon.hpp"
#include "coordinates.hpp"
#include "distance_utils.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include <algorithm>
#include <bits/ranges_algo.h>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/detail/convex_hull/interface.hpp>
#include <boost/geometry/algorithms/union.hpp>
#include <boost/qvm/mat_operations.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include "logger.hpp"
#include "concave_fitting.hpp"
#include "correct_boost_polygon.hpp"

namespace rgbd_slam::utils {

/**
 * \brief Select the transform vector furthest from the normal
 */
vector3 select_correct_transform(const vector3& normal) noexcept
{
    const double distX = abs(normal.dot(vector3::UnitX()));
    const double distY = abs(normal.dot(vector3::UnitY()));
    const double distZ = abs(normal.dot(vector3::UnitZ()));

    const double res = std::min(distX, std::min(distY, distZ));
    if (double_equal(res, distX, 0.1))
        return vector3::UnitX();
    if (double_equal(res, distY, 0.1))
        return vector3::UnitY();
    if (double_equal(res, distZ, 0.1))
        return vector3::UnitZ();

    // return a random variation of the normal
    outputs::log_error("Could not find the furthest base vector");
    return vector3(normal.z(), normal.x(), normal.y()).normalized();
}

/**
 * \brief Compute the two vectors that span the plane
 * \return a pair of vector u and v, normal to the plane normal
 */
std::pair<vector3, vector3> get_plane_coordinate_system(const vector3& normal) noexcept
{
    assert(double_equal(normal.norm(), 1.0));

    // define a vector orthogonal to the normal (r.dot normal should be close to 0)
    const vector3& r = select_correct_transform(normal);
    assert(double_equal(r.norm(), 1.0));

    // get two vectors that will span the plane
    const vector3 xAxis = normal.cross(r).normalized();
    const vector3 yAxis = normal.cross(xAxis).normalized();

    // check that angles between vectors is close to 0
    assert(double_equal(xAxis.norm(), 1.0));
    assert(double_equal(yAxis.norm(), 1.0));
    assert(abs(xAxis.dot(normal)) <= .01);
    assert(abs(yAxis.dot(xAxis)) <= .01);
    assert(abs(yAxis.dot(normal)) <= .01);

    return std::make_pair(xAxis, yAxis);
}

/**
 * \brief Compute the position of a point in the plane coordinate system
 * \param[in] pointToProject The point to project to plane, in world coordinates
 * \param[in] planeCenter The center point of the plane
 * \param[in] xAxis The unit y vector of the plane, othogonal to the normal
 * \param[in] yAxis The unit x vector of the plane, othogonal to the normal and u
 * \return A 2D point corresponding to pointToProject, in plane coordinate system
 */
vector2 get_projected_plan_coordinates(const vector3& pointToProject,
                                       const vector3& planeCenter,
                                       const vector3& xAxis,
                                       const vector3& yAxis) noexcept
{
    assert(double_equal(xAxis.norm(), 1.0));
    assert(double_equal(yAxis.norm(), 1.0));
    assert(abs(xAxis.dot(yAxis)) <= .01);

    const vector3& reducedPoint = pointToProject - planeCenter;
    return vector2(xAxis.dot(reducedPoint), yAxis.dot(reducedPoint));
}

/**
 * \brief Compute the projection of a point from the plane coordinate system to world
 * \param[in] pointToProject The point to project to world, in plane coordinates
 * \param[in] planeCenter The center point of the plane
 * \param[in] xAxis The unit x vector of the plane, othogonal to the normal
 * \param[in] yAxis The unit y vector of the plane, othogonal to the normal and u
 * \return A 3D point corresponding to pointToProject, in world coordinate system
 */
vector3 get_point_from_plane_coordinates(const vector2& pointToProject,
                                         const vector3& planeCenter,
                                         const vector3& xAxis,
                                         const vector3& yAxis) noexcept
{
    assert(double_equal(xAxis.norm(), 1.0));
    assert(double_equal(yAxis.norm(), 1.0));
    assert(abs(xAxis.dot(yAxis)) <= .01);

    return planeCenter + pointToProject.x() * xAxis + pointToProject.y() * yAxis;
}

Polygon::polygon get_static_screen_boundary_polygon() noexcept
{
    // define a polygon that span the screen space
    static const uint screenSizeX = Parameters::get_camera_1_image_size_x();
    static const uint screenSizeY = Parameters::get_camera_1_image_size_y();
    static const std::array<Polygon::point_2d, 5> screenBoundaryPoints(
            {Polygon::point_2d(0, 0),
             Polygon::point_2d(static_cast<int>(round(screenSizeX)), 0),
             Polygon::point_2d(static_cast<int>(round(screenSizeX)), static_cast<int>(round(screenSizeY))),
             Polygon::point_2d(0, static_cast<int>(round(screenSizeY))),
             Polygon::point_2d(0, 0)});
    static Polygon::polygon boundary;
    if (boundary.outer().size() <= 0)
    {
        boost::geometry::assign_points(boundary, screenBoundaryPoints);
        boost::geometry::correct(boundary);
    }
    return boundary;
}

Polygon::Polygon(const std::vector<vector3>& points, const vector3& normal, const vector3& center) : _center(center)
{
    assert(double_equal(normal.norm(), 1.0));
    assert(points.size() >= 3);

    // find the polygon axis
    const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(normal);
    _xAxis = res.first;
    _yAxis = res.second;

    // project to polygon space
    std::vector<vector2> boundaryPoints;
    boundaryPoints.reserve(points.size());
    std::ranges::transform(
            points.rbegin(), points.rend(), std::back_inserter(boundaryPoints), [this](const vector3& point) {
                return utils::get_projected_plan_coordinates(point, this->_center, this->_xAxis, this->_yAxis);
            });

    // compute boundary
    _polygon = utils::Polygon::compute_concave_hull(boundaryPoints);
    if (!is_valid())
    {
        // try to correct it using special third party method (bad solution to a bad problem)
        multi_polygon result;
        geometry::correct(_polygon, result);
        if (result.empty())
        {
            outputs::log_warning("Invalid concave polygon fit detected, trying to use convex hull algorithm");
            _polygon = utils::Polygon::compute_convex_hull(boundaryPoints);

            if (!is_valid())
            {
                outputs::log_error("Invalid convex and concave polygon fit detected, fitting failed");
            }
        }
        else
        {
            // TODO: should select the greatest area ?
            _polygon = result[0];

            if (!is_valid())
            {
                outputs::log_error("Invalid concave polygon fit after correction step, fitting failed");
            }
        }
    }

    _area = area();

    // simplify the input mesh
    simplify();
}

Polygon::Polygon(const Polygon& otherPolygon, const vector3& normal, const vector3& center)
{
    *this = otherPolygon.project(normal, center);
}

Polygon::Polygon(const std::vector<point_2d>& boundaryPoints,
                 const vector3& xAxis,
                 const vector3& yAxis,
                 const vector3& center) :
    _center(center),
    _xAxis(xAxis),
    _yAxis(yAxis)
{
    assert(double_equal(_xAxis.norm(), 1.0));
    assert(double_equal(_yAxis.norm(), 1.0));
    assert(abs(_xAxis.dot(_yAxis)) <= .01);

    // set boundary in reverse (clockwise)
    boost::geometry::assign_points(_polygon, boundaryPoints);
    boost::geometry::correct(_polygon);
    _area = area();

    if (_polygon.outer().size() <= 3)
    {
        outputs::log_warning("Polygon does not contain any edges after correction");
    }
}

Polygon::polygon Polygon::compute_convex_hull(const std::vector<vector2>& pointsIn) noexcept
{
    boost::geometry::model::multi_point<point_2d> hull;
    boost::geometry::model::multi_point<point_2d> input;

    input.reserve(pointsIn.size());
    std::ranges::transform(pointsIn.rbegin(), pointsIn.rend(), std::back_inserter(input), [](const vector2& point) {
        return point_2d(point.x(), point.y());
    });

    polygon pol;
    boost::geometry::convex_hull(input, pol);
    return pol;
}

Polygon::polygon Polygon::compute_concave_hull(const std::vector<vector2>& pointsIn) noexcept
{
    polygon poly;
    if (pointsIn.size() < 3)
    {
        outputs::log_warning("Cannot compute a polygon with less than 3 sides");
        return poly;
    }

    ::polygon::PointVector newPointVector;
    newPointVector.reserve(pointsIn.size());

    uint64_t id = 0;
    for (const auto& point: pointsIn)
    {
        newPointVector.emplace_back(point.x(), point.y());
        newPointVector.back().id = id++;
    }

    ::polygon::PointVector resultBoundary;
    // each iteration takes more time than the last, so do not increase this too much
    if (not ::polygon::compute_concave_hull(newPointVector, resultBoundary, 8))
    {
        // empty vector, failure
        return poly;
    }

    std::vector<point_2d> boundary;
    boundary.reserve(resultBoundary.size());
    std::ranges::transform(resultBoundary, std::back_inserter(boundary), [](const ::polygon::Point& point) {
        return boost::geometry::make<point_2d>(point.x, point.y);
    });

    boost::geometry::assign_points(poly, boundary);
    return poly;
}

bool Polygon::contains(const vector2& point) const noexcept
{
    return boost::geometry::within(boost::geometry::make<point_2d>(point.x(), point.y()), _polygon);
}

void Polygon::merge_union(const Polygon& other) noexcept
{
    const polygon& res = union_one(other.project(_xAxis, _yAxis, _center));
    if (res.outer().empty())
    {
        outputs::log_warning("Merge of two polygons produces no overlaps, returning without merge operation");
        return;
    }
    _polygon = res;
    // simplify the final mesh
    simplify();
}

Polygon Polygon::project(const vector3& nextNormal, const vector3& nextCenter) const noexcept
{
    assert(double_equal(nextNormal.norm(), 1.0));
    const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(nextNormal);

    return project(res.first, res.second, nextCenter);
}

Polygon Polygon::project(const vector3& nextXAxis, const vector3& nextYAxis, const vector3& nextCenter) const noexcept
{
    assert(double_equal(nextXAxis.norm(), 1.0));
    assert(double_equal(nextYAxis.norm(), 1.0));
    assert(abs(nextXAxis.dot(nextYAxis)) <= .01);

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

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

    return Polygon(newBoundary, nextXAxis, nextYAxis, nextCenter);
}

Polygon Polygon::transform(const vector3& nextNormal, const vector3& nextCenter) const noexcept
{
    assert(double_equal(nextNormal.norm(), 1.0));
    const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(nextNormal);
    const vector3& nextXAxis = res.first;
    const vector3& nextYAxis = res.second;

    assert(double_equal(nextXAxis.norm(), 1.0));
    assert(double_equal(nextYAxis.norm(), 1.0));
    assert(abs(nextXAxis.dot(nextYAxis)) <= .01);

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

    return transform(nextXAxis, nextYAxis, nextCenter);
}

Polygon Polygon::transform(const vector3& nextXAxis, const vector3& nextYAxis, const vector3& nextCenter) const noexcept
{
    assert(double_equal(nextXAxis.norm(), 1.0));
    assert(double_equal(nextYAxis.norm(), 1.0));
    assert(abs(nextXAxis.dot(nextYAxis)) <= .01);

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

    // compute transformation matrix between the two spaces
    // TODO: the coordinates system is XZY for the world, and should be changed to XYZ for this to work
    const matrix44& transfoMatrix =
            utils::get_transformation_matrix(_xAxis, _yAxis, _center, nextXAxis, nextYAxis, nextCenter);
    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(transfoMatrix, nextXAxis, nextYAxis, nextCenter);

    // compute new polygon
    return Polygon(newBoundary, nextXAxis, nextYAxis, nextCenter);
}

std::vector<Polygon::point_2d> Polygon::transform_boundary(const matrix44& transformationMatrix,
                                                           const vector3& nextXAxis,
                                                           const vector3& nextYAxis,
                                                           const vector3& nextCenter) const noexcept
{
    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    // transform each boundary points
    for (const auto& p: _polygon.outer())
    {
        const vector3& retroProjected =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);

        const vector3& transformed = (transformationMatrix * retroProjected.homogeneous()).head<3>();
        const vector2& projected = utils::get_projected_plan_coordinates(transformed, nextCenter, nextXAxis, nextYAxis);
        newBoundary.emplace_back(projected.x(), projected.y());
    }

    return newBoundary;
}

double Polygon::area() const noexcept
{
    if (_polygon.outer().size() < 3)
    {
        outputs::log_error("Cannot compute the area of an empty polygon");
        return 0;
    }
    return boost::geometry::area(_polygon);
}

Polygon::polygon Polygon::union_one(const Polygon& other) const noexcept
{
    multi_polygon res;
    boost::geometry::union_(_polygon, other.project(_xAxis, _yAxis, _center)._polygon, res);
    if (res.empty())
        return polygon(); // empty polygon

    // only one residual, return it
    if (res.size() == 1)
        return res.front();

    // TODO: find a better way: create hole ?
    // more than one, return the biggest overlap
    double biggestArea = 0;
    polygon& biggestPol = res.front();
    for (const polygon& p: res)
    {
        const double pArea = boost::geometry::area(p);
        if (pArea > biggestArea)
        {
            biggestArea = pArea;
            biggestPol = p;
        }
    }
    assert(biggestArea > 0);
    return biggestPol;
}

Polygon::polygon Polygon::inter_one(const Polygon& other) const noexcept
{
    multi_polygon res;
    boost::geometry::intersection(_polygon, other.project(_xAxis, _yAxis, _center)._polygon, res);
    if (res.empty())
        return polygon(); // empty polygon, no intersection

    // only one residual, return it
    if (res.size() == 1)
        return res.front();

    // more than one, return the biggest overlap
    double biggestArea = 0;
    polygon& biggestPol = res.front();
    for (const polygon& p: res)
    {
        const double pArea = boost::geometry::area(p);
        if (pArea > biggestArea)
        {
            biggestArea = pArea;
            biggestPol = p;
        }
    }
    assert(biggestArea > 0);
    return biggestPol;
}

double Polygon::inter_over_union(const Polygon& other) const noexcept
{
    const Polygon& projectedOther = other.project(_xAxis, _yAxis, _center);
    const polygon& un = union_one(projectedOther);
    if (un.outer().size() < 3)
        return 0.0;

    const double finalUnion = boost::geometry::area(un);
    if (finalUnion <= 0)
        return 0.0;

    const double finalInter = boost::geometry::area(inter_one(projectedOther));
    if (finalInter <= 0)
        return 0.0;
    return finalInter / finalUnion;
}

double Polygon::inter_area(const Polygon& other) const noexcept
{
    multi_polygon res;
    const bool processSuccess =
            boost::geometry::intersection(_polygon, other.project(_xAxis, _yAxis, _center)._polygon, res);
    if (!processSuccess)
    {
        outputs::log_error("Polygon intersection returned error");
        return 0;
    }

    // compute the sum of area of the inter
    double areaSum = 0;
    for (const polygon& p: res)
    {
        const double pArea = boost::geometry::area(p);
        areaSum += pArea;
    }
    return areaSum;
}

double Polygon::union_area(const Polygon& other) const noexcept
{
    multi_polygon res;
    boost::geometry::union_(_polygon, other.project(_xAxis, _yAxis, _center)._polygon, res);

    // compute the sum of area of the union
    double areaSum = 0;
    for (const polygon& p: res)
    {
        const double pArea = boost::geometry::area(p);
        areaSum += pArea;
    }
    return areaSum;
}

void Polygon::simplify(const double distanceThreshold) noexcept
{
    // pre compute the area
    _area = area();
    // compute a simplification distance threshold: it depends on the area.
    // the bigger the area, the higer the threshold. A lower limit is fixed by the distanceThreshold parameter
    const double distanceThres = std::max(_area / 1e5, distanceThreshold);

    // use temporary object to prevent segfault
    polygon out;
    boost::geometry::simplify(_polygon, out, distanceThres);

    if (boost::geometry::is_valid(out))
    {
        const double newArea = boost::geometry::area(out);
        // check that the area is not too reduced
        if (newArea > _area * 0.75)
        {
            _area = newArea;
            _polygon = out;
        }
    }
    // else: Could not optimize polygon boundary cause it would have been reduced to a non shape
    // dont change the polygon
}

std::vector<vector3> Polygon::get_unprojected_boundary() const noexcept
{
    std::vector<vector3> projectedBoundary;
    projectedBoundary.reserve(_polygon.outer().size());

    for (const auto& p: _polygon.outer())
    {
        const vector3& retroProjected =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);
        projectedBoundary.emplace_back(retroProjected);
    }

    return projectedBoundary;
}

/**
 *
 * CAMERA POLYGON
 *
 */

void CameraPolygon::display(const cv::Scalar& color, cv::Mat& debugImage) const noexcept
{
    ScreenCoordinate previousPoint;
    bool isPreviousPointSet = false;
    for (const ScreenCoordinate& screenPoint: get_screen_points())
    {
        if (isPreviousPointSet and previousPoint.z() > 0 and screenPoint.z() > 0)
        {
            cv::line(debugImage,
                     cv::Point(static_cast<int>(previousPoint.x()), static_cast<int>(previousPoint.y())),
                     cv::Point(static_cast<int>(screenPoint.x()), static_cast<int>(screenPoint.y())),
                     color,
                     2);
        }

        // set the previous point for the new line
        previousPoint = screenPoint;
        isPreviousPointSet = true;
    }
}

WorldPolygon CameraPolygon::to_world_space(const CameraToWorldMatrix& cameraToWorld) const noexcept
{
    const WorldCoordinate& newCenter = CameraCoordinate(_center).to_world_coordinates(cameraToWorld);

    // rotate axis
    const matrix33& rotationMatrix = cameraToWorld.block<3, 3>(0, 0);
    const vector3 newXAxis = (rotationMatrix * _xAxis).normalized();
    const vector3 newYAxis = (rotationMatrix * _yAxis).normalized();

    assert(double_equal(newXAxis.norm(), 1.0));
    assert(double_equal(newYAxis.norm(), 1.0));
    assert(abs(newXAxis.dot(newYAxis)) <= .01);

    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(cameraToWorld, newXAxis, newYAxis, newCenter);

    // compute new polygon
    return WorldPolygon(newBoundary, newXAxis, newYAxis, newCenter);
}

std::vector<ScreenCoordinate> CameraPolygon::get_screen_points() const noexcept
{
    std::vector<ScreenCoordinate> screenBoundary;
    screenBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const CameraCoordinate& projectedPoint =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);

        ScreenCoordinate screenpoint;
        if (not projectedPoint.to_screen_coordinates(screenpoint))
        {
            // happens only if the projection center z coordinate is 0
            outputs::log_warning("Could not transform polygon boundary to screen coordinates");
            return std::vector<ScreenCoordinate>();
        }
        else
        {
            screenBoundary.emplace_back(screenpoint);
        }
    }
    return screenBoundary;
}

Polygon::polygon CameraPolygon::to_screen_space() const noexcept
{
    const auto& t = get_screen_points();

    std::vector<point_2d> boundary;
    boundary.reserve(t.size());

    // convert to point_2d vector, inneficient but rare use
    std::ranges::transform(t.cbegin(), t.cend(), std::back_inserter(boundary), [](const ScreenCoordinate& s) {
        return boost::geometry::make<point_2d>(s.x(), s.y());
    });

    polygon pol;
    boost::geometry::assign_points(pol, boundary);
    boost::geometry::correct(pol);
    return pol;
}

bool CameraPolygon::is_visible_in_screen_space() const noexcept
{
    // intersecton of this polygon in screen space, and the screen limits; if it exists, the polygon is visible
    multi_polygon res;
    // TODO: can this be a problem  when the polygon is behind the camera ?
    boost::geometry::intersection(to_screen_space(), get_static_screen_boundary_polygon(), res);
    return not res.empty(); // intersection exists, polygon is visible
}

/**
 *
 * WORLD POLYGON
 *
 */

CameraPolygon WorldPolygon::to_camera_space(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    const CameraCoordinate& newCenter = WorldCoordinate(_center).to_camera_coordinates(worldToCamera);

    // rotate axis
    const matrix33& rotationMatrix = worldToCamera.block<3, 3>(0, 0);
    const vector3 newXAxis = (rotationMatrix * _xAxis).normalized();
    const vector3 newYAxis = (rotationMatrix * _yAxis).normalized();

    assert(double_equal(newXAxis.norm(), 1.0));
    assert(double_equal(newYAxis.norm(), 1.0));
    assert(abs(newXAxis.dot(newYAxis)) <= .01);

    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(worldToCamera, newXAxis, newYAxis, newCenter);

    // compute new polygon
    return CameraPolygon(newBoundary, newXAxis, newYAxis, newCenter);
}

void WorldPolygon::merge(const WorldPolygon& other) noexcept
{
    Polygon::merge_union(other.Polygon::project(_xAxis, _yAxis, _center));
    // no need to correct to polygon
};

void WorldPolygon::display(const WorldToCameraMatrix& worldToCamera,
                           const cv::Scalar& color,
                           cv::Mat& debugImage) const noexcept
{
    this->to_camera_space(worldToCamera).display(color, debugImage);
}

} // namespace rgbd_slam::utils
