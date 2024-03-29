#include "polygon.hpp"
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
#include <stdexcept>
#include "logger.hpp"
#include "concave_fitting.hpp"
#include "correct_boost_polygon.hpp"
#include <format>

#include "coordinates/point_coordinates.hpp"

namespace rgbd_slam::utils {

Polygon::polygon get_static_screen_boundary_polygon() noexcept
{
    // prevent floatting number problems
    static constexpr uint boundarySize = 1;

    // define a polygon that span the screen space
    static const uint screenSizeX = Parameters::get_camera_1_image_size().x() - boundarySize;
    static const uint screenSizeY = Parameters::get_camera_1_image_size().y() - boundarySize;
    static const std::array<Polygon::point_2d, 5> screenBoundaryPoints(
            {Polygon::point_2d(boundarySize, boundarySize),
             Polygon::point_2d(static_cast<int>(round(screenSizeX)), boundarySize),
             Polygon::point_2d(static_cast<int>(round(screenSizeX)), static_cast<int>(round(screenSizeY))),
             Polygon::point_2d(boundarySize, static_cast<int>(round(screenSizeY))),
             Polygon::point_2d(boundarySize, boundarySize)});

    static Polygon::polygon boundary;
    if (boundary.outer().size() <= 0)
    {
        boost::geometry::assign_points(boundary, screenBoundaryPoints);
        boost::geometry::correct(boundary);
    }
    return boundary;
}

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
std::pair<vector3, vector3> get_plane_coordinate_system(const vector3& normal)
{
    if (not double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument("get_plane_coordinate_system: The normal should have a norm of 1");
    }

    // define a vector orthogonal to the normal (r.dot normal should be close to 0)
    const vector3& r = select_correct_transform(normal);
    if (not double_equal(r.norm(), 1.0))
    {
        throw std::logic_error("get_plane_coordinate_system: The selected vector r should have a norm of 1");
    }

    // get two vectors that will span the plane
    const vector3 xAxis = normal.cross(r).normalized();
    const vector3 yAxis = normal.cross(xAxis).normalized();

    // check that angles between vectors is close to 0
    if (not double_equal(xAxis.norm(), 1.0))
    {
        throw std::logic_error("get_plane_coordinate_system: The x axis as an invalid norm");
    }
    if (not double_equal(yAxis.norm(), 1.0))
    {
        throw std::logic_error("get_plane_coordinate_system: The y axis as an invalid norm");
    }
    if (abs(xAxis.dot(normal)) > .01)
    {
        throw std::logic_error("get_plane_coordinate_system: The x axis and normal are not orthogonals");
    }
    if (abs(yAxis.dot(xAxis)) > .01)
    {
        throw std::logic_error("get_plane_coordinate_system: The y axis and x axis are not orthogonals");
    }
    if (abs(yAxis.dot(normal)) > .01)
    {
        throw std::logic_error("get_plane_coordinate_system: The y axis and normal are not orthogonals");
    }

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
                                       const vector3& yAxis)
{
    if (not double_equal(xAxis.norm(), 1.0))
    {
        throw std::invalid_argument("get_projected_plan_coordinates: xAxis norm should be 1");
    }
    if (not double_equal(yAxis.norm(), 1.0))
    {
        throw std::invalid_argument("get_projected_plan_coordinates: yAxis norm should be 1");
    }
    if (abs(yAxis.dot(xAxis)) > .01)
    {
        throw std::invalid_argument("get_projected_plan_coordinates: yAxis and xAxis should be orthogonals");
    }

    const vector3& reducedPoint = pointToProject - planeCenter;
    return vector2(xAxis.dot(reducedPoint), yAxis.dot(reducedPoint));
}

vector3 get_point_from_plane_coordinates(const vector2& pointToProject,
                                         const vector3& planeCenter,
                                         const vector3& xAxis,
                                         const vector3& yAxis)
{
    if (not double_equal(xAxis.norm(), 1.0))
    {
        throw std::invalid_argument("get_point_from_plane_coordinates: xAxis norm should be 1");
    }
    if (not double_equal(yAxis.norm(), 1.0))
    {
        throw std::invalid_argument("get_point_from_plane_coordinates: yAxis norm should be 1");
    }
    if (abs(yAxis.dot(xAxis)) > .01)
    {
        throw std::invalid_argument("get_point_from_plane_coordinates: yAxis and xAxis should be orthogonals");
    }

    return planeCenter + pointToProject.x() * xAxis + pointToProject.y() * yAxis;
}

Polygon::Polygon(const std::vector<vector3>& points, const vector3& normal, const vector3& center) : _center(center)
{
    if (not double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon: normal norm should be 1");
    }
    if (points.size() < 3)
    {
        throw std::invalid_argument("Polygon: need at least 3 points to fit a polygon");
    }

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

            std::string failureReason;
            if (!is_valid(failureReason))
            {
                outputs::log_error(std::format(
                        "Invalid convex and concave polygon fit detected, fitting failed. reason {}", failureReason));
            }
        }
        else
        {
            // TODO: should select the greatest area ?
            _polygon = result[0];

            std::string failureReason;
            if (!is_valid(failureReason))
            {
                outputs::log_error(std::format(
                        "Invalid concave polygon fit after correction step, fitting failed. reason {}", failureReason));
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
    if (not double_equal(_xAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon: _xAxis norm should be 1");
    }
    if (not double_equal(_yAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon: _yAxis norm should be 1");
    }
    if (abs(_yAxis.dot(_xAxis)) > .01)
    {
        throw std::invalid_argument("Polygon: _yAxis and _xAxis should be orthogonals");
    }

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
    for (const vector2& point: pointsIn)
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

void Polygon::merge_union(const Polygon& other)
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

Polygon Polygon::project(const vector3& nextNormal, const vector3& nextCenter) const
{
    if (not double_equal(nextNormal.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::project: nextNormal norm should be 1");
    }
    const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(nextNormal);

    return project(res.first, res.second, nextCenter);
}

Polygon Polygon::project(const vector3& nextXAxis, const vector3& nextYAxis, const vector3& nextCenter) const
{
    if (not double_equal(nextXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::project: nextXAxis norm should be 1");
    }
    if (not double_equal(nextXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::project: nextYAxis norm should be 1");
    }
    if (abs(nextXAxis.dot(nextYAxis)) > .01)
    {
        throw std::invalid_argument("Polygon::project: nextXAxis and nextYAxis should be orthogonals");
    }

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const vector3& retroProjected =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);

        const vector2& projected =
                utils::get_projected_plan_coordinates(retroProjected, nextCenter, nextXAxis, nextYAxis);
        newBoundary.emplace_back(projected.x(), projected.y());
    }

    return Polygon(newBoundary, nextXAxis, nextYAxis, nextCenter);
}

Polygon Polygon::transform(const vector3& nextNormal, const vector3& nextCenter) const
{
    if (not double_equal(nextNormal.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::transform: nextNormal norm should be 1");
    }

    const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(nextNormal);
    const vector3& nextXAxis = res.first;
    const vector3& nextYAxis = res.second;

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

    return transform(nextXAxis, nextYAxis, nextCenter);
}

Polygon Polygon::transform(const vector3& nextXAxis, const vector3& nextYAxis, const vector3& nextCenter) const
{
    if (not double_equal(nextXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::transform: nextXAxis norm should be 1");
    }
    if (not double_equal(nextXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::transform: nextYAxis norm should be 1");
    }
    if (abs(nextXAxis.dot(nextYAxis)) > .01)
    {
        throw std::invalid_argument("Polygon::transform: nextXAxis and nextYAxis should be orthogonals");
    }

    // if the projection is the same as this one, do not project
    if (_center.isApprox(nextCenter) and _xAxis.isApprox(nextXAxis) and _yAxis.isApprox(nextYAxis))
        return *this;

    // compute transformation matrix between the two spaces
    const matrix44& transfoMatrix =
            get_transformation_matrix(_xAxis, _yAxis, _center, nextXAxis, nextYAxis, nextCenter);
    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(transfoMatrix, nextXAxis, nextYAxis, nextCenter);

    // compute new polygon
    return Polygon(newBoundary, nextXAxis, nextYAxis, nextCenter);
}

std::vector<Polygon::point_2d> Polygon::transform_boundary(const matrix44& transformationMatrix,
                                                           const vector3& nextXAxis,
                                                           const vector3& nextYAxis,
                                                           const vector3& nextCenter) const
{
    std::vector<point_2d> newBoundary;
    newBoundary.reserve(_polygon.outer().size());

    // transform each boundary points
    for (const point_2d& p: _polygon.outer())
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

Polygon::polygon Polygon::union_one(const Polygon& other) const
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

    if (biggestArea <= 0)
    {
        throw std::logic_error("Polygon::union_one: biggestArea is <= 0");
    }
    return biggestPol;
}

Polygon::polygon Polygon::inter_one(const Polygon& other) const
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
    if (biggestArea <= 0)
    {
        throw std::logic_error("Polygon::inter_one: biggestArea is <= 0");
    }
    return biggestPol;
}

double Polygon::inter_over_union(const Polygon& other) const
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

double Polygon::inter_area(const Polygon& other) const
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

double Polygon::union_area(const Polygon& other) const
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

std::vector<vector3> Polygon::get_unprojected_boundary() const
{
    std::vector<vector3> projectedBoundary;
    projectedBoundary.reserve(_polygon.outer().size());

    for (const point_2d& p: _polygon.outer())
    {
        const vector3& retroProjected =
                utils::get_point_from_plane_coordinates(vector2(p.x(), p.y()), _center, _xAxis, _yAxis);
        projectedBoundary.emplace_back(retroProjected);
    }

    return projectedBoundary;
}

} // namespace rgbd_slam::utils