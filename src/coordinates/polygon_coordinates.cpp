#include "polygon_coordinates.hpp"

#include "coordinates/point_coordinates.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include "logger.hpp"

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

namespace rgbd_slam {

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

WorldPolygon CameraPolygon::to_world_space(const CameraToWorldMatrix& cameraToWorld) const
{
    const WorldCoordinate& newCenter = CameraCoordinate(_center).to_world_coordinates(cameraToWorld);

    // rotate axis
    const matrix33& rotationMatrix = cameraToWorld.rotation();
    const vector3 newXAxis = (rotationMatrix * _xAxis).normalized();
    const vector3 newYAxis = (rotationMatrix * _yAxis).normalized();

    if (not utils::double_equal(newXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::to_world_space: newXAxis norm should be 1");
    }
    if (not utils::double_equal(newYAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::to_world_space: newYAxis norm should be 1");
    }
    if (abs(newYAxis.dot(newXAxis)) > .01)
    {
        throw std::invalid_argument("Polygon::to_world_space: newYAxis and newXAxis should be orthogonals");
    }

    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(cameraToWorld, newXAxis, newYAxis, newCenter);

    // compute new polygon
    return WorldPolygon(newBoundary, newXAxis, newYAxis, newCenter);
}

std::vector<ScreenCoordinate> CameraPolygon::get_screen_points() const
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

utils::Polygon::polygon CameraPolygon::to_screen_space() const
{
    const std::vector<ScreenCoordinate>& t = get_screen_points();

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

bool CameraPolygon::is_visible_in_screen_space() const
{
    // intersecton of this polygon in screen space, and the screen limits; if it exists, the polygon is visible
    multi_polygon res;
    // TODO: can this be a problem  when the polygon is behind the camera ?
    boost::geometry::intersection(to_screen_space(), utils::get_static_screen_boundary_polygon(), res);
    return not res.empty(); // intersection exists, polygon is visible
}

/**
 *
 * WORLD POLYGON
 *
 */

CameraPolygon WorldPolygon::to_camera_space(const WorldToCameraMatrix& worldToCamera) const
{
    const CameraCoordinate& newCenter = WorldCoordinate(_center).to_camera_coordinates(worldToCamera);

    // rotate axis
    const matrix33& rotationMatrix = worldToCamera.rotation();
    const vector3 newXAxis = (rotationMatrix * _xAxis).normalized();
    const vector3 newYAxis = (rotationMatrix * _yAxis).normalized();

    if (not utils::double_equal(newXAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::to_camera_space: newXAxis norm should be 1");
    }
    if (not utils::double_equal(newYAxis.norm(), 1.0))
    {
        throw std::invalid_argument("Polygon::to_camera_space: newYAxis norm should be 1");
    }
    if (abs(newYAxis.dot(newXAxis)) > .01)
    {
        throw std::invalid_argument("Polygon::to_camera_space: newYAxis and newXAxis should be orthogonals");
    }

    // project the boundary to the new space
    const std::vector<point_2d>& newBoundary = transform_boundary(worldToCamera, newXAxis, newYAxis, newCenter);

    // compute new polygon
    return CameraPolygon(newBoundary, newXAxis, newYAxis, newCenter);
}

void WorldPolygon::merge(const WorldPolygon& other)
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

} // namespace rgbd_slam
