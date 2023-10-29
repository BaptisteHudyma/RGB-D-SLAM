#include "triangulation.hpp"
#include "../parameters.hpp"

namespace rgbd_slam::tracking {

utils::Pose Triangulation::get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition) noexcept
{
    const vector3 pt(baselinePoseSupposition, 0, 0);
    const vector3 newPosition = (pose.get_orientation_matrix() * pt) + pose.get_position();
    return utils::Pose(newPosition, pose.get_orientation_quaternion());
}

bool Triangulation::is_retroprojection_valid(const utils::WorldCoordinate& worldPoint,
                                             const utils::ScreenCoordinate2D& screenPoint,
                                             const WorldToCameraMatrix& worldToCamera,
                                             const double maximumRetroprojectionErrorSqr_px) noexcept
{
    utils::ScreenCoordinate projectedScreenPoint;
    if (worldPoint.to_screen_coordinates(worldToCamera, projectedScreenPoint) and
        projectedScreenPoint.is_in_screen_boundaries() and projectedScreenPoint.z() > 0)
    {
        const double retroprojectionError = (screenPoint - projectedScreenPoint.get_2D()).squaredNorm();
        // true if retroprojection error is small enough
        return (retroprojectionError < maximumRetroprojectionErrorSqr_px);
    }
    return false;
}

bool Triangulation::triangulate(const WorldToCameraMatrix& currentWorldToCamera,
                                const WorldToCameraMatrix& newWorldToCamera,
                                const utils::ScreenCoordinate2D& point2Da,
                                const utils::ScreenCoordinate2D& newPoint2Db,
                                utils::WorldCoordinate& triangulatedPoint) noexcept
{
    // project x and y coordinates
    const utils::CameraCoordinate2D& pointA = point2Da.to_camera_coordinates();
    const utils::CameraCoordinate2D& pointB = newPoint2Db.to_camera_coordinates();

    // Linear-LS triangulation
    matrix44 triangulationMatrix;
    triangulationMatrix.row(0) = pointA.x() * currentWorldToCamera.row(2) - currentWorldToCamera.row(0);
    triangulationMatrix.row(1) = pointA.y() * currentWorldToCamera.row(2) - currentWorldToCamera.row(1);
    triangulationMatrix.row(2) = pointB.x() * newWorldToCamera.row(2) - newWorldToCamera.row(0);
    triangulationMatrix.row(3) = pointB.y() * newWorldToCamera.row(2) - newWorldToCamera.row(1);

    // this happens but I have no idea why
    if (triangulationMatrix.hasNaN())
    {
        return false;
    }

    // singular value decomposition
    const utils::WorldCoordinate worldPoint = triangulationMatrix.leftCols<3>()
                                                      .jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV)
                                                      .solve(-triangulationMatrix.col(3));

    // Check retroprojection of point in frame A
    if (not worldPoint.hasNaN() and std::isfinite(worldPoint.x()) and std::isfinite(worldPoint.y()) and
        std::isfinite(worldPoint.z()) and
        Triangulation::is_retroprojection_valid(
                worldPoint,
                point2Da,
                currentWorldToCamera,
                parameters::mapping::maximumRetroprojectionErrorForTriangulatePow_px) and
        Triangulation::is_retroprojection_valid(worldPoint,
                                                newPoint2Db,
                                                newWorldToCamera,
                                                parameters::mapping::maximumRetroprojectionErrorForTriangulatePow_px))
    {
        // We have a good triangulation ! Maybe not good enough but still usable
        triangulatedPoint = worldPoint;
        return true;
    }
    // invalid triangulation
    return false;
}

} // namespace rgbd_slam::tracking
