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
                                             const double& maximumRetroprojectionError) noexcept
{
    utils::ScreenCoordinate2D projectedScreenPoint;
    if (not worldPoint.to_screen_coordinates(worldToCamera, projectedScreenPoint))
    {
        return false;
    }
    const double retroprojectionError = (screenPoint - projectedScreenPoint).norm();

    // true if retroprojection error is small enough
    return (retroprojectionError > maximumRetroprojectionError);
}

bool Triangulation::triangulate(const WorldToCameraMatrix& currentWorldToCamera,
                                const WorldToCameraMatrix& newWorldToCamera,
                                const utils::ScreenCoordinate2D& point2Da,
                                const utils::ScreenCoordinate2D& newPoint2Db,
                                utils::WorldCoordinate& triangulatedPoint) noexcept
{
    constexpr double maximumRetroprojectionError = parameters::optimization::maximumRetroprojectionError;

    // project x and y coordinates
    const utils::CameraCoordinate2D& pointA = point2Da.to_camera_coordinates();
    const utils::CameraCoordinate2D& pointB = newPoint2Db.to_camera_coordinates();

    // Linear-LS triangulation
    matrix44 triangulationMatrix;
    triangulationMatrix << pointA.x() * currentWorldToCamera.row(2) - currentWorldToCamera.row(0),
            pointA.y() * currentWorldToCamera.row(2) - currentWorldToCamera.row(1),
            pointB.x() * newWorldToCamera.row(2) - newWorldToCamera.row(0),
            pointB.y() * newWorldToCamera.row(2) - newWorldToCamera.row(1);

    // singular value decomposition
    const utils::WorldCoordinate worldPoint(triangulationMatrix.leftCols<3>()
                                                    .jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV)
                                                    .solve(-triangulationMatrix.col(3)));

    if (std::isfinite(worldPoint.x()) and std::isfinite(worldPoint.y()) and std::isfinite(worldPoint.z()))
    {
        // We have a good triangulation ! Maybe not good enough but still usable
        triangulatedPoint = worldPoint;

        // Check retroprojection of point in frame A
        if (not Triangulation::is_retroprojection_valid(
                    worldPoint, point2Da, currentWorldToCamera, maximumRetroprojectionError))
            return false;

        // Check retroprojection of point in frame B
        if (not Triangulation::is_retroprojection_valid(
                    worldPoint, newPoint2Db, newWorldToCamera, maximumRetroprojectionError))
            return false;

        // retroprojection is good enough
        return true;
    }
    // invalid triangulation
    return false;
}

} // namespace rgbd_slam::tracking
