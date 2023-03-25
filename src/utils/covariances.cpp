#include "covariances.hpp"
#include "../parameters.hpp"
#include "camera_transformation.hpp"
#include "types.hpp"

namespace rgbd_slam {
namespace utils {

double get_depth_quantization(const double depth)
{
    // minimum depth diparity at z is the quadratic function  a + b z + c z^2
    const static double depthSigmaError = Parameters::get_depth_sigma_error() * pow(1.0 / 1000.0, 2.0);
    const static double depthSigmaMultiplier = Parameters::get_depth_sigma_multiplier() / 1000.0;
    const static double depthSigmaMargin = Parameters::get_depth_sigma_margin();
    return std::max(depthSigmaMargin + depthSigmaMultiplier * depth + depthSigmaError * pow(depth, 2.0), 0.5);
}

const ScreenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance)
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();

    // Jacobian of the world to screen function. Use absolutes to prevent negative variances
    const matrix33 jacobian {{cameraFX / point.z(), 0.0, -cameraFX * point.x() / pow(point.z(), 2.0)},
                             {0.0, cameraFY / point.z(), -cameraFY * point.y() / pow(point.z(), 2.0)},
                             {0.0, 0.0, 1.0}};
    ScreenCoordinateCovariance screenPointCovariance;
    screenPointCovariance.base() = (jacobian * pointCovariance * jacobian.transpose());
    return screenPointCovariance;
}

const ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                             const WorldCoordinateCovariance& pointCovariance)
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

const ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                             const CameraCoordinateCovariance& pointCovariance)
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

const WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                           const matrix33& poseCovariance)
{
    WorldCoordinateCovariance cov;
    cov.base() << cameraPointCovariance + poseCovariance;
    return cov;
}

const WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                           const matrix33& poseCovariance)
{
    return get_world_point_covariance(utils::get_camera_point_covariance(screenPoint), poseCovariance);
}

const CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint)
{
    return get_camera_point_covariance(screenPoint, screenPoint.get_covariance());
}

const CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
                                                             const ScreenCoordinateCovariance& screenPointCovariance)
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    // Jacobian of the screen to camera function. Use absolutes to prevent negative variances
    const matrix33 jacobian {{screenPoint.z() / cameraFX, 0.0, abs(screenPoint.x() - cameraCX) / cameraFX},
                             {0.0, screenPoint.z() / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY},
                             {0.0, 0.0, 1.0}};

    CameraCoordinateCovariance cameraPointCovariance;
    cameraPointCovariance.base() = jacobian * screenPointCovariance * jacobian.transpose();
    return cameraPointCovariance;
}

} // namespace utils
} // namespace rgbd_slam
