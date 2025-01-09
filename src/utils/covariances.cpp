#include "covariances.hpp"

#include "../parameters.hpp"
#include "coordinates/point_coordinates.hpp"
#include "distance_utils.hpp"
#include "types.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <stdexcept>

namespace rgbd_slam::utils {

double get_depth_quantization(const double depth) noexcept
{
    // minimum depth diparity at z is the quadratic function  a + b z + c z^2
    const static double depthSigmaError = parameters::depthSigmaError * SQR(1.0 / 1000.0);
    constexpr double depthSigmaMultiplier = parameters::depthSigmaMultiplier / 1000.0;
    constexpr double depthSigmaMargin = parameters::depthSigmaMargin;
    return std::max(depthSigmaMargin + depthSigmaMultiplier * depth + depthSigmaError * SQR(depth), 0.5);
}

matrix23 get_camera_to_screen2d_jacobian(const CameraCoordinate& point)
{
    const static vector2 cameraF = Parameters::get_camera_1_focal();
    // Jacobian of the camera to screen function
    matrix23 jacobian {{cameraF.x() / point.z(), 0.0, -cameraF.x() * point.x() / SQR(point.z())},
                       {0.0, cameraF.y() / point.z(), -cameraF.y() * point.y() / SQR(point.z())}};
    return jacobian;
}

matrix33 get_camera_to_screen_jacobian(const CameraCoordinate& point)
{
    const matrix23& camToScreenJac = get_camera_to_screen2d_jacobian(point);
    matrix33 jacobian;
    jacobian << camToScreenJac, 0.0, 0.0, 1.0;
    return jacobian;
}

ScreenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance) noexcept
{
    // Jacobian of the camera to screen function
    const matrix33& jacobian = get_camera_to_screen_jacobian(point);

    ScreenCoordinateCovariance screenPointCovariance;
    screenPointCovariance << utils::propagate_covariance(pointCovariance, jacobian, 0.0);
    return screenPointCovariance;
}

ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                       const WorldCoordinateCovariance& pointCovariance,
                                                       const WorldToCameraMatrix& worldToCamera) noexcept
{
    return get_screen_point_covariance(point.to_camera_coordinates(worldToCamera),
                                       get_camera_point_covariance(pointCovariance, worldToCamera, matrix33::Zero()));
}

ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const CameraCoordinateCovariance& pointCovariance) noexcept
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

CameraCoordinateCovariance get_camera_point_covariance(const WorldCoordinateCovariance& worldPointCovariance,
                                                       const WorldToCameraMatrix& worldToCamera,
                                                       const matrix33& poseCovariance) noexcept
{
    const matrix33& rotation = worldToCamera.rotation();

    CameraCoordinateCovariance cov;
    cov << utils::propagate_covariance(worldPointCovariance, rotation, 0.0) + poseCovariance;
    return cov;
}

WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix33& poseCovariance) noexcept
{
    const matrix33& rotation = cameraToWorld.rotation();

    WorldCoordinateCovariance cov;
    cov << utils::propagate_covariance(cameraPointCovariance, rotation, 0.0) + poseCovariance;
    return cov;
}

WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix33& poseCovariance) noexcept
{
    return get_world_point_covariance(utils::get_camera_point_covariance(screenPoint), cameraToWorld, poseCovariance);
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint) noexcept
{
    return get_camera_point_covariance(screenPoint, screenPoint.get_covariance());
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
                                                       const ScreenCoordinateCovariance& screenPointCovariance) noexcept
{
    const static vector2 cameraF = Parameters::get_camera_1_focal();
    const static vector2 cameraC = Parameters::get_camera_1_center();

    // Jacobian of the screen to camera function. Use absolutes to prevent negative variances
    const matrix33 jacobian {{screenPoint.z() / cameraF.x(), 0.0, abs(screenPoint.x() - cameraC.x()) / cameraF.x()},
                             {0.0, screenPoint.z() / cameraF.y(), abs(screenPoint.y() - cameraC.y()) / cameraF.y()},
                             {0.0, 0.0, 1.0}};

    CameraCoordinateCovariance cameraPointCovariance;
    cameraPointCovariance << utils::propagate_covariance(screenPointCovariance, jacobian, 0.0);
    return cameraPointCovariance;
}

matrix44 compute_plane_covariance(const PlaneCoordinates& planeParameters, const matrix33& pointCloudCovariance)
{
    if (not is_covariance_valid(pointCloudCovariance))
    {
        throw std::invalid_argument(
                "compute_plane_covariance: the argument pointCloudCovariance is an invalid covariance matrix");
    }

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    if (utils::double_equal(d, 0.0))
    {
        throw std::invalid_argument("compute_plane_covariance: The d of planeParameters should not be 0");
    }
    if (not utils::double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument("compute_plane_covariance: The normal of planeParameters should have a norm of 1");
    }

    // reduce the parametrization
    const vector3 parameters = normal * d;
    const double a = parameters.x();
    const double b = parameters.y();
    const double c = parameters.z();

    const double aSquared = SQR(a);
    const double bSquared = SQR(b);
    const double cSquared = SQR(c);

    // common divider of all partial derivatives
    const double divider = pow(aSquared + bSquared + cSquared, 3.0 / 2.0);
    const double common = 1.0 / sqrt(aSquared + bSquared + cSquared);

    // compute the jacobian of the 3 parameter plane to 4 parameters plane transformation -> (vect, 1) / norm(vect)
    // with vect = normal * d
    const matrix43 jacobian({
            {common - aSquared / divider, -(a * b) / divider, -(a * c) / divider},
            {-(a * b) / divider, common - bSquared / divider, -(b * c) / divider},
            {-(a * c) / divider, -(b * c) / divider, common - cSquared / divider},
            {-a / divider, -b / divider, -c / divider},
    });

    const matrix44& planeParameterCovariance = utils::propagate_covariance(pointCloudCovariance, jacobian);
    std::string failureReason;
    if (not is_covariance_valid(planeParameterCovariance, failureReason))
    {
        throw std::logic_error(
                "compute_plane_covariance: planeParameterCovariance is an invalid covariance matrix after process:" +
                failureReason);
    }
    return planeParameterCovariance;
}

matrix33 compute_reduced_plane_point_cloud_covariance(const PlaneCoordinates& planeParameters,
                                                      const matrix44& planeCloudCovariance)
{
    if (not is_covariance_valid(planeCloudCovariance))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: planeCloudCovariance is an invalid covariance matrix");
    }

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    if (utils::double_equal(d, 0.0))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: compute_plane_covariance: The d of planeParameters "
                "should not be 0");
    }
    if (not utils::double_equal(normal.norm(), 1.0))
    {
        throw std::invalid_argument(
                "compute_reduced_plane_point_cloud_covariance: compute_plane_covariance: The normal of planeParameters "
                "should have a norm of 1");
    }

    // compute the jacobian of the 4 parameter plane to 3 parameters plane transformation -> vect = normal * d
    const matrix34 jacobian({
            {d, 0, 0, normal.x()},
            {0, d, 0, normal.y()},
            {0, 0, d, normal.z()},
    });

    const matrix33& pointCloudCovariance = utils::propagate_covariance(planeCloudCovariance, jacobian, 1e-2);
    if (not is_covariance_valid(pointCloudCovariance))
    {
        throw std::logic_error(
                "compute_reduced_plane_point_cloud_covariance: pointCloudCovariance is an invalid covariance matrix "
                "after process");
    }
    return pointCloudCovariance;
}

matrix44 get_world_plane_covariance(const PlaneCameraCoordinates& planeCoordinates,
                                    const CameraToWorldMatrix& cameraToWorldMatrix,
                                    const PlaneCameraToWorldMatrix& planeCameraToWorldMatrix,
                                    const matrix44& planeCovariance,
                                    const matrix33& worldPoseCovariance)
{
    if (not is_covariance_valid(planeCovariance))
    {
        throw std::invalid_argument("get_world_plane_covariance: planeCovariance is an invalid covariance matrix");
    }

    // transform to point form
    const matrix33& pointCloudCovariance =
            compute_reduced_plane_point_cloud_covariance(planeCoordinates, planeCovariance);

    // convert covariance to world
    const WorldCoordinateCovariance& pointCloudWorlCovariance = WorldCoordinateCovariance(
            utils::propagate_covariance(pointCloudCovariance, cameraToWorldMatrix.rotation(), 0.0) +
            worldPoseCovariance);

    std::string failureReason;
    if (not is_covariance_valid(pointCloudWorlCovariance, failureReason))
    {
        throw std::logic_error(
                "get_world_plane_covariance: pointCloudWorlCovariance is an invalid covariance matrix after process:" +
                failureReason);
    }
    // convert back to plane hessian form
    return compute_plane_covariance(planeCoordinates.to_world_coordinates(planeCameraToWorldMatrix),
                                    pointCloudWorlCovariance);
}

} // namespace rgbd_slam::utils