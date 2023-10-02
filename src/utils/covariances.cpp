#include "covariances.hpp"
#include "../parameters.hpp"
#include "camera_transformation.hpp"
#include "coordinates.hpp"
#include "distance_utils.hpp"
#include "logger.hpp"
#include "types.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <cmath>
#include <iostream>

namespace rgbd_slam::utils {

double get_depth_quantization(const double depth) noexcept
{
    // minimum depth diparity at z is the quadratic function  a + b z + c z^2
    const static double depthSigmaError = parameters::depthSigmaError * pow(1.0 / 1000.0, 2.0);
    constexpr double depthSigmaMultiplier = parameters::depthSigmaMultiplier / 1000.0;
    constexpr double depthSigmaMargin = parameters::depthSigmaMargin;
    return std::max(depthSigmaMargin + depthSigmaMultiplier * depth + depthSigmaError * pow(depth, 2.0), 0.5);
}

ScreenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance) noexcept
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();

    // Jacobian of the camera to screen function
    const matrix33 jacobian {{cameraFX / point.z(), 0.0, -cameraFX * point.x() / pow(point.z(), 2.0)},
                             {0.0, cameraFY / point.z(), -cameraFY * point.y() / pow(point.z(), 2.0)},
                             {0.0, 0.0, 1.0}};
    ScreenCoordinateCovariance screenPointCovariance;
    screenPointCovariance << (jacobian * pointCovariance.selfadjointView<Eigen::Lower>() * jacobian.transpose());
    return screenPointCovariance;
}

ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                       const WorldCoordinateCovariance& pointCovariance) noexcept
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const CameraCoordinateCovariance& pointCovariance) noexcept
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                     const CameraToWorldMatrix& cameraToWorld,
                                                     const matrix33& poseCovariance) noexcept
{
    const matrix33& rotation = cameraToWorld.block<3, 3>(0, 0);

    WorldCoordinateCovariance cov;
    cov << rotation * cameraPointCovariance.selfadjointView<Eigen::Lower>() * rotation.transpose() + poseCovariance;
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
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    // Jacobian of the screen to camera function. Use absolutes to prevent negative variances
    const matrix33 jacobian {{screenPoint.z() / cameraFX, 0.0, abs(screenPoint.x() - cameraCX) / cameraFX},
                             {0.0, screenPoint.z() / cameraFY, abs(screenPoint.y() - cameraCY) / cameraFY},
                             {0.0, 0.0, 1.0}};

    CameraCoordinateCovariance cameraPointCovariance;
    cameraPointCovariance << jacobian * screenPointCovariance.selfadjointView<Eigen::Lower>() * jacobian.transpose();
    return cameraPointCovariance;
}

matrix44 compute_plane_covariance(const PlaneCoordinates& planeParameters,
                                  const matrix33& pointCloudCovariance) noexcept
{
    assert(is_covariance_valid(pointCloudCovariance));

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    assert(not utils::double_equal(d, 0.0));
    assert(utils::double_equal(normal.norm(), 1.0));

    // reduce the parametrization
    const vector3 parameters = normal * d;
    const double a = parameters.x();
    const double b = parameters.y();
    const double c = parameters.z();

    const double aSquared = a * a;
    const double bSquared = b * b;
    const double cSquared = c * c;

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

    const matrix44& planeParameterCovariance =
            // add a little bit of variance on the diagonal to counter floatting points errors
            (jacobian * pointCloudCovariance.selfadjointView<Eigen::Lower>() * jacobian.transpose()) +
            matrix44::Identity() * 0.01;
    assert(is_covariance_valid(planeParameterCovariance));
    return planeParameterCovariance;
}

matrix33 compute_reduced_plane_point_cloud_covariance(const PlaneCoordinates& planeParameters,
                                                      const matrix44& planeCloudCovariance) noexcept
{
    assert(is_covariance_valid(planeCloudCovariance));

    const vector3& normal = planeParameters.get_normal();
    const double d = planeParameters.get_d();
    assert(not utils::double_equal(d, 0.0));
    assert(utils::double_equal(normal.norm(), 1.0));

    // compute the jacobian of the 4 parameter plane to 3 parameters plane transformation -> vect = normal * d
    const matrix34 jacobian({
            {d, 0, 0, normal.x()},
            {0, d, 0, normal.y()},
            {0, 0, d, normal.z()},
    });

    const matrix33& pointCloudCovariance =
            // add a little bit of variance on the diagonal to counter floatting points errors
            (jacobian * planeCloudCovariance.selfadjointView<Eigen::Lower>() * jacobian.transpose()) +
            matrix33::Identity() * 0.01;
    assert(is_covariance_valid(pointCloudCovariance));
    return pointCloudCovariance;
}

matrix44 get_world_plane_covariance(const PlaneCameraCoordinates& planeCoordinates,
                                    const CameraToWorldMatrix& cameraToWorldMatrix,
                                    const PlaneCameraToWorldMatrix& planeCameraToWorldMatrix,
                                    const matrix44& planeCovariance,
                                    const matrix33& worldPoseCovariance) noexcept
{
    // transform to point form
    const matrix33& pointCloudCovariance =
            compute_reduced_plane_point_cloud_covariance(planeCoordinates, planeCovariance);

    // covert covariance to world
    const matrix33& rotation = cameraToWorldMatrix.block<3, 3>(0, 0);
    const matrix33& pointCloudWorlCovariance =
            rotation * pointCloudCovariance * rotation.transpose() + worldPoseCovariance;
    assert(is_covariance_valid(pointCloudWorlCovariance));
    // conver back to plane hessian form
    return compute_plane_covariance(planeCoordinates.to_world_coordinates(planeCameraToWorldMatrix),
                                    pointCloudWorlCovariance);
}

} // namespace rgbd_slam::utils