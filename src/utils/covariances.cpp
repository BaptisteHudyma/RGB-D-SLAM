#include "covariances.hpp"
#include "../parameters.hpp"
#include "camera_transformation.hpp"
#include "coordinates.hpp"
#include "distance_utils.hpp"
#include "logger.hpp"
#include "types.hpp"
#include <cmath>

namespace rgbd_slam::utils {

double get_depth_quantization(const double depth)
{
    // minimum depth diparity at z is the quadratic function  a + b z + c z^2
    const static double depthSigmaError = Parameters::get_depth_sigma_error() * pow(1.0 / 1000.0, 2.0);
    const static double depthSigmaMultiplier = Parameters::get_depth_sigma_multiplier() / 1000.0;
    const static double depthSigmaMargin = Parameters::get_depth_sigma_margin();
    return std::max(depthSigmaMargin + depthSigmaMultiplier * depth + depthSigmaError * pow(depth, 2.0), 0.5);
}

ScreenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance)
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

ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                       const WorldCoordinateCovariance& pointCovariance)
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                       const CameraCoordinateCovariance& pointCovariance)
{
    return get_screen_point_covariance(point.base(), pointCovariance.base());
}

WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                     const matrix33& poseCovariance)
{
    WorldCoordinateCovariance cov;
    cov.base() << cameraPointCovariance + poseCovariance;
    return cov;
}

WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                     const matrix33& poseCovariance)
{
    return get_world_point_covariance(utils::get_camera_point_covariance(screenPoint), poseCovariance);
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint)
{
    return get_camera_point_covariance(screenPoint, screenPoint.get_covariance());
}

CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
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

matrix44 compute_plane_covariance(const vector4& planeParameters,
                                  const matrix33& pointCloudhessian,
                                  const matrix33& positionCovariance)
{
    const vector3& normal = planeParameters.head(3);
    const double d = planeParameters(3);
    assert(not utils::double_equal(d, 0.0));
    assert(utils::double_equal(normal.norm(), 1.0));

    // 0 determinant cannot be inverted
    assert(not utils::double_equal(pointCloudhessian.determinant(), 0.0));

    // compute covariance with the addition of an eventual position covariance
    const matrix33 covariance = pointCloudhessian.inverse() + positionCovariance;

    // reduce the parametrization
    const vector3 parameters = normal / d;
    const double a = parameters.x();
    const double b = parameters.y();
    const double c = parameters.z();

    const double aSquared = a * a;
    const double bSquared = b * b;
    const double cSquared = c * c;

    // common divider of all partial derivatives
    const double divider = pow(aSquared + bSquared + cSquared, 3.0 / 2.0);

    // compute the jacobian of the transformation
    matrix43 jacobian({
            {bSquared + cSquared, -a * b, -a * c},
            {-a * b, aSquared + cSquared, -b * c},
            {-a * c, -b * c, aSquared + bSquared},
            {-a, -b, -c},
    });
    jacobian /= divider;

    const matrix44& planeParameterCovariance = jacobian * covariance * jacobian.transpose();
    assert(planeParameterCovariance.diagonal()(0) >= 0 and planeParameterCovariance.diagonal()(1) >= 0 and
           planeParameterCovariance.diagonal()(2) >= 0 and planeParameterCovariance.diagonal()(3) >= 0);
    return planeParameterCovariance;
}

} // namespace rgbd_slam::utils