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

matrix44 compute_plane_covariance(const matrix33& leastSquarePlaneParameters,
                                  const vector3& normal,
                                  const vector3& centroid,
                                  const matrix33& centroidError)
{
    // assert parameters
    assert(double_equal(normal.norm(), 1.0));
    assert(leastSquarePlaneParameters.diagonal()(0) >= 0 and leastSquarePlaneParameters.diagonal()(1) >= 0 and
           leastSquarePlaneParameters.diagonal()(2) >= 0);
    assert(centroidError.diagonal()(0) >= 0 and centroidError.diagonal()(1) >= 0 and centroidError.diagonal()(2) >= 0);

    // project the covariance of centroid of the plane to world space
    utils::ScreenCoordinate projected;
    if (not CameraCoordinate(centroid).to_screen_coordinates(projected))
    {
        // this should never happen, as the detected planes are always in camera view
        outputs::log_error("Could not backproject the plane centroid to screen coordinates");
        exit(-1);
    }
    // get the covariance of the centroid distance to origin
    const matrix33& centroidCovariance = utils::get_camera_point_covariance(projected) + centroidError;
    assert(centroidCovariance.diagonal()(0) >= 0 and centroidCovariance.diagonal()(1) >= 0 and
           centroidCovariance.diagonal()(2) >= 0);

    const vector3& absoluteCentroid = centroid.cwiseAbs();
    const vector3& weightedCentroid = absoluteCentroid / absoluteCentroid.sum(); // sum of all components == 1
    assert(double_equal(weightedCentroid.sum(), 1.0));

    // TODO: hack. pondered sum od the centroid covariance
    const double distanceCovariance = (weightedCentroid.cwiseProduct(centroidCovariance.diagonal())).sum();

    assert(not std::isnan(distanceCovariance) and distanceCovariance >= 0);

    // TODO: set normal uncertainty
    matrix44 planeCovariance = matrix44::Identity();
    planeCovariance(3, 3) = distanceCovariance;
    return planeCovariance;
}

} // namespace rgbd_slam::utils