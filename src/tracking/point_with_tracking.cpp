#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"
#include "utils/camera_transformation.hpp"

namespace rgbd_slam::tracking {

/**
 * Point
 */

Point::Point(const utils::WorldCoordinate& coordinates,
             const WorldCoordinateCovariance& covariance,
             const cv::Mat& descriptor) :
    _coordinates(coordinates),
    _descriptor(descriptor),
    _covariance(covariance)
{
    build_kalman_filter();

    assert(not _descriptor.empty() and _descriptor.cols > 0);
    assert(not _coordinates.hasNaN());
};

double Point::track(const utils::WorldCoordinate& newDetectionCoordinates,
                    const matrix33& newDetectionCovariance) noexcept
{
    assert(_kalmanFilter != nullptr);
    if (not utils::is_covariance_valid(newDetectionCovariance))
    {
        outputs::log_error("newDetectionCovariance: the covariance in invalid");
        return -1;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance: the covariance in invalid");
        return -1;
    }

    try
    {
        const auto& [newCoordinates, newCovariance] = _kalmanFilter->get_new_state(
                _coordinates, _covariance, newDetectionCoordinates, newDetectionCovariance);

        const double score = (_coordinates - newCoordinates).norm();

        _coordinates << newCoordinates;
        _covariance << newCovariance;
        assert(not _coordinates.hasNaN());
        return score;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return -1;
    }
}

void Point::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix33 systemDynamics = matrix33::Identity(); // points are not supposed to move, so no dynamics
        const matrix33 outputMatrix = matrix33::Identity();   // we need all positions

        const double parametersProcessNoise = 0; // TODO set in parameters
        const matrix33 processNoiseCovariance =
                matrix33::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<3, 3>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

/**
 * Point Inverse Depth
 */

PointInverseDepth::PointInverseDepth(const utils::ScreenCoordinate2D& observation,
                                     const CameraToWorldMatrix& c2w,
                                     const matrix33& stateCovariance,
                                     const cv::Mat& descriptor) :
    _coordinates(observation, c2w),
    _descriptor(descriptor)
{
    // new mesurment always as the same uncertainty in depth (and another one in position)
    _covariance.block<3, 3>(0, 0) = stateCovariance;
    _covariance(3, 3) = 0.1;       // theta angle covariance
    _covariance(4, 4) = 0.1;       // phi angle covariance
    _covariance(5, 5) = 0.1 * 0.1; // depth covariance

    assert(utils::is_covariance_valid(_covariance));
}

PointInverseDepth::PointInverseDepth(const PointInverseDepth& other) :
    _coordinates(other._coordinates),
    _covariance(other._covariance),
    _descriptor(other._descriptor)
{
    assert(utils::is_covariance_valid(_covariance));
}

PointInverseDepth::PointInverseDepth(const utils::CameraCoordinate& cameraCoordinates,
                                     const CameraCoordinateCovariance& cameraCovariance,
                                     const CameraToWorldMatrix& c2w,
                                     const matrix33& posevariance) :
    _coordinates(cameraCoordinates, c2w)
{
    const WorldCoordinateCovariance& worldCovariance =
            utils::get_world_point_covariance(cameraCovariance, c2w, matrix33::Zero());
    _covariance =
            get_inverse_depth_covariance(cameraCoordinates.to_world_coordinates(c2w), worldCovariance, posevariance);
    assert(utils::is_covariance_valid(_covariance));
}

bool PointInverseDepth::track(const utils::ScreenCoordinate2D& screenObservation,
                              const CameraToWorldMatrix& c2w,
                              const matrix33& stateCovariance,
                              const cv::Mat& descriptor)
{
    // get observation in world space
    const PointInverseDepth newObservation(screenObservation, c2w, stateCovariance, descriptor);

    // project to this point coordinates to perform the fusion
    const WorldToCameraMatrix& w2c = utils::compute_world_to_camera_transform(c2w);
    const PointInverseDepth observationInSameSpace(newObservation._coordinates.to_camera_coordinates(w2c),
                                                   newObservation.get_camera_coordinate_variance(w2c, matrix33::Zero()),
                                                   _coordinates._c2w,
                                                   stateCovariance);

    build_kalman_filter();
    assert(_kalmanFilter != nullptr);
    const auto& [newState, newCovariance] =
            _kalmanFilter->get_new_state(_coordinates.get_vector_state(),
                                         _covariance,
                                         observationInSameSpace._coordinates.get_vector_state(),
                                         observationInSameSpace._covariance);

    _coordinates.from_vector_state(newState);
    _covariance = newCovariance;
    assert(utils::is_covariance_valid(_covariance));

    if (not descriptor.empty())
        _descriptor = descriptor;

    return true;
}

CameraCoordinateCovariance PointInverseDepth::get_camera_coordinate_variance(
        const WorldToCameraMatrix& w2c, const matrix33& stateCovariance) const noexcept
{
    // get world coordinates covariance, transform it to camera
    return utils::get_camera_point_covariance(get_cartesian_covariance(), w2c, stateCovariance);
}

ScreenCoordinateCovariance PointInverseDepth::get_screen_coordinate_variance(
        const WorldToCameraMatrix& w2c, const matrix33& stateCovariance) const noexcept
{
    return utils::get_screen_point_covariance(_coordinates.to_camera_coordinates(w2c),
                                              get_camera_coordinate_variance(w2c, stateCovariance));
}

WorldCoordinateCovariance PointInverseDepth::get_cartesian_covariance() const noexcept
{
    // jacobian of the to_cartesian() operation
    using matrix36 = Eigen::Matrix<double, 3, 6>;
    matrix36 jacobian(matrix36::Zero());
    jacobian.block<3, 3>(0, 0) = matrix33::Identity();

    // compute sin and cos in advance (opti)
    const double cosTheta = cos(_coordinates._theta_rad);
    const double sinTheta = sin(_coordinates._theta_rad);
    const double cosPhi = cos(_coordinates._phi_rad);
    const double sinPhi = sin(_coordinates._phi_rad);

    const double depth = 1.0 / _coordinates._inverseDepth_mm;
    const double theta1 = cosPhi * sinPhi;
    const double theta2 = cosPhi * cosTheta;
    jacobian.block<3, 3>(0, 3) = matrix33({{theta2 * depth, -sinTheta * sinPhi * depth, -theta1},
                                           {-cosTheta * depth, 0, sinTheta},
                                           {-theta1 * depth, -cosTheta * sinPhi * depth, -theta2}});

    WorldCoordinateCovariance worldCovariance(jacobian * _covariance * jacobian.transpose());
    assert(utils::is_covariance_valid(worldCovariance));
    return worldCovariance;
}

matrix66 PointInverseDepth::get_inverse_depth_covariance(const utils::WorldCoordinate& point,
                                                         const WorldCoordinateCovariance& pointCovariance,
                                                         const matrix33& posevariance) const noexcept
{
    // jacobian of the from_cartesian() operation
    using matrix63 = Eigen::Matrix<double, 6, 3>;
    matrix63 jacobian(matrix63::Zero());

    const double xSqr = SQR(point.x());
    const double ySqr = SQR(point.y());
    const double zSqr = SQR(point.z());

    const double xSqrOverzSqr = xSqr / zSqr + 1;

    const double theta3 = xSqr + zSqr;
    const double theta2 = ySqr / theta3 + 1;
    const double theta1 = theta2 * pow(theta3, 3.0 / 2.0);

    // TODO: handle the angle covariances (use atan2 jacobian)
    /*jacobian(3, 0) = 1.0 / (point.z() * xSqrOverzSqr);
    jacobian(3, 2) = -point.x() / (zSqr * xSqrOverzSqr);

    jacobian(4, 0) = point.x() * point.y() / theta1;
    jacobian(4, 1) = -1 / ((ySqr / theta2 + 1) * sqrt(theta2));
    jacobian(4, 2) = point.x() * point.z() / theta1;*/

    jacobian(5, 2) = -1 / zSqr;

    // this is the jacobian of
    //
    // theta = atan(x/z)
    // phi = atan(-y / sqrt(x*x + z*z))
    // invDepth = 1/z

    matrix66 resCovariance = jacobian * pointCovariance * jacobian.transpose();
    // set pose base the covariance
    resCovariance.block<3, 3>(0, 0) = posevariance;
    assert(utils::is_covariance_valid(resCovariance));
    return resCovariance;
}

void PointInverseDepth::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix66 systemDynamics = matrix66::Identity(); // points are not supposed to move, so no dynamics
        const matrix66 outputMatrix = matrix66::Identity();   // we need all positions

        const double parametersProcessNoise = 0; // TODO set in parameters
        const matrix66 processNoiseCovariance =
                matrix66::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<6, 6>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

} // namespace rgbd_slam::tracking