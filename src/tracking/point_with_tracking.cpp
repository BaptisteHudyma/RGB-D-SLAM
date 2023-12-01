#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"

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
    constexpr double anglevariance = SQR(2 * EulerToRadian);
    _covariance(3, 3) = anglevariance; // theta angle covariance
    _covariance(4, 4) = anglevariance; // phi angle covariance

    constexpr double dmin = 1000;
    _covariance(5, 5) = SQR((1.0 / dmin) / 4); // 1 meters depth covariance
    assert(utils::is_covariance_valid(_covariance));
}

PointInverseDepth::PointInverseDepth(const PointInverseDepth& other) :
    _coordinates(other._coordinates),
    _covariance(other._covariance),
    _descriptor(other._descriptor)
{
    assert(utils::is_covariance_valid(_covariance));
}

bool PointInverseDepth::track(const utils::ScreenCoordinate2D& screenObservation,
                              const CameraToWorldMatrix& c2w,
                              const matrix33& stateCovariance,
                              const cv::Mat& descriptor)
{
    // get observation in world space
    PointInverseDepth newObservation(screenObservation, c2w, stateCovariance, descriptor);

    // project to cartesian
    const utils::WorldCoordinate& cartesianProj = newObservation._coordinates.to_world_coordinates();
    WorldCoordinateCovariance covarianceProj =
            compute_cartesian_covariance(newObservation._coordinates, newObservation._covariance);

    assert(utils::is_covariance_valid(covarianceProj));

    // pass it through the filter
    build_kalman_filter();
    assert(_kalmanFilter != nullptr);

    const auto& [newState, newCovariance] =
            _kalmanFilter->get_new_state(_coordinates.to_world_coordinates(),
                                         compute_cartesian_covariance(_coordinates, _covariance),
                                         cartesianProj,
                                         covarianceProj);
    assert(utils::is_covariance_valid(newCovariance));

    // put back in inverse depth coordinates
    _coordinates.from_cartesian(utils::WorldCoordinate(newState), _coordinates._firstObservation);
    const matrix33& finalCovar = compute_inverse_depth_covariance(newState - _coordinates._firstObservation,
                                                                  WorldCoordinateCovariance(newCovariance),
                                                                  stateCovariance)
                                         .block<3, 3>(3, 3);
    assert(utils::is_covariance_valid(finalCovar));

    _covariance.block<3, 3>(3, 3) = finalCovar;
    assert(utils::is_covariance_valid(_covariance));

    if (not descriptor.empty())
        _descriptor = descriptor;

    return true;
}

CameraCoordinateCovariance PointInverseDepth::get_camera_coordinate_variance(
        const WorldToCameraMatrix& w2c) const noexcept
{
    // get world coordinates covariance, transform it to camera
    return utils::get_camera_point_covariance(
            PointInverseDepth::compute_cartesian_covariance(_coordinates, _covariance),
            w2c,
            get_covariance_of_observed_pose());
}

ScreenCoordinateCovariance PointInverseDepth::get_screen_coordinate_variance(
        const WorldToCameraMatrix& w2c) const noexcept
{
    return utils::get_screen_point_covariance(_coordinates.to_camera_coordinates(w2c),
                                              get_camera_coordinate_variance(w2c));
}

WorldCoordinateCovariance PointInverseDepth::compute_cartesian_covariance(
        const utils::InverseDepthWorldPoint& coordinates, const matrix66& covariance) noexcept
{
    assert(utils::is_covariance_valid(covariance));

    // jacobian of the to_world_coordinates() operation
    using matrix36 = Eigen::Matrix<double, 3, 6>;
    matrix36 jacobian(matrix36::Zero());
    jacobian.block<3, 3>(0, 0) = matrix33::Identity();

    // compute sin and cos in advance (opti)
    const double cosTheta = cos(coordinates._theta_rad);
    const double sinTheta = sin(coordinates._theta_rad);
    const double cosPhi = cos(coordinates._phi_rad);
    const double sinPhi = sin(coordinates._phi_rad);

    const double depth = 1.0 / coordinates._inverseDepth_mm;
    const double depthSqr = 1.0 / SQR(coordinates._inverseDepth_mm);
    const double cosPhiSinTheta = cosPhi * sinTheta;
    const double cosPhiCosTheta = cosPhi * cosTheta;

    jacobian.block<3, 3>(0, 3) =
            matrix33({{cosPhiCosTheta * depth, -sinPhi * sinTheta * depth, -cosPhiSinTheta * depthSqr},
                      {0, -cosPhi * depth, sinPhi * depthSqr},
                      {-cosPhiSinTheta * depth, -cosTheta * sinPhi * depth, -cosPhiCosTheta * depthSqr}});
    // jacobian of:
    // _firstObservation + 1.0 / _inverseDepth_mm * get_bearing_vector()

    WorldCoordinateCovariance worldCovariance(jacobian * covariance * jacobian.transpose());
    assert(utils::is_covariance_valid(worldCovariance));
    return worldCovariance;
}

matrix66 PointInverseDepth::compute_inverse_depth_covariance(const vector3& observationVector,
                                                             const WorldCoordinateCovariance& pointCovariance,
                                                             const matrix33& posevariance) noexcept
{
    assert(utils::is_covariance_valid(pointCovariance));
    // jacobian of the from_cartesian() operation
    using matrix63 = Eigen::Matrix<double, 6, 3>;
    matrix63 jacobian(matrix63::Zero());

    const double xSqr = SQR(observationVector.x());
    const double ySqr = SQR(observationVector.y());
    const double zSqr = SQR(observationVector.z());

    const double oneOverXZ = 1.0 / (xSqr + zSqr);
    const double sqrtXZ = sqrt(xSqr + zSqr);
    const double theta1 = 1.0 / (sqrtXZ * (xSqr + ySqr + zSqr));

    jacobian(3, 0) = observationVector.z() * oneOverXZ;
    jacobian(3, 2) = -observationVector.x() * oneOverXZ;

    jacobian(4, 0) = observationVector.x() * observationVector.y() * theta1;
    jacobian(4, 1) = -sqrtXZ / (xSqr + ySqr + zSqr);
    jacobian(4, 2) = observationVector.y() * observationVector.z() * theta1;

    jacobian(5, 2) = -1 / zSqr;

    // this is the jacobian of
    // theta = atan2(x, z)
    // phi = atan2(-y, sqrt(x*x + z*z))
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
        const matrix33 systemDynamics = matrix33::Identity(); // points are not supposed to move, so no dynamics
        const matrix33 outputMatrix = matrix33::Identity();   // we need all positions

        const double parametersProcessNoise = 0.0001; // TODO set in parameters
        const matrix33 processNoiseCovariance =
                matrix33::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<3, 3>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

} // namespace rgbd_slam::tracking