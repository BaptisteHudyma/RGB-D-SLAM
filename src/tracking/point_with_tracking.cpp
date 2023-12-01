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
    Eigen::Matrix<double, 6, 3> fromCartesianJacobian;
    _coordinates.from_cartesian(
            utils::WorldCoordinate(newState), _coordinates._firstObservation, fromCartesianJacobian);
    _covariance = compute_inverse_depth_covariance(
            WorldCoordinateCovariance(newCovariance), _covariance.block<3, 3>(0, 0), fromCartesianJacobian);
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

    Eigen::Matrix<double, 3, 6> jacobian;
    coordinates.to_world_coordinates(jacobian);
    return PointInverseDepth::compute_cartesian_covariance(covariance, jacobian);
}

WorldCoordinateCovariance PointInverseDepth::compute_cartesian_covariance(
        const matrix66& covariance, const Eigen::Matrix<double, 3, 6>& jacobian) noexcept
{
    assert(utils::is_covariance_valid(covariance));

    WorldCoordinateCovariance worldCovariance(jacobian * covariance * jacobian.transpose());
    assert(utils::is_covariance_valid(worldCovariance));
    return worldCovariance;
}

matrix66 PointInverseDepth::compute_inverse_depth_covariance(const WorldCoordinateCovariance& pointCovariance,
                                                             const matrix33& posevariance,
                                                             const Eigen::Matrix<double, 6, 3>& jacobian) noexcept
{
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