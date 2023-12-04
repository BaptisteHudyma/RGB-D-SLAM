#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"
#include <stdexcept>

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
    _covariance.setZero();

    // new mesurment always as the same uncertainty in depth (and another one in position)
    _covariance.block<3, 3>(firstPoseIndex, firstPoseIndex) = stateCovariance;

    _covariance(inverseDepthIndex, inverseDepthIndex) =
            SQR(parameters::detection::inverseDepthBaseline / 4.0); // inverse depth covariance

    constexpr double anglevariance =
            SQR(parameters::detection::inverseDepthAngleBaseline * EulerToRadian); // angle uncertainty
    _covariance(thetaIndex, thetaIndex) = anglevariance;                           // theta angle covariance
    _covariance(phiIndex, phiIndex) = anglevariance;                               // phi angle covariance

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
    try
    {
        // get observation in world space
        const PointInverseDepth newObservation(screenObservation, c2w, stateCovariance, descriptor);
        // project to cartesian
        const utils::WorldCoordinate& cartesianProj = newObservation._coordinates.to_world_coordinates();
        WorldCoordinateCovariance covarianceProj =
                compute_cartesian_covariance(newObservation._coordinates, newObservation._covariance);

        return update_with_cartesian(cartesianProj, covarianceProj, descriptor);
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return false;
    }
}

bool PointInverseDepth::track(const utils::ScreenCoordinate& observation,
                              const CameraToWorldMatrix& c2w,
                              const matrix33& stateCovariance,
                              const cv::Mat& descriptor)
{
    try
    {
        // transform screen point to world point
        const utils::WorldCoordinate& worldPointCoordinates = observation.to_world_coordinates(c2w);
        // get a measure of the estimated variance of the new world point
        const WorldCoordinateCovariance& worldCovariance =
                utils::get_world_point_covariance(observation, c2w, stateCovariance);

        return update_with_cartesian(worldPointCoordinates, worldCovariance, descriptor);
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return false;
    }
}

bool PointInverseDepth::update_with_cartesian(const utils::WorldCoordinate& point,
                                              const WorldCoordinateCovariance& covariance,
                                              const cv::Mat& descriptor)
{
    try
    {
        // pass it through the filter
        build_kalman_filter();
        assert(_kalmanFilter != nullptr);

        const auto& [newState, newCovariance] =
                _kalmanFilter->get_new_state(_coordinates.to_world_coordinates(),
                                             compute_cartesian_covariance(_coordinates, _covariance),
                                             point,
                                             covariance);
        if (not utils::is_covariance_valid(newCovariance))
            throw std::invalid_argument("Inverse depth point covariance is invalid at the kalman output");

        // put back in inverse depth coordinates
        Eigen::Matrix<double, 6, 6> fromCartesianJacobian;
        _coordinates.from_cartesian(
                utils::WorldCoordinate(newState), _coordinates._firstObservation, fromCartesianJacobian);
        _covariance = compute_inverse_depth_covariance(WorldCoordinateCovariance(newCovariance),
                                                       _covariance.get_first_pose_covariance(),
                                                       fromCartesianJacobian);

        if (not utils::is_covariance_valid(_covariance))
            throw std::invalid_argument("Inverse depth point covariance is invalid after merge");

        if (not descriptor.empty())
            _descriptor = descriptor;

        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exeption: " + std::string(ex.what()));
        return false;
    }
}

CameraCoordinateCovariance PointInverseDepth::get_camera_coordinate_variance(const WorldToCameraMatrix& w2c) const
{
    // get world coordinates covariance, transform it to camera
    return utils::get_camera_point_covariance(
            PointInverseDepth::compute_cartesian_covariance(_coordinates, _covariance),
            w2c,
            get_covariance_of_observed_pose());
}

ScreenCoordinateCovariance PointInverseDepth::get_screen_coordinate_variance(const WorldToCameraMatrix& w2c) const
{
    return utils::get_screen_point_covariance(_coordinates.to_camera_coordinates(w2c),
                                              get_camera_coordinate_variance(w2c));
}

WorldCoordinateCovariance PointInverseDepth::compute_cartesian_covariance(
        const utils::InverseDepthWorldPoint& coordinates, const matrix66& covariance)
{
    if (not utils::is_covariance_valid(covariance))
        throw std::invalid_argument("compute_cartesian_covariance cannot use incorrect covariance in covariance");

    Eigen::Matrix<double, 3, 6> jacobian;
    coordinates.to_world_coordinates(jacobian);
    return PointInverseDepth::compute_cartesian_covariance(covariance, jacobian);
}

WorldCoordinateCovariance PointInverseDepth::compute_cartesian_covariance(const matrix66& covariance,
                                                                          const Eigen::Matrix<double, 3, 6>& jacobian)
{
    if (not utils::is_covariance_valid(covariance))
        throw std::invalid_argument("compute_cartesian_covariance cannot use incorrect covariance in covariance");

    WorldCoordinateCovariance worldCovariance(jacobian * covariance.selfadjointView<Eigen::Lower>() *
                                              jacobian.transpose());
    if (not utils::is_covariance_valid(worldCovariance))
        throw std::logic_error("compute_cartesian_covariance produced an invalid covariance");
    return worldCovariance;
}

PointInverseDepth::Covariance PointInverseDepth::compute_inverse_depth_covariance(
        const WorldCoordinateCovariance& pointCovariance,
        const matrix33& firstPoseCovariance,
        const Eigen::Matrix<double, 6, 6>& jacobian)
{
    if (not utils::is_covariance_valid(pointCovariance))
        throw std::invalid_argument(
                "compute_inverse_depth_covariance cannot use incorrect covariance in pointCovariance");

    matrix66 pointCov;
    pointCov.setZero();
    pointCov.block<3, 3>(0, 0) = firstPoseCovariance;
    pointCov.block<3, 3>(3, 3) = pointCovariance;

    matrix66 resCovariance = jacobian * pointCov.selfadjointView<Eigen::Lower>() * jacobian.transpose();

    if (not utils::is_covariance_valid(resCovariance))
        throw std::logic_error("compute_inverse_depth_covariance produced an invalid covariance");

    Covariance c;
    c.base() = resCovariance;
    return c;
}

void PointInverseDepth::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix33 systemDynamics = matrix33::Identity(); // points are not supposed to move, so no dynamics
        const matrix33 outputMatrix = matrix33::Identity();   // we need all positions

        const double parametersProcessNoise = 0.0; // TODO set in parameters
        const matrix33 processNoiseCovariance =
                matrix33::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<3, 3>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

} // namespace rgbd_slam::tracking