#include "point_with_tracking.hpp"

#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
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

    if (_descriptor.empty() or _descriptor.cols <= 0)
        throw std::invalid_argument("Point constructor: descriptor is empty");
    if (_coordinates.hasNaN())
        throw std::invalid_argument("Point constructor: point coordinates contains NaN");
    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("Point constructor: covariance in invalid");
};

double Point::track(const utils::WorldCoordinate& newDetectionCoordinates,
                    const matrix33& newDetectionCovariance) noexcept
{
    assert(_kalmanFilter != nullptr);
    if (not utils::is_covariance_valid(newDetectionCovariance))
    {
        outputs::log_error("newDetectionCovariance: the covariance is invalid");
        return -1;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance : the covariance is invalid");
        exit(-1);
    }

    try
    {
        const auto& [newCoordinates, newCovariance] = _kalmanFilter->get_new_state(
                _coordinates, _covariance, newDetectionCoordinates, newDetectionCovariance);

        const double score = (_coordinates - newCoordinates).norm();

        _coordinates << newCoordinates;
        _covariance << newCovariance;
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

        const double parametersProcessNoise = 0.001; // TODO set in parameters
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
                                     const matrix33& stateCovariance) :
    PointInverseDepth(observation, c2w, stateCovariance, cv::Mat())
{
}

PointInverseDepth::PointInverseDepth(const utils::ScreenCoordinate2D& observation,
                                     const CameraToWorldMatrix& c2w,
                                     const matrix33& stateCovariance,
                                     const cv::Mat& descriptor) :
    _coordinates(observation, c2w),
    _descriptor(descriptor)
{
    if (not utils::is_covariance_valid(stateCovariance))
    {
        throw std::invalid_argument("Inverse depth stateCovariance is invalid in constructor");
    }

    _covariance.setZero();

    // new mesurment always as the same uncertainty in depth (and another one in position)
    _covariance.block<3, 3>(firstPoseIndex, firstPoseIndex) = stateCovariance;

    _covariance(inverseDepthIndex, inverseDepthIndex) =
            SQR(parameters::detection::inverseDepthBaseline / 4.0); // inverse depth covariance

    constexpr double anglevariance =
            SQR(parameters::detection::inverseDepthAngleBaseline * EulerToRadian); // angle uncertainty
    _covariance(thetaIndex, thetaIndex) = anglevariance;                           // theta angle covariance
    _covariance(phiIndex, phiIndex) = anglevariance;                               // phi angle covariance

    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("PointInverseDepth constructor: the builded covariance is invalid");
}

PointInverseDepth::PointInverseDepth(const PointInverseDepth& other) :
    _coordinates(other._coordinates),
    _covariance(other._covariance),
    _descriptor(other._descriptor)
{
    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("PointInverseDepth constructor: the given covariance is invalid");
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
        const WorldCoordinateCovariance& covarianceProj =
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
    if (not utils::is_depth_valid(observation.z()))
    {
        outputs::log_error("depth is invalid in a function depending on depth");
        return false;
    }

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
    if (not utils::is_covariance_valid(covariance))
    {
        outputs::log_error("covariance: covariance is invalid");
        return false;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance: covariance is invalid");
        exit(-1);
    }

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
        Eigen::Matrix<double, 6, 3> fromCartesianJacobian;
        _coordinates = utils::InverseDepthWorldPoint::from_cartesian(
                utils::WorldCoordinate(newState), _coordinates.get_first_observation(), fromCartesianJacobian);
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
        outputs::log_error("Caught exeption while tracking: " + std::string(ex.what()));
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
    // ignore result: waste of cpu cycle, but the user did not provide the jacobian
    std::ignore = coordinates.to_world_coordinates(jacobian);
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
        const Eigen::Matrix<double, 6, 3>& jacobian)
{
    if (not utils::is_covariance_valid(pointCovariance))
        throw std::invalid_argument(
                "compute_inverse_depth_covariance cannot use incorrect covariance in pointCovariance");
    if (not utils::is_covariance_valid(firstPoseCovariance))
        throw std::invalid_argument(
                "compute_inverse_depth_covariance cannot use incorrect covariance in firstPoseCovariance");

    Covariance resCovariance = jacobian * pointCovariance.selfadjointView<Eigen::Lower>() * jacobian.transpose();
    resCovariance.block<3, 3>(0, 0) = firstPoseCovariance;
    if (not utils::is_covariance_valid(resCovariance))
    {
        throw std::logic_error("compute_inverse_depth_covariance produced an invalid covariance");
    }

    return resCovariance;
}

double PointInverseDepth::compute_linearity_score(const CameraToWorldMatrix& cameraToWorld) const noexcept
{
    Eigen::Matrix<double, 3, 6> jacobian;
    const utils::WorldCoordinate& cartesian = _coordinates.to_world_coordinates(jacobian);

    const vector3 hc(cartesian - cameraToWorld.translation());
    const double cosAlpha = static_cast<double>(_coordinates.get_bearing_vector().transpose() * hc) / hc.norm();
    const double thetad_meters = (sqrt(_covariance.diagonal()(PointInverseDepth::inverseDepthIndex)) /
                                  SQR(_coordinates.get_inverse_depth())) /
                                 1000.0;
    const double d1_meters = hc.norm() / 1000.0;

    return 4.0 * thetad_meters / d1_meters * abs(cosAlpha);
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