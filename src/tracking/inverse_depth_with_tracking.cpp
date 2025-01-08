#include "inverse_depth_with_tracking.hpp"

#include "camera_transformation.hpp"
#include "coordinates/point_coordinates.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/covariances.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <stdexcept>

namespace rgbd_slam::tracking {

/**
 * Point Inverse Depth
 */

PointInverseDepth::PointInverseDepth(const ScreenCoordinate2D& observation,
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

    // span from variance from 0 to 1:
    constexpr double inverseDepthVar = parameters::detection::inverseDepthBaseline / 4.0;
    _covariance(inverseDepthIndex, inverseDepthIndex) = SQR(inverseDepthVar);

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

bool PointInverseDepth::track(const ScreenCoordinate2D& screenObservation,
                              const CameraToWorldMatrix& c2w,
                              const matrix33& stateCovariance,
                              const cv::Mat& descriptor) noexcept
{
    try
    {
        build_kalman_filter();
        assert(_extendedKalmanFilter != nullptr);

        const auto& w2c = utils::compute_world_to_camera_transform(c2w);

        // build the estimator
        InverseDepthEstimator estimator(_coordinates.get_vector(),
                                        _covariance,
                                        screenObservation,
                                        (matrix22::Identity() * SQR(0.5)).eval(),
                                        w2c);

        const auto& [newState, newCovariance] = _extendedKalmanFilter->get_new_state(&estimator);

        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("Inverse depth point covariance is invalid after merge");
            return false;
        }

        /*if ((newCovariance.diagonal().array() > _covariance.diagonal().array()).any())
        {
            std::cout << (newCovariance.diagonal() - _covariance.diagonal()).transpose() << std::endl;
            outputs::log_error("new covariance is worse !");
            return false;
        }*/

        _coordinates.set_vector(newState);
        _covariance = newCovariance;

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

bool PointInverseDepth::track(const ScreenCoordinate& observation,
                              const CameraToWorldMatrix& c2w,
                              const matrix33& stateCovariance,
                              const cv::Mat& descriptor)
{
    if (not is_depth_valid(observation.z()))
    {
        outputs::log_error("depth is invalid in a function depending on depth");
        return false;
    }

    try
    {
        return false;
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
    // this use of the projection to screen is ok as long as the inverse depth point uncertainty is fairly low
    return utils::get_screen_point_covariance(_coordinates.to_world_coordinates().to_camera_coordinates(w2c),
                                              get_camera_coordinate_variance(w2c));
}

WorldCoordinateCovariance PointInverseDepth::compute_cartesian_covariance(const InverseDepthWorldPoint& coordinates,
                                                                          const matrix66& covariance)
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

    WorldCoordinateCovariance worldCovariance {utils::propagate_covariance(covariance, jacobian, 0.0)};
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

    Covariance resCovariance = utils::propagate_covariance(pointCovariance, jacobian, 0.0);
    resCovariance.block<3, 3>(0, 0) = firstPoseCovariance;
    if (not utils::is_covariance_valid(resCovariance))
    {
        throw std::logic_error("compute_inverse_depth_covariance produced an invalid covariance");
    }

    return resCovariance;
}

double PointInverseDepth::compute_linearity_score(const CameraToWorldMatrix& cameraToWorld) const noexcept
{
    // gaussian linearity index, taken from:
    // "Inverse Depth Parametrization for Monocular SLAM"

    Eigen::Matrix<double, 3, 6> jacobian;
    const WorldCoordinate& cartesian = _coordinates.to_world_coordinates(jacobian);

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

        // TODO: all process noises should be crafted with care, those values are just handwaved
        _extendedKalmanFilter = std::make_unique<tracking::ExtendedKalmanFilter<6, 2>>(matrix66::Identity() * 1e-8);
    }
}

bool PointInverseDepth::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                              utils::Segment<2>& screenSegment) const noexcept
{
    return _coordinates.to_screen_coordinates(w2c, _covariance.get_inverse_depth_variance(), screenSegment);
}

} // namespace rgbd_slam::tracking