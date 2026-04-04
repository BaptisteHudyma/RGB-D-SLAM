#include "inverse_depth_with_tracking.hpp"

#include "coordinates/inverse_depth_coordinates.hpp"
#include "coordinates/point_coordinates.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include "utils/covariances.hpp"

#include <stdexcept>

namespace rgbd_slam::tracking {

/**
 * Define the estimator for the inverse depth fuse/tracker
 */
template<int N = 6, int M = 2, int NE = N, int ME = M> class InverseDepthEstimator : public StateEstimator<N, M, NE, ME>
{
  public:
    virtual ~InverseDepthEstimator() = default;

    std::pair<Eigen::Vector<double, M>, Eigen::Matrix<double, ME, NE>> h(
            const Eigen::Vector<double, N>& state) const noexcept override
    {
        InverseDepthWorldPoint idepthState(state);
        return {idepthState.get_observation_model(_w2c), idepthState.get_observation_model_jacobian(_w2c)};
    }

    InverseDepthEstimator(const Eigen::Vector<double, N>& feature,
                          const Eigen::Matrix<double, NE, NE>& featureCovariance,
                          const Eigen::Vector<double, M>& measurment,
                          const Eigen::Matrix<double, ME, ME>& measurmentCovariance,
                          const WorldToCameraMatrix& w2c,
                          const matrix66& poseCovariance) :
        StateEstimator<N, M, NE, ME>(feature, featureCovariance, measurment, measurmentCovariance),
        _w2c(w2c),
        _poseCovariance(poseCovariance)
    {
    }

    inline Eigen::Matrix<double, ME, ME> h_innovation(
            const Eigen::Vector<double, N>& state,
            const Eigen::Matrix<double, NE, NE>& estimateErrorCovariance,
            const Eigen::Matrix<double, ME, NE>& hJacobian) const noexcept override
    {
        const Eigen::Matrix<double, 2, 6>& hPoseJacobian =
                utils::world_transform_of_2d_point_jacobian(InverseDepthWorldPoint(state).to_world_coordinates(), _w2c);

        return utils::propagate_covariance(estimateErrorCovariance, hJacobian) +
               utils::propagate_covariance(_poseCovariance, hPoseJacobian);
    }

  private:
    const WorldToCameraMatrix _w2c;
    const matrix66 _poseCovariance;
};

/**
 * Point Inverse Depth
 */

PointInverseDepth::PointInverseDepth(const ScreenCoordinate2D& observation,
                                     const CameraToWorldMatrix& c2w,
                                     const matrix66& stateCovariance,
                                     const cv::Mat& descriptor) :
    _coordinates(observation, c2w),
    _descriptor(descriptor)
{
    if (_extendedKalmanFilter == nullptr)
        build_kalman_filter();

    if (not utils::is_covariance_valid(stateCovariance))
    {
        throw std::invalid_argument("Inverse depth stateCovariance is invalid in constructor");
    }

    _covariance.setZero();

    // TODO: this covariance is wrong: it omits the image noise and rotation error

    // new mesurment always as the same uncertainty in depth (and another one in position)
    _covariance.block<3, 3>(firstPoseIndex, firstPoseIndex) = stateCovariance.block<3, 3>(0, 0);

    // span from variance from 0 to 1:
    const double pMin = 1.0 / 0.1;    // 10cm
    const double pMax = 1.0 / 1000.0; // 1000m
    const double standardDevSpan = (pMin - pMax) / 2.0;
    _covariance(inverseDepthIndex, inverseDepthIndex) = SQR(standardDevSpan);

    // TODO: integrate pose rotation variance here
    constexpr double anglevariance =
            SQR(parameters::detection::inverseDepthAngleBaseline_deg * EulerToRadian); // angle uncertainty
    _covariance(thetaIndex, thetaIndex) = anglevariance;                               // theta angle covariance
    _covariance(phiIndex, phiIndex) = anglevariance;                                   // phi angle covariance

    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("PointInverseDepth constructor: the builded covariance is invalid");
}

PointInverseDepth::PointInverseDepth(const PointInverseDepth& other) :
    _coordinates(other._coordinates),
    _covariance(other._covariance),
    _descriptor(other._descriptor)
{
    if (_extendedKalmanFilter == nullptr)
        build_kalman_filter();

    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("PointInverseDepth constructor: the given covariance is invalid");
}

PointInverseDepth::PointInverseDepth(const InverseDepthWorldPoint& coordinates, const Covariance& covariance) :
    _coordinates(coordinates),
    _covariance(covariance)
{
    if (_extendedKalmanFilter == nullptr)
        build_kalman_filter();

    if (not utils::is_covariance_valid(_covariance))
        throw std::invalid_argument("PointInverseDepth constructor: the given covariance is invalid");

    outputs::log_error("You are using a constructor dedicated for testing");
}

bool PointInverseDepth::track_2D(const ScreenCoordinate2D& observation,
                                 const matrix22& observationCovariance,
                                 const CameraToWorldMatrix& c2w,
                                 const matrix66& stateCovariance) noexcept
{
    assert(_extendedKalmanFilter != nullptr);
    try
    {
        const auto& w2c = utils::compute_world_to_camera_transform(c2w);

        // build the estimator
        InverseDepthEstimator estimator(
                _coordinates.get_vector(), _covariance, observation, observationCovariance, w2c, stateCovariance);

        const auto& [newState, newCovariance] = _extendedKalmanFilter->get_new_state(&estimator);

        if (newState(inverseDepthIndex) < 0.0)
        {
            // invalid merge OR valid merge with degenerate case (point estimation at infinity)
            // This indicates a wrong association
            return false;
        }
        if (not utils::is_covariance_valid(newCovariance))
        {
            outputs::log_error("Inverse depth point covariance is invalid after merge");
            return false;
        }

        _coordinates.set_vector(newState);
        _covariance = newCovariance;
        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exception: " + std::string(ex.what()));
        return false;
    }
}

bool PointInverseDepth::track_3D(const ScreenCoordinate& observation,
                                 const matrix33& observationCovariance,
                                 const CameraToWorldMatrix& c2w,
                                 const matrix66& stateCovariance) noexcept
{
    if (not is_depth_valid(observation.z()))
    {
        outputs::log_error("depth is invalid in a function depending on depth");
        return false;
    }

    try
    {
        // HACK: too much problems with the 3D tracking mathematically, juste replace this point

        matrix33 screenToWorldJacobian;
        const WorldCoordinate& wc = observation.to_world_coordinates(c2w, screenToWorldJacobian);
        const WorldCoordinateCovariance& worldCovariance = ScreenCoordinate::get_world_point_covariance(
                observationCovariance, stateCovariance, screenToWorldJacobian);

        Eigen::Matrix<double, 6, 3> worldToIDepthJacobian;
        const auto& state = InverseDepthWorldPoint::from_cartesian(wc, c2w.translation(), worldToIDepthJacobian);
        const auto& statecovariance = utils::propagate_covariance(worldCovariance, worldToIDepthJacobian);

        if (not utils::is_covariance_valid(statecovariance))
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

        _coordinates = state;
        _covariance = statecovariance;
        return true;
    }
    catch (const std::exception& ex)
    {
        outputs::log_error("Catch exception: " + std::string(ex.what()));
        return false;
    }
}

CameraCoordinateCovariance PointInverseDepth::get_camera_coordinate_variance(const WorldToCameraMatrix& w2c) const
{
    matrix33 worldToCamerajacobian;
    std::ignore = _coordinates.to_world_coordinates().to_camera_coordinates(w2c, worldToCamerajacobian);

    // get world coordinates covariance, transform it to camera
    const matrix33& cameraCovariance =
            utils::propagate_covariance(PointInverseDepth::compute_cartesian_covariance(_coordinates, _covariance),
                                        worldToCamerajacobian) +
            utils::propagate_covariance(this->get_covariance_of_observed_pose(), worldToCamerajacobian);
    return CameraCoordinateCovariance {cameraCovariance};
}

ScreenCoordinateCovariance PointInverseDepth::get_screen_coordinate_variance(const WorldToCameraMatrix& w2c) const
{
    const CameraCoordinate& cameraProjection = _coordinates.to_world_coordinates().to_camera_coordinates(w2c);

    // get camera covariance
    const CameraCoordinateCovariance& cameraCovariance = get_camera_coordinate_variance(w2c);

    matrix33 toScreenJacobian;
    ScreenCoordinate sc;
    std::ignore = cameraProjection.to_screen_coordinates(sc, toScreenJacobian);

    // this use of the projection to screen is ok as long as the inverse depth point uncertainty is fairly low
    return ScreenCoordinateCovariance {utils::propagate_covariance(cameraCovariance, toScreenJacobian)};
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

    WorldCoordinateCovariance worldCovariance {utils::propagate_covariance(covariance, jacobian)};
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

    Covariance resCovariance = utils::propagate_covariance(pointCovariance, jacobian);
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
    const WorldCoordinate& cartesian = _coordinates.to_world_coordinates();

    const vector3 hc(cartesian - cameraToWorld.translation());
    const double norm = hc.norm();
    const double cosAlpha = _coordinates.get_bearing_vector().dot(hc / norm);
    const double thetad_meters =
            sqrt(_covariance.diagonal()(PointInverseDepth::inverseDepthIndex)) / SQR(_coordinates.get_inverse_depth());
    const double d1_meters = norm;

    return 4.0 * thetad_meters / d1_meters * abs(cosAlpha);
}

bool PointInverseDepth::is_new_inverse_depth_valid(const double inverseDepth) const
{
    const double iDepthStandardDev = sqrt(_covariance.diagonal()(PointInverseDepth::inverseDepthIndex));
    return inverseDepth >= std::max(0.0, _coordinates.get_inverse_depth() - 2.0 * iDepthStandardDev) and
           inverseDepth <= _coordinates.get_inverse_depth() + 2.0 * iDepthStandardDev;
}

void PointInverseDepth::build_kalman_filter() noexcept
{
    // TODO: all process noises should be crafted with care, those values are just handwaved
    _extendedKalmanFilter = std::make_unique<tracking::ExtendedKalmanFilter<6, 2>>(matrix66::Zero());
}

bool PointInverseDepth::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                              utils::Segment<2>& screenSegment) const noexcept
{
    return _coordinates.to_screen_coordinates(w2c, _covariance.get_inverse_depth_variance(), screenSegment);
}

} // namespace rgbd_slam::tracking
