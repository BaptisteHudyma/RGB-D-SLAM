#include "inverse_depth_coordinates.hpp"

#include "camera_transformation.hpp"
#include "coordinates/basis_changes.hpp"

#include "distance_utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {

static constexpr double maximumAllowedDepth_m = 100;

/**
 *      INVERSE DEPTH COORDINATES
 */

static constexpr auto firstPoseIndex = InverseDepthWorldPoint::firstPoseIndex;
static constexpr auto inverseDepthIndex = InverseDepthWorldPoint::inverseDepthIndex;
static constexpr auto thetaIndex = InverseDepthWorldPoint::thetaIndex;
static constexpr auto phiIndex = InverseDepthWorldPoint::phiIndex;

InverseDepthWorldPoint::InverseDepthWorldPoint(const WorldCoordinate& firstPose,
                                               const double inverseDepth_m,
                                               const double theta,
                                               const double phi) :
    _firstObservation(firstPose),
    _inverseDepth_m(inverseDepth_m),
    _theta_rad(theta),
    _phi_rad(phi)
{
    if (_inverseDepth_m < 0.0)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Inverse depth should be >= 0");
    if (_theta_rad < 0.0 or _theta_rad > M_PI)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Theta should be in [0, M_PI]");
    if (_phi_rad < -M_PI or _phi_rad > M_PI)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Phi should be in [-Pi, Pi]");

    recompute_bearing_vector();
    /*
    // in the paper, defined as the following:
    _bearingVector.x() = cos(_phi_rad) * sin(_theta_rad);
    _bearingVector.y() = -sin(_phi_rad);
    _bearingVector.z() = cos(_phi_rad) * cos(_theta_rad);
    // but we use another referencial (standard spherical referencial)
    */
}

void InverseDepthWorldPoint::recompute_bearing_vector() noexcept
{
    // norm will be one by construction (diameter at 1 unit)
    _bearingVector = Cartesian::from(Spherical(1.0, _theta_rad, _phi_rad)).vec();
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const ScreenCoordinate2D& observation, const CameraToWorldMatrix& c2w) :
    // use a baseline to project a ray from start point to observed point
    InverseDepthWorldPoint(
            observation.to_camera_coordinates_baseline(1.0 / parameters::detection::inverseDepthBaseline_m), c2w)
{
    // no known depth, so set the baseline
    _inverseDepth_m = parameters::detection::inverseDepthBaseline_m;
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const CameraCoordinate& observation, const CameraToWorldMatrix& c2w) :
    InverseDepthWorldPoint(from_cartesian(observation.to_world_coordinates(c2w), WorldCoordinate(c2w.translation())))
{
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const vector6& other) { set_vector(other); }

vector3 InverseDepthWorldPoint::compute_signed_distance(const InverseDepthWorldPoint& other) const
{
    return utils::signed_line_distance<3>(
            _firstObservation, _bearingVector, other._firstObservation, other._bearingVector);
}

vector3 InverseDepthWorldPoint::compute_signed_distance(const ScreenCoordinate2D& other,
                                                        const WorldToCameraMatrix& w2c) const
{
    return compute_signed_distance(InverseDepthWorldPoint(other, utils::compute_camera_to_world_transform(w2c)));
}

vector2 InverseDepthWorldPoint::compute_signed_screen_distance(const ScreenCoordinate2D& other,
                                                               const double inverseDepthCovariance,
                                                               const WorldToCameraMatrix& w2c) const
{
    utils::Segment<2> screenLine;
    if (to_screen_coordinates(w2c, inverseDepthCovariance, screenLine))
    {
        return screenLine.distance(other);
    }
    return vector2::Constant(std::numeric_limits<double>::max());
}

matrix22 InverseDepthWorldPoint::compute_signed_screen_distance_covariance(const ScreenCoordinate2D& other,
                                                                           const matrix66& cov,
                                                                           const WorldToCameraMatrix& w2c) const
{
    utils::Segment<2> screenLine;
    matrix44 covariance;
    if (to_screen_coordinates(w2c, cov, screenLine, covariance))
    {
        return screenLine.get_distance_covariance(other, covariance);
    }
    return matrix22::Identity();
}

InverseDepthWorldPoint InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point,
                                                              const WorldCoordinate& origin) noexcept
{
    const vector3 directionalVector(point - origin);

    const Spherical& s = Spherical::from(Cartesian(directionalVector));
    return InverseDepthWorldPoint(origin, 1.0 / s.p, s.polar_rad, s.azimuth_rad);
}

InverseDepthWorldPoint InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point,
                                                              const WorldCoordinate& origin,
                                                              Eigen::Matrix<double, 6, 3>& jacobian) noexcept
{
    jacobian.setZero();
    // initial pose block
    jacobian.block<3, 3>(firstPoseIndex, firstPoseIndex) = matrix33::Zero();

    // jacobian of the [xo, yo, zo, x, y, z] =>
    // [xo, yo, zo, inverse depth spherical projection of (x - xo, y - yo, z - zo)]

    const vector3 directionalVector(point - origin);

    matrix33 toCartesianJacobian;
    const auto& s = Spherical::from(Cartesian(directionalVector), toCartesianJacobian);

    matrix33 toIDepthJacobian;
    // add the derivative of 1/d part to have -1/pow(d, 3/2) instead of 1/pow(d, 1/2)
    toIDepthJacobian.row(inverseDepthIndex - 3) = -1.0 / SQR(s.p) * toCartesianJacobian.row(Spherical::RadiusIndex);
    toIDepthJacobian.row(thetaIndex - 3) = toCartesianJacobian.row(Spherical::PolarIndex);
    toIDepthJacobian.row(phiIndex - 3) = toCartesianJacobian.row(Spherical::AzimuthIndex);

    jacobian.block<3, 3>(firstPoseIndex + 3, 0) = toIDepthJacobian;
    return from_cartesian(point, origin);
}

WorldCoordinate InverseDepthWorldPoint::to_world_coordinates(const double addedStandardDev_m) const noexcept
{
    assert(_inverseDepth_m != 0.0);
    // limit the projection to infinity
    const double addedDepth = std::max(1.0 / maximumAllowedDepth_m, _inverseDepth_m + addedStandardDev_m);
    return _firstObservation + 1.0 / addedDepth * _bearingVector;
}

Eigen::Matrix<double, 3, 6> to_world_coordinates_jacobian(const double inverseDepth,
                                                          const double theta,
                                                          const double phi)
{
    // jacobian of _firstObservation + 1.0 / _inverseDepth_m * _bearingVector

    matrix33 bearingJacobian;
    Cartesian::from(Spherical(1.0, theta, phi), bearingJacobian);

    matrix33 iDeptJacobian;
    const double depth = 1.0 / inverseDepth;
    iDeptJacobian.col(inverseDepthIndex - 3) = -SQR(depth) * bearingJacobian.col(Spherical::RadiusIndex);
    iDeptJacobian.col(thetaIndex - 3) = depth * bearingJacobian.col(Spherical::PolarIndex);
    iDeptJacobian.col(phiIndex - 3) = depth * bearingJacobian.col(Spherical::AzimuthIndex);

    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, firstPoseIndex) = matrix33::Identity();
    jacobian.block<3, 3>(0, firstPoseIndex + 3) = iDeptJacobian;

    return jacobian;
}

WorldCoordinate InverseDepthWorldPoint::to_world_coordinates(Eigen::Matrix<double, 3, 6>& jacobian,
                                                             const double addedStandardDev_m) const noexcept
{
    jacobian = to_world_coordinates_jacobian(_inverseDepth_m + addedStandardDev_m, _theta_rad, _phi_rad);
    return to_world_coordinates(addedStandardDev_m);
}

Eigen::Matrix<double, 2, 6> InverseDepthWorldPoint::get_projected_screen_estimation_jacobian(
        const WorldToCameraMatrix& w2c, const double addedStandardDev) const noexcept
{
    Eigen::Matrix<double, 3, 6> inverseDepthToWorldJacobian;
    const WorldCoordinate& worldPoint = to_world_coordinates(inverseDepthToWorldJacobian, addedStandardDev);

    const matrix23& toScreenJacobian = worldPoint.to_screen2d_coordinates_jacobian(w2c);

    return (toScreenJacobian * inverseDepthToWorldJacobian).eval();
}

ScreenCoordinate2D InverseDepthWorldPoint::get_projected_screen_estimation(
        const WorldToCameraMatrix& w2c, const double addedStandardDev_m) const noexcept
{
    const CameraCoordinate& projectedCam = get_projected_camera_estimation(w2c, addedStandardDev_m);

    ScreenCoordinate2D resCoords;
    const bool res = projectedCam.to_screen_coordinates(resCoords);
    // TODO: something better than assert...
    assert(res);
    return resCoords;
}

CameraCoordinate InverseDepthWorldPoint::get_camera_observation_projection(
        const WorldToCameraMatrix& w2c) const noexcept
{
    const auto& c2w = utils::compute_camera_to_world_transform(w2c);
    return CameraCoordinate(w2c.rotation() *
                            (_inverseDepth_m * (_firstObservation - c2w.translation()) + _bearingVector));
}

ScreenCoordinate2D InverseDepthWorldPoint::get_observation_model(const WorldToCameraMatrix& w2c) const noexcept
{
    ScreenCoordinate2D s2d;
    assert(get_camera_observation_projection(w2c).to_screen_coordinates(s2d));
    return s2d;
}

Eigen::Matrix<double, 2, 6> InverseDepthWorldPoint::get_observation_model_jacobian(
        const WorldToCameraMatrix& w2c) const noexcept
{
    const auto& c2w = utils::compute_camera_to_world_transform(w2c);

    matrix33 bearingJacobian;
    Cartesian::from(Spherical(1.0, _theta_rad, _phi_rad), bearingJacobian);

    const vector3& trVec = _firstObservation - c2w.translation();

    matrix33 paramJacobians;
    paramJacobians.col(inverseDepthIndex - 3) = trVec;
    paramJacobians.col(thetaIndex - 3) = bearingJacobian.col(Spherical::PolarIndex);
    paramJacobians.col(phiIndex - 3) = bearingJacobian.col(Spherical::AzimuthIndex);

    matrix33 poseJacobian = vector3::Constant(_inverseDepth_m).asDiagonal();

    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, firstPoseIndex) = poseJacobian;
    jacobian.block<3, 3>(0, firstPoseIndex + 3) = paramJacobians;

    // convert to screen
    return get_camera_observation_projection(w2c).to_screen2d_coordinates_jacobian() * w2c.rotation() * jacobian;
}

Eigen::Matrix<double, 3, 6> InverseDepthWorldPoint::get_projected_screen3d_estimation_jacobian(
        const WorldToCameraMatrix& w2c, const double addedStandardDev) const noexcept
{
    Eigen::Matrix<double, 3, 6> inverseDepthToWorldJacobian;
    const WorldCoordinate& worldPoint = to_world_coordinates(inverseDepthToWorldJacobian, addedStandardDev);

    const matrix33& toScreenJacobian = worldPoint.to_screen_coordinates_jacobian(w2c);

    return (toScreenJacobian * inverseDepthToWorldJacobian).eval();
}

ScreenCoordinate InverseDepthWorldPoint::get_projected_screen3d_estimation(
        const WorldToCameraMatrix& w2c, const double addedStandardDev_m) const noexcept
{
    const CameraCoordinate& projectedCam = get_projected_camera_estimation(w2c, addedStandardDev_m);

    ScreenCoordinate resCoords;
    const bool res = projectedCam.to_screen_coordinates(resCoords);
    // TODO: something better than assert...
    assert(res);
    return resCoords;
}

ScreenCoordinate2D InverseDepthWorldPoint::get_projected_screen_estimation(const WorldToCameraMatrix& w2c,
                                                                           Eigen::Matrix<double, 2, 6>& jacobian,
                                                                           const double addedStandardDev) const noexcept
{
    jacobian = get_projected_screen_estimation_jacobian(w2c, addedStandardDev);
    return get_projected_screen_estimation(w2c, addedStandardDev);
}
ScreenCoordinate InverseDepthWorldPoint::get_projected_screen3d_estimation(const WorldToCameraMatrix& w2c,
                                                                           Eigen::Matrix<double, 3, 6>& jacobian,
                                                                           const double addedStandardDev) const noexcept
{
    jacobian = get_projected_screen3d_estimation_jacobian(w2c, addedStandardDev);
    return get_projected_screen3d_estimation(w2c, addedStandardDev);
}

// 2 standard dev, 95% confidence interval
// 3 standard dev, 99% confidence interval
constexpr double standardDevIntervals = 2;

Eigen::Matrix<double, 2, 6> InverseDepthWorldPoint::get_furthest_estimation_jacobian(
        const WorldToCameraMatrix& w2c, const double inverseDepthStandardDev_m) const
{
    return get_projected_screen_estimation_jacobian(w2c, standardDevIntervals * inverseDepthStandardDev_m);
}
ScreenCoordinate2D InverseDepthWorldPoint::get_furthest_estimation(const WorldToCameraMatrix& w2c,
                                                                   const double inverseDepthStandardDev_m) const
{
    return get_projected_screen_estimation(w2c, standardDevIntervals * inverseDepthStandardDev_m);
}

Eigen::Matrix<double, 2, 6> InverseDepthWorldPoint::get_closest_estimation_jacobian(
        const WorldToCameraMatrix& w2c, const double inverseDepthStandardDev_m) const
{
    return get_projected_screen_estimation_jacobian(w2c, -standardDevIntervals * inverseDepthStandardDev_m);
}
ScreenCoordinate2D InverseDepthWorldPoint::get_closest_estimation(const WorldToCameraMatrix& w2c,
                                                                  const double inverseDepthStandardDev_m) const
{
    return get_projected_screen_estimation(w2c, -standardDevIntervals * inverseDepthStandardDev_m);
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   const double inverseDepthCovariance_m,
                                                   utils::Segment<2>& screenSegment) const noexcept
{
    const double depthStandardDev_m = sqrt(inverseDepthCovariance_m);
    const ScreenCoordinate2D& firstPoint = get_furthest_estimation(w2c, depthStandardDev_m);
    const ScreenCoordinate2D& endPoint = get_closest_estimation(w2c, depthStandardDev_m);

    screenSegment.set_points(firstPoint, endPoint);
    return true;
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   const matrix66& cov,
                                                   utils::Segment<2>& screenSegment,
                                                   matrix44& covariance) const noexcept
{
    const double inverseDepthCovariance = cov.diagonal()(inverseDepthIndex);

    const double depthStandardDev = sqrt(inverseDepthCovariance);
    const ScreenCoordinate2D& firstPoint = get_furthest_estimation(w2c, depthStandardDev);
    const ScreenCoordinate2D& endPoint = get_closest_estimation(w2c, depthStandardDev);

    screenSegment.set_points(firstPoint, endPoint);

    const auto& firstPointJacobian = get_furthest_estimation_jacobian(w2c, depthStandardDev);
    const auto& secondPointJacobian = get_closest_estimation_jacobian(w2c, depthStandardDev);

    covariance.setZero();
    covariance.block<2, 2>(0, 0) = utils::propagate_covariance(cov, firstPointJacobian);
    covariance.block<2, 2>(2, 2) = utils::propagate_covariance(cov, secondPointJacobian);
    return true;
}

CameraCoordinate InverseDepthWorldPoint::get_projected_camera_estimation(const WorldToCameraMatrix& w2c,
                                                                         const double addedStandardDev_m) const noexcept
{
    return to_world_coordinates(addedStandardDev_m).to_camera_coordinates(w2c);
}

} // namespace rgbd_slam
