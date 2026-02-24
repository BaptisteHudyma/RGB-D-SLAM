#include "inverse_depth_coordinates.hpp"

#include "camera_transformation.hpp"
#include "coordinates/basis_changes.hpp"

#include "distance_utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {

/**
 *      INVERSE DEPTH COORDINATES
 */

InverseDepthWorldPoint::InverseDepthWorldPoint(const WorldCoordinate& firstPose,
                                               const double inverseDepth_m,
                                               const double theta,
                                               const double phi) :
    _firstObservation(firstPose / 1000.0),
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
    // use homogenous to create a vector from the camera center outward
    InverseDepthWorldPoint(CameraCoordinate(observation.to_camera_coordinates().homogeneous()), c2w)
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

    const Spherical& s = Spherical::from(Cartesian(directionalVector / 1000.0));
    return InverseDepthWorldPoint(origin, 1.0 / s.p, s.polar_rad, s.azimuth_rad);
}

InverseDepthWorldPoint InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point,
                                                              const WorldCoordinate& origin,
                                                              Eigen::Matrix<double, 6, 3>& jacobian) noexcept
{
    jacobian.setZero();
    // TODO: should be Identity ?
    jacobian.block<3, 3>(firstPoseIndex, firstPoseIndex) = matrix33::Zero();

    // jacobian of the [xo, yo, zo, x, y, z] =>
    // [xo, yo, zo, inverse depth spherical projection of (x - xo, y - yo, z - zo)]

    const vector3 directionalVector(point - origin);

    matrix33 toCartesianJacobian;
    const auto& s = Spherical::from(Cartesian(directionalVector / 1000.0), toCartesianJacobian);

    // add the 1/d part
    toCartesianJacobian.row(0) *= -1.0 / s.p;

    jacobian.block<3, 3>(inverseDepthIndex, 0) = toCartesianJacobian;
    return from_cartesian(point, origin);
}

WorldCoordinate InverseDepthWorldPoint::to_world_coordinates(const double addedStandardDev_m) const noexcept
{
    assert(_inverseDepth_m != 0.0);
    return WorldCoordinate(1000.0 * (_firstObservation + _bearingVector / (_inverseDepth_m + addedStandardDev_m)));
}

Eigen::Matrix<double, 3, 6> to_world_coordinates_jacobian(const double inverseDepth,
                                                          const double theta,
                                                          const double phi)
{
    // jacobian of _firstObservation + 1.0 / _inverseDepth_m * _bearingVector

    matrix33 bearingJacobian;
    Cartesian::from(Spherical(1.0, theta, phi), bearingJacobian);

    // Add derivation of 1/d
    bearingJacobian.col(0) *= -1.0 / SQR(inverseDepth);
    bearingJacobian.col(1) *= 1.0 / inverseDepth;
    bearingJacobian.col(2) *= 1.0 / inverseDepth;

    Eigen::Matrix<double, 3, 6> jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    jacobian.block<3, 3>(0, InverseDepthWorldPoint::firstPoseIndex) = 1000.0 * matrix33::Identity();
    jacobian.block<3, 3>(0, InverseDepthWorldPoint::inverseDepthIndex) = 1000.0 * bearingJacobian;

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
    const auto& c2w = utils::compute_camera_to_world_transform(w2c);
    // limit a maximum distance of 100 meters, the depth cannot be observed behind the camera
    const double realDepthvalue = std::max(1.0 / 100.0, _inverseDepth_m + addedStandardDev_m);

    const CameraCoordinate projectedCam =
            w2c.rotation() * 1000.0 *
            (realDepthvalue * (_firstObservation - c2w.translation() / 1000.0) + _bearingVector);

    ScreenCoordinate2D resCoords;
    const bool res = projectedCam.to_screen_coordinates(resCoords);
    // TODO: something better than assert...
    assert(res);
    return resCoords;
}

// 2 standard dev, 95% confidence interval
// 3 standard dev, 99% confidence interval
constexpr double standardDevIntervals = 3;

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

} // namespace rgbd_slam
