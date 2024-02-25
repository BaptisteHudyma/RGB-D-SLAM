#include "inverse_depth_coordinates.hpp"

#include "camera_transformation.hpp"
#include "coordinates/basis_changes.hpp"

#include "coordinates/point_coordinates.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "parameters.hpp"
#include "types.hpp"

namespace rgbd_slam {

/**
 *      INVERSE DEPTH COORDINATES
 */

InverseDepthWorldPoint::InverseDepthWorldPoint(const WorldCoordinate& firstPose,
                                               const double inverseDepth,
                                               const double theta,
                                               const double phi) :
    _firstObservation(firstPose),
    _inverseDepth_mm(inverseDepth),
    _theta_rad(theta),
    _phi_rad(phi),
    _bearingVector(Cartesian::from(Spherical(1.0, _theta_rad, _phi_rad)).vec())
{
    if (_inverseDepth_mm < 0.0)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Inverse depth should be >= 0");
    if (_theta_rad < 0.0 or _theta_rad > M_PI)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Theta should be in [0, M_PI]");
    if (_phi_rad < -M_PI or _phi_rad > M_PI)
        throw std::invalid_argument("Constructor of InverseDepthWorldPoint: Phi should be in [-Pi, Pi]");
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const ScreenCoordinate2D& observation, const CameraToWorldMatrix& c2w) :
    // use homogenous to create a vector from the camera center outward
    InverseDepthWorldPoint(CameraCoordinate(observation.to_camera_coordinates().homogeneous()), c2w)
{
    // no known depth, so set the baseline
    _inverseDepth_mm = parameters::detection::inverseDepthBaseline / 2.0;
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const CameraCoordinate& observation, const CameraToWorldMatrix& c2w) :
    InverseDepthWorldPoint(from_cartesian(observation.to_world_coordinates(c2w), WorldCoordinate(c2w.translation())))
{
}

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
    return InverseDepthWorldPoint(origin, 1.0 / s.p, s.theta, s.phi);
}

InverseDepthWorldPoint InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point,
                                                              const WorldCoordinate& origin,
                                                              Eigen::Matrix<double, 6, 3>& jacobian) noexcept
{
    jacobian.setZero();
    jacobian.block<3, 3>(firstPoseIndex, firstPoseIndex) = matrix33::Zero();

    // jacobian of the [xo, yo, zo, x, y, z] =>
    // [xo, yo, zo, inverse depth spherical projection of (x - xo, y - yo, z - zo)]

    const vector3 v(point - origin);
    const double theta1 = SQR(v.x()) + SQR(v.y());
    const double theta5 = theta1 + SQR(v.z());
    const double theta4 = 1.0 / pow(theta5, 3.0 / 2.0);

    const double oneOverTheta1 = 1.0 / theta1;
    const double sqrtTheta1 = sqrt(theta1);
    const double sqrtTheta1Theta5 = 1.0 / (sqrtTheta1 * theta5);

    matrix33 jac({{-v.x() * theta4, -v.y() * theta4, -v.z() * theta4},
                  {v.x() * v.z() * sqrtTheta1Theta5, v.y() * v.z() * sqrtTheta1Theta5, -sqrtTheta1 / theta5},
                  {-v.y() * oneOverTheta1, v.x() * oneOverTheta1, 0}});

    jacobian.block<3, 3>(inverseDepthIndex, 0) = jac;
    return from_cartesian(point, origin);
}

WorldCoordinate InverseDepthWorldPoint::to_world_coordinates() const noexcept
{
    assert(_inverseDepth_mm != 0.0);
    return WorldCoordinate(_firstObservation + _bearingVector / _inverseDepth_mm);
}

Eigen::Matrix<double, 3, 6> to_world_coordinates_jacobian(const double inverseDepth,
                                                          const double theta,
                                                          const double phi)
{
    // jacobian of _firstObservation + 1.0 / _inverseDepth_mm * _bearingVector
    matrixd jacobian = Eigen::Matrix<double, 3, 6>::Zero();

    // x = xo + sin(theta) cos(phi) / d
    // y = yo + sin(theta) sin(phi) / d
    // z = zo + cos(theta) / d

    const double sinTheta = sin(theta);
    const double cosTheta = cos(theta);
    const double sinPhi = sin(phi);
    const double cosPhi = cos(phi);
    const double d = 1.0 / inverseDepth;
    const double dSqr = 1.0 / SQR(inverseDepth);

    const double theta1 = sinPhi * sinTheta;
    const double theta2 = cosPhi * sinTheta;
    const double cosThetaOverd = cosTheta * d;

    const matrix33 reducedJacobian({{-theta2 * dSqr, cosPhi * cosThetaOverd, -theta1 * d},
                                    {-theta1 * dSqr, sinPhi * cosThetaOverd, theta2 * d},
                                    {-cosTheta * dSqr, -sinTheta * d, 0}});

    jacobian.block<3, 3>(0, InverseDepthWorldPoint::firstPoseIndex) = matrix33::Identity();
    jacobian.block<3, 3>(0, InverseDepthWorldPoint::inverseDepthIndex) = reducedJacobian;

    return jacobian;
}

WorldCoordinate InverseDepthWorldPoint::to_world_coordinates(Eigen::Matrix<double, 3, 6>& jacobian) const noexcept
{
    jacobian = to_world_coordinates_jacobian(_inverseDepth_mm, _theta_rad, _phi_rad);
    return to_world_coordinates();
}

WorldCoordinate InverseDepthWorldPoint::get_furthest_estimation(const double inverseDepthStandardDev) const
{
    // 3 standard deviations (>99% of certainty in this interval)
    const double depthVariation = inverseDepthStandardDev * 3;
    return WorldCoordinate(_firstObservation + _bearingVector / (_inverseDepth_mm - depthVariation));
}

WorldCoordinate InverseDepthWorldPoint::get_closest_estimation(const double inverseDepthStandardDev) const
{
    // 3 standard deviations (>99% of certainty in this interval)
    const double depthVariation = inverseDepthStandardDev * 3;
    return WorldCoordinate(_firstObservation + _bearingVector / (_inverseDepth_mm + depthVariation));
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   const double inverseDepthCovariance,
                                                   utils::Segment<2>& screenSegment) const noexcept
{
    const double depthStandardDev = sqrt(inverseDepthCovariance);
    const WorldCoordinate& firstPoint = get_furthest_estimation(depthStandardDev);
    const WorldCoordinate& endPoint = get_closest_estimation(depthStandardDev);

    ScreenCoordinate2D firstScreenPoint;
    ScreenCoordinate2D endScreenPoint;
    if (firstPoint.to_screen_coordinates(w2c, firstScreenPoint) and endPoint.to_screen_coordinates(w2c, endScreenPoint))
    {
        screenSegment.set_points(firstScreenPoint, endScreenPoint);
        return true;
    }
    // should never happend
    return false;
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   const matrix66& cov,
                                                   utils::Segment<2>& screenSegment,
                                                   matrix44& covariance) const noexcept
{
    const double inverseDepthCovariance = cov.diagonal()(inverseDepthIndex);

    const double depthStandardDev = sqrt(inverseDepthCovariance);
    const WorldCoordinate& firstPoint = get_furthest_estimation(depthStandardDev);
    const WorldCoordinate& endPoint = get_closest_estimation(depthStandardDev);

    ScreenCoordinate2D firstScreenPoint;
    ScreenCoordinate2D endScreenPoint;
    if (firstPoint.to_screen_coordinates(w2c, firstScreenPoint) and endPoint.to_screen_coordinates(w2c, endScreenPoint))
    {
        screenSegment.set_points(firstScreenPoint, endScreenPoint);

        const Eigen::Matrix<double, 3, 6>& firstPointJacobian =
                to_world_coordinates_jacobian(_inverseDepth_mm - 3.0 * depthStandardDev, _theta_rad, _phi_rad);
        const Eigen::Matrix<double, 3, 6>& secondPointJacobian =
                to_world_coordinates_jacobian(_inverseDepth_mm + 3.0 * depthStandardDev, _theta_rad, _phi_rad);

        covariance.setZero();
        covariance.block<2, 2>(0, 0) =
                utils::get_screen_2d_point_covariance(
                        firstPoint,
                        WorldCoordinateCovariance(firstPointJacobian * cov.selfadjointView<Eigen::Lower>() *
                                                  firstPointJacobian.transpose()),
                        w2c)
                        .selfadjointView<Eigen::Lower>();
        covariance.block<2, 2>(2, 2) =
                utils::get_screen_2d_point_covariance(
                        endPoint,
                        WorldCoordinateCovariance(secondPointJacobian * cov.selfadjointView<Eigen::Lower>() *
                                                  secondPointJacobian.transpose()),
                        w2c)
                        .selfadjointView<Eigen::Lower>();
        return true;
    }
    // should never happend
    return false;
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   const double inverseDepthCovariance,
                                                   utils::Segment<3>& screenSegment) const noexcept
{
    const double depthStandardDev = sqrt(inverseDepthCovariance);
    const WorldCoordinate& firstPoint = get_furthest_estimation(depthStandardDev);
    const WorldCoordinate& endPoint = get_closest_estimation(depthStandardDev);

    ScreenCoordinate firstScreenPoint;
    ScreenCoordinate endScreenPoint;
    if (firstPoint.to_screen_coordinates(w2c, firstScreenPoint) and endPoint.to_screen_coordinates(w2c, endScreenPoint))
    {
        screenSegment.set_points(firstScreenPoint, endScreenPoint);
        return true;
    }

    return false;
}

} // namespace rgbd_slam