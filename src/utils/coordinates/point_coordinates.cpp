#include "point_coordinates.hpp"
#include "../parameters.hpp"
#include "../utils/distance_utils.hpp"
#include "covariances.hpp"
#include "types.hpp"
#include <cmath>
#include <math.h>
#include <stdexcept>

namespace rgbd_slam::utils {

// TODO set in parameters
const double MIN_DEPTH_DISTANCE = 40;   // (millimeters) is the depth camera minimum reliable distance
const double MAX_DEPTH_DISTANCE = 6000; // (millimeters) is the depth camera maximum reliable distance

bool is_depth_valid(const double depth) noexcept
{
    return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
}

matrix44 get_transformation_matrix(const vector3& xFrom,
                                   const vector3& yFrom,
                                   const vector3& centerFrom,
                                   const vector3& xTo,
                                   const vector3& yTo,
                                   const vector3& centerTo)
{
    // sanity checks
    if (not double_equal(xFrom.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: xFrom as a norm different than 1");
    }
    if (not double_equal(yFrom.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: yFrom as a norm different than 1");
    }
    if (abs(yFrom.dot(xFrom)) > .01)
    {
        throw std::invalid_argument("get_transformation_matrix: yFrom and xFrom should be orthogonals");
    }

    if (not double_equal(xTo.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: xTo as a norm different than 1");
    }
    if (not double_equal(yTo.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: yTo as a norm different than 1");
    }
    if (abs(yTo.dot(xTo)) > .01)
    {
        throw std::invalid_argument("get_transformation_matrix: yTo and xTo should be orthogonals");
    }

    Eigen::Affine3d T1 = Eigen::Affine3d::Identity();
    T1.linear() << xFrom, yFrom, xFrom.cross(yFrom); // get coordinate system
    T1.translation() << centerFrom;

    Eigen::Affine3d T2 = Eigen::Affine3d::Identity();
    T2.linear() << xTo, yTo, xTo.cross(yTo); // get next coordinate system
    T2.translation() << centerTo;

    matrix44 res = matrix44::Identity(); // make bottom row of Matrix 0,0,0,1
    res.block<3, 3>(0, 0) = T2.rotation() * T1.rotation().inverse();
    res.block<3, 1>(0, 3) = T2.translation() - T1.translation();
    return res;
}

/**
 *      SCREEN COORDINATES
 */

/**
 * \brief Transform screen coordinates to camera coordinates
 */
vector2 transform_screen_to_camera(const vector2& screenPoint)
{
    const static matrix33 cameraIntrinsics = Parameters::get_camera_1_intrinsics().inverse();
    return (cameraIntrinsics * screenPoint.homogeneous()).head<2>();
}

/**
 * \brief Transform camera coordinates to screen coordinates
 */
vector2 transform_camera_to_screen(const vector3& screenPoint)
{
    const static matrix33 cameraIntrinsics = Parameters::get_camera_1_intrinsics();
    return (cameraIntrinsics * screenPoint).head<2>();
}

CameraCoordinate2D ScreenCoordinate2D::to_camera_coordinates() const
{
    return CameraCoordinate2D(transform_screen_to_camera(this->base()));
}

matrix22 ScreenCoordinate2D::get_covariance() const
{
    // TODO xy variance should also depend on the placement of the pixel in x and y
    const double xyVariance = SQR(0.1);
    matrix22 cov({{xyVariance, 0.0}, {0.0, xyVariance}});

    if (not utils::is_covariance_valid(cov))
    {
        throw std::logic_error("ScreenCoordinate2D::get_covariance: invalid covariance");
    }

    return cov;
}

bool ScreenCoordinate2D::is_in_screen_boundaries() const noexcept
{
    static const uint screenSizeX = Parameters::get_camera_1_image_size().x();
    static const uint screenSizeY = Parameters::get_camera_1_image_size().y();

    return
            // in screen space
            x() >= 0 and x() <= screenSizeX and y() >= 0 and y() <= screenSizeY;
}

ScreenCoordinateCovariance ScreenCoordinate::get_covariance() const
{
    const double xyVariance = SQR(0.1);
    const matrix22 covariance2D({{xyVariance, 0.0}, {0.0, xyVariance}});

    const double depthQuantization = utils::is_depth_valid(z()) ? get_depth_quantization(z()) : 1000.0;
    // a zero variance will break the kalman gain
    if (depthQuantization <= 0)
    {
        throw std::logic_error("ScreenCoordinate::get_covariance: depthQuantization should always be positive");
    }

    ScreenCoordinateCovariance cov;
    cov << covariance2D, vector2::Zero(), 0.0, 0.0, depthQuantization;

    if (not utils::is_covariance_valid(cov))
    {
        throw std::logic_error("ScreenCoordinate::get_covariance: invalid covariance");
    }
    return cov;
}

WorldCoordinate ScreenCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const
{
    return this->to_camera_coordinates().to_world_coordinates(cameraToWorld);
}

CameraCoordinate ScreenCoordinate::to_camera_coordinates() const
{
    if (x() < 0)
    {
        throw std::invalid_argument("ScreenCoordinate::to_camera_coordinates: x should be >= 0");
    }
    if (y() < 0)
    {
        throw std::invalid_argument("ScreenCoordinate::to_camera_coordinates: y should be >= 0");
    }
    if (double_equal(z(), 0.0))
    {
        throw std::invalid_argument("ScreenCoordinate::to_camera_coordinates: z should not be 0");
    }

    const vector2 cameraPoint = this->z() * transform_screen_to_camera(this->head<2>());
    return CameraCoordinate(cameraPoint.x(), cameraPoint.y(), z());
}

bool ScreenCoordinate::is_in_screen_boundaries() const noexcept
{
    static const uint screenSizeX = Parameters::get_camera_1_image_size().x();
    static const uint screenSizeY = Parameters::get_camera_1_image_size().y();

    return
            // in screen space
            x() >= 0 and x() <= screenSizeX and y() >= 0 and y() <= screenSizeY and
            // in front of the camera
            z() > 0;
}

/**
 *      CAMERA COORDINATES
 */

bool CameraCoordinate2D::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const noexcept
{
    const vector2 screenCoordinates = transform_camera_to_screen(this->homogeneous());
    if (not screenCoordinates.hasNaN())
    {
        screenPoint = ScreenCoordinate2D(screenCoordinates.x(), screenCoordinates.y());
        return true;
    }
    return false;
}

WorldCoordinate CameraCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const noexcept
{
    return WorldCoordinate((cameraToWorld * this->homogeneous()).head<3>());
}

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate& screenPoint) const noexcept
{
    const vector2 screenCoordinates = 1.0 / z() * transform_camera_to_screen(this->base());
    if (not screenCoordinates.hasNaN())
    {
        screenPoint = ScreenCoordinate(screenCoordinates.x(), screenCoordinates.y(), z());
        return true;
    }
    return false;
}

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const noexcept
{
    ScreenCoordinate screenCoordinates;
    if (to_screen_coordinates(screenCoordinates))
    {
        screenPoint = ScreenCoordinate2D(screenCoordinates.x(), screenCoordinates.y());
        return true;
    }
    return false;
}

/**
 *      WORLD COORDINATES
 */

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate& screenPoint) const noexcept
{
    return this->to_camera_coordinates(worldToCamera).to_screen_coordinates(screenPoint);
}

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate2D& screenPoint) const noexcept
{
    ScreenCoordinate screenCoordinates;
    if (to_screen_coordinates(worldToCamera, screenCoordinates))
    {
        screenPoint = ScreenCoordinate2D(screenCoordinates.x(), screenCoordinates.y());
        return true;
    }
    return false;
}

vector2 WorldCoordinate::get_signed_distance_2D_px(const ScreenCoordinate2D& screenPoint,
                                                   const WorldToCameraMatrix& worldToCamera) const
{
    ScreenCoordinate2D projectedScreenPoint;
    if (to_screen_coordinates(worldToCamera, projectedScreenPoint))
    {
        vector2 distance = screenPoint - projectedScreenPoint;
        if (distance.hasNaN())
        {
            throw std::invalid_argument("WorldCoordinate::get_signed_distance_2D_px: distance as some NaN");
        }
        return distance;
    }
    // high number
    return vector2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
}

double WorldCoordinate::get_distance_px(const ScreenCoordinate2D& screenPoint,
                                        const WorldToCameraMatrix& worldToCamera) const
{
    const vector2& distance2D = get_signed_distance_2D_px(screenPoint, worldToCamera);
    if (distance2D.x() >= std::numeric_limits<double>::max() or distance2D.y() >= std::numeric_limits<double>::max())
        // high number
        return std::numeric_limits<double>::max();
    // compute manhattan distance (norm of power 1)
    const double distance = distance2D.lpNorm<1>();

    if (std::isnan(distance) or distance < 0)
    {
        throw std::logic_error("WorldCoordinate::get_distance_px: found an invalid distance");
    }

    return distance;
}

vector3 WorldCoordinate::get_signed_distance_mm(const ScreenCoordinate& screenPoint,
                                                const CameraToWorldMatrix& cameraToWorld) const
{
    return this->base() - screenPoint.to_world_coordinates(cameraToWorld);
}

double WorldCoordinate::get_distance_mm(const ScreenCoordinate& screenPoint,
                                        const CameraToWorldMatrix& cameraToWorld) const
{
    return get_signed_distance_mm(screenPoint, cameraToWorld).lpNorm<1>();
}

/**
 *      CAMERA COORDINATES
 */

CameraCoordinate WorldCoordinate::to_camera_coordinates(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    // WorldCoordinate
    const vector4& cameraHomogenousCoordinates = worldToCamera * this->homogeneous();
    return CameraCoordinate(cameraHomogenousCoordinates);
}

/**
 *      INVERSE DEPTH COORDINATES
 */

InverseDepthWorldPoint::InverseDepthWorldPoint(const utils::ScreenCoordinate2D& observation,
                                               const CameraToWorldMatrix& c2w) :
    // use homogenous to create a vector from the camera center outward
    InverseDepthWorldPoint(utils::CameraCoordinate(observation.to_camera_coordinates().homogeneous()), c2w)
{
}

InverseDepthWorldPoint::InverseDepthWorldPoint(const utils::CameraCoordinate& observation,
                                               const CameraToWorldMatrix& c2w)
{
    from_cartesian(observation.to_world_coordinates(c2w), WorldCoordinate(c2w.translation()));

    constexpr double dmin = 1000;
    _inverseDepth_mm = (1.0 / dmin) / 2; // 10 meters baseline (infinity is approx to 10 meters right ?)
}

void InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point, const WorldCoordinate& origin) noexcept
{
    _firstObservation = origin;

    const vector3 directionalVector(point - _firstObservation);

    _theta_rad = atan2(directionalVector.x(), directionalVector.z());
    _phi_rad = atan2(-directionalVector.y(), sqrt(SQR(directionalVector.x()) + SQR(directionalVector.z())));
    _inverseDepth_mm = 1.0 / point.z();
}

void InverseDepthWorldPoint::from_cartesian(const WorldCoordinate& point,
                                            const WorldCoordinate& origin,
                                            Eigen::Matrix<double, 6, 3>& jacobian) noexcept
{
    // this is the jacobian of
    // theta = atan2(x, z)
    // phi = atan2(-y, sqrt(x*x + z*z))
    // invDepth = 1/z
    const vector3 directionalVector(point - _firstObservation);

    jacobian = Eigen::Matrix<double, 6, 3>::Zero();

    const double xSqr = SQR(directionalVector.x());
    const double ySqr = SQR(directionalVector.y());
    const double zSqr = SQR(directionalVector.z());

    const double oneOverXZ = 1.0 / (xSqr + zSqr);
    const double sqrtXZ = sqrt(xSqr + zSqr);
    const double theta1 = 1.0 / (sqrtXZ * (xSqr + ySqr + zSqr));

    jacobian(3, 0) = directionalVector.z() * oneOverXZ;
    jacobian(3, 2) = -directionalVector.x() * oneOverXZ;

    jacobian(4, 0) = directionalVector.x() * directionalVector.y() * theta1;
    jacobian(4, 1) = -sqrtXZ / (xSqr + ySqr + zSqr);
    jacobian(4, 2) = directionalVector.y() * directionalVector.z() * theta1;

    jacobian(5, 2) = -1 / zSqr;

    from_cartesian(point, origin);
}

utils::WorldCoordinate InverseDepthWorldPoint::to_world_coordinates() const noexcept
{
    assert(_inverseDepth_mm != 0.0);
    return utils::WorldCoordinate(_firstObservation + 1.0 / _inverseDepth_mm * get_bearing_vector());
}

utils::WorldCoordinate InverseDepthWorldPoint::to_world_coordinates(
        Eigen::Matrix<double, 3, 6>& jacobian) const noexcept
{
    // jacobian of:
    // _firstObservation + 1.0 / _inverseDepth_mm * get_bearing_vector()
    jacobian = Eigen::Matrix<double, 3, 6>::Zero();
    jacobian.block<3, 3>(0, 0) = matrix33::Identity();

    // compute sin and cos in advance (opti)
    const double cosTheta = cos(_theta_rad);
    const double sinTheta = sin(_theta_rad);
    const double cosPhi = cos(_phi_rad);
    const double sinPhi = sin(_phi_rad);

    const double depth = 1.0 / _inverseDepth_mm;
    const double depthSqr = 1.0 / SQR(_inverseDepth_mm);
    const double cosPhiSinTheta = cosPhi * sinTheta;
    const double cosPhiCosTheta = cosPhi * cosTheta;

    jacobian.block<3, 3>(0, 3) =
            matrix33({{cosPhiCosTheta * depth, -sinPhi * sinTheta * depth, -cosPhiSinTheta * depthSqr},
                      {0, -cosPhi * depth, sinPhi * depthSqr},
                      {-cosPhiSinTheta * depth, -cosTheta * sinPhi * depth, -cosPhiCosTheta * depthSqr}});

    return to_world_coordinates();
}

utils::CameraCoordinate InverseDepthWorldPoint::to_camera_coordinates(const WorldToCameraMatrix& w2c) const noexcept
{
    // this is a transformation of retroprojection of the inverse depth (world) to camera
    return to_world_coordinates().to_camera_coordinates(w2c);
    // this as the advantage of handling nicely a point at infinite distance (_inverseDepth_mm == 0)
    // return utils::CameraCoordinate(w2c.rotation() * (_inverseDepth_mm * (_firstObservation - w2c.translation()) +
    // get_bearing_vector()));
}

bool InverseDepthWorldPoint::to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                                   utils::ScreenCoordinate2D& screenCoordinates) const noexcept
{
    return to_camera_coordinates(w2c).to_screen_coordinates(screenCoordinates);
}

vector3 InverseDepthWorldPoint::get_bearing_vector() const noexcept
{
    const double cosPhi = cos(_phi_rad);
    return vector3(cosPhi * sin(_theta_rad), -sin(_phi_rad), cosPhi * cos(_theta_rad)).normalized();
}

vector6 InverseDepthWorldPoint::get_vector_state() const noexcept
{
    return vector6(_firstObservation.x(),
                   _firstObservation.y(),
                   _firstObservation.z(),
                   _theta_rad,
                   _phi_rad,
                   _inverseDepth_mm);
}

void InverseDepthWorldPoint::from_vector_state(const vector6& state) noexcept
{
    // do not change the translation (for now !)
    _theta_rad = state(3);
    _phi_rad = state(4);
    _inverseDepth_mm = state(5);
}

} // namespace rgbd_slam::utils