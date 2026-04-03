#include "point_coordinates.hpp"

#include "parameters.hpp"
#include "utils/distance_utils.hpp"
#include "covariances.hpp"
#include "types.hpp"

#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <math.h>
#include <stdexcept>

namespace rgbd_slam {

// TODO set in parameters
const double MIN_DEPTH_DISTANCE = 0.04; // (meters) is the depth camera minimum reliable distance
const double MAX_DEPTH_DISTANCE = 6.0;  // (meters) is the depth camera maximum reliable distance

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
    if (not utils::double_equal(xFrom.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: xFrom as a norm different than 1");
    }
    if (not utils::double_equal(yFrom.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: yFrom as a norm different than 1");
    }
    if (abs(yFrom.dot(xFrom)) > .01)
    {
        throw std::invalid_argument("get_transformation_matrix: yFrom and xFrom should be orthogonals");
    }

    if (not utils::double_equal(xTo.norm(), 1.0))
    {
        throw std::invalid_argument("get_transformation_matrix: xTo as a norm different than 1");
    }
    if (not utils::double_equal(yTo.norm(), 1.0))
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
vector2 transform_camera_to_screen(const vector3& cameraPoint)
{
    const static matrix33 cameraIntrinsics = Parameters::get_camera_1_intrinsics();
    return (cameraIntrinsics * cameraPoint).head<2>();
}

CameraCoordinate2D ScreenCoordinate2D::to_camera_coordinates() const
{
    return CameraCoordinate2D(transform_screen_to_camera(this->base()));
}

CameraCoordinate ScreenCoordinate2D::to_camera_coordinates_baseline(const double baseline) const
{
    ScreenCoordinate sc {this->x(), this->y(), baseline};
    return sc.to_camera_coordinates();
}

CameraCoordinate2D ScreenCoordinate2D::to_camera_coordinates(Eigen::Matrix<double, 2, 2>& jacobian) const
{
    static const vector2 cameraF = Parameters::get_camera_1_focal();

    // Jacobian of the screen to camera function
    // clang-format off
    jacobian <<
            1.0 / cameraF.x(), 0.0,
            0.0,               1.0 / cameraF.y();
    // clang-format on

    return to_camera_coordinates();
}
CameraCoordinate ScreenCoordinate2D::to_camera_coordinates_baseline(Eigen::Matrix<double, 3, 3>& jacobian,
                                                                    const double baseline) const
{
    static const vector2 cameraF = Parameters::get_camera_1_focal();
    static const vector2 cameraC = Parameters::get_camera_1_center();

    // Jacobian of the screen to camera function
    // clang-format off
    jacobian <<
            baseline / cameraF.x(), 0.0,                    (this->x() - cameraC.x()) / cameraF.x(),
            0.0,                    baseline / cameraF.y(), (this->y() - cameraC.y()) / cameraF.y(),
            0.0,                    0.0,                    1.0;
    // clang-format on

    return to_camera_coordinates_baseline(baseline);
}

matrix22 ScreenCoordinate2D::get_covariance() const
{
    // TODO xy variance should also depend on the placement of the pixel in x and y
    const double xyVariance = SQR(1.0);
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
    const matrix22& covariance2D = this->get_2D().get_covariance();

    if (not is_depth_valid(z()))
    {
        throw std::logic_error("ScreenCoordinate::get_covariance: no depth information for point");
    }

    const double depthQuantization = utils::get_depth_quantization(z());
    // a zero variance will break the kalman gain
    if (depthQuantization <= 0)
    {
        throw std::logic_error("ScreenCoordinate::get_covariance: depthQuantization should always be positive");
    }

    ScreenCoordinateCovariance cov;
    cov << covariance2D, vector2::Zero(), 0.0, 0.0, SQR(depthQuantization);

    if (not utils::is_covariance_valid(cov))
    {
        throw std::logic_error("ScreenCoordinate::get_covariance: invalid covariance");
    }
    return cov;
}

WorldCoordinateCovariance ScreenCoordinate::get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                                       const matrix33& screenCovariance,
                                                                       const CameraToWorldMatrix& cameraToWorld,
                                                                       const matrix66& poseCovariance) noexcept
{
    matrix33 screenToWorldJacobian;
    std::ignore = screenPoint.to_world_coordinates(cameraToWorld, screenToWorldJacobian);

    return ScreenCoordinate::get_world_point_covariance(screenCovariance, poseCovariance, screenToWorldJacobian);
}

WorldCoordinateCovariance ScreenCoordinate::get_world_point_covariance(const matrix33& screenCovariance,
                                                                       const matrix66& poseCovariance,
                                                                       const matrix33& screenToWorldJacobian) noexcept
{
    // TODO: Add pose rotation covariance
    const matrix33& positionCovariance = poseCovariance.block<3, 3>(0, 0);

    return WorldCoordinateCovariance {utils::propagate_covariance(screenCovariance, screenToWorldJacobian) +
                                      positionCovariance};
}

WorldCoordinate ScreenCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const
{
    return this->to_camera_coordinates().to_world_coordinates(cameraToWorld);
}

WorldCoordinate ScreenCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld,
                                                       Eigen::Matrix<double, 3, 3>& jacobian) const
{
    Eigen::Matrix<double, 3, 3> screenToCameraJacobian;
    const CameraCoordinate& camProjection = this->to_camera_coordinates(screenToCameraJacobian);

    Eigen::Matrix<double, 3, 3> cameraToWorldJacobian;
    const WorldCoordinate& worldProjection = camProjection.to_world_coordinates(cameraToWorld, cameraToWorldJacobian);

    jacobian = cameraToWorldJacobian * screenToCameraJacobian;
    return worldProjection;
}

CameraCoordinate ScreenCoordinate::to_camera_coordinates() const
{
    if (utils::double_equal(z(), 0.0))
    {
        throw std::invalid_argument("ScreenCoordinate::to_camera_coordinates: z should not be 0");
    }

    const vector2 cameraPoint = this->z() * transform_screen_to_camera(this->head<2>());
    return CameraCoordinate(cameraPoint.x(), cameraPoint.y(), z());
}

CameraCoordinate ScreenCoordinate::to_camera_coordinates(Eigen::Matrix<double, 3, 3>& jacobian) const
{
    static const vector2 cameraF = Parameters::get_camera_1_focal();
    static const vector2 cameraC = Parameters::get_camera_1_center();

    // clang-format off
    jacobian <<
            this->z() / cameraF.x(),    0.0,                        (this->x() - cameraC.x()) / cameraF.x(),
            0.0,                        this->z() / cameraF.y(),    (this->y() - cameraC.y()) / cameraF.y(),
            0.0,                        0.0,                        1.0;
    // clang-format on
    return to_camera_coordinates();
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

bool CameraCoordinate2D::to_screen_coordinates(ScreenCoordinate2D& screenPoint,
                                               Eigen::Matrix<double, 2, 2>& jacobian) const noexcept
{
    static const vector2 cameraF = Parameters::get_camera_1_focal();

    // clang-format off
    jacobian <<
            cameraF.x(),    0.0,
            0.0,            cameraF.y();
    // clang-format on

    return to_screen_coordinates(screenPoint);
}

WorldCoordinate CameraCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const noexcept
{
    return WorldCoordinate((cameraToWorld * this->homogeneous()).head<3>());
}

WorldCoordinate CameraCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld,
                                                       Eigen::Matrix<double, 3, 3>& jacobian) const noexcept
{
    // jacobian of a point transformation is just the rotation
    jacobian = cameraToWorld.rotation();

    return to_world_coordinates(cameraToWorld);
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

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate& screenPoint,
                                             Eigen::Matrix<double, 3, 3>& jacobian) const noexcept
{
    const static vector2 cameraF = Parameters::get_camera_1_focal();

    // clang-format off
    jacobian <<
        cameraF.x() / this->z(),    0.0,                        -cameraF.x() * this->x() / SQR(this->z()),
        0.0,                        cameraF.y() / this->z(),    -cameraF.y() * this->y() / SQR(this->z()),
        0.0,                        0.0,                        1.0;
    // clang-format on

    return to_screen_coordinates(screenPoint);
}

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate2D& screenPoint,
                                             Eigen::Matrix<double, 2, 3>& jacobian) const noexcept
{
    const static vector2 cameraF = Parameters::get_camera_1_focal();

    // clang-format off
    jacobian <<
        cameraF.x() / this->z(),    0.0,                        -cameraF.x() * this->x() / SQR(this->z()),
        0.0,                        cameraF.y() / this->z(),    -cameraF.y() * this->y() / SQR(this->z());
    // clang-format on

    return to_screen_coordinates(screenPoint);
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

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate& screenPoint,
                                            Eigen::Matrix<double, 3, 3>& jacobian) const noexcept
{
    Eigen::Matrix<double, 3, 3> toCameraJacobian;
    const CameraCoordinate& cameraProjection = this->to_camera_coordinates(worldToCamera, toCameraJacobian);

    Eigen::Matrix<double, 3, 3> toScreenJacobian;
    ScreenCoordinate screenProjection;
    if (cameraProjection.to_screen_coordinates(screenProjection, toScreenJacobian))
    {
        screenPoint = screenProjection;
        jacobian = toScreenJacobian * toCameraJacobian;
        return true;
    }
    return false;
}

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate2D& screenPoint,
                                            Eigen::Matrix<double, 2, 3>& jacobian) const noexcept
{
    Eigen::Matrix<double, 3, 3> toCameraJacobian;
    const CameraCoordinate& cameraProjection = this->to_camera_coordinates(worldToCamera, toCameraJacobian);

    Eigen::Matrix<double, 2, 3> toScreenJacobian;
    ScreenCoordinate2D screenProjection;
    if (cameraProjection.to_screen_coordinates(screenProjection, toScreenJacobian))
    {
        screenPoint = screenProjection;
        jacobian = toScreenJacobian * toCameraJacobian;
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

vector3 WorldCoordinate::get_signed_distance(const ScreenCoordinate& screenPoint,
                                             const CameraToWorldMatrix& cameraToWorld) const
{
    return this->base() - screenPoint.to_world_coordinates(cameraToWorld);
}

double WorldCoordinate::get_distance(const ScreenCoordinate& screenPoint,
                                     const CameraToWorldMatrix& cameraToWorld) const
{
    return get_signed_distance(screenPoint, cameraToWorld).lpNorm<1>();
}

CameraCoordinate WorldCoordinate::to_camera_coordinates(const WorldToCameraMatrix& worldToCamera) const noexcept
{
    // WorldCoordinate
    const vector4& cameraHomogenousCoordinates = worldToCamera * this->homogeneous();
    return CameraCoordinate(cameraHomogenousCoordinates);
}

CameraCoordinate WorldCoordinate::to_camera_coordinates(const WorldToCameraMatrix& worldToCamera,
                                                        Eigen::Matrix<double, 3, 3>& jacobian) const noexcept
{
    jacobian = worldToCamera.rotation();

    return to_camera_coordinates(worldToCamera);
}

} // namespace rgbd_slam
