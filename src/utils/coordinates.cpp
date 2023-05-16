#include "coordinates.hpp"
#include "../parameters.hpp"
#include "../utils/distance_utils.hpp"
#include "camera_transformation.hpp"
#include "covariances.hpp"
#include "types.hpp"
#include <cmath>
#include <math.h>

namespace rgbd_slam::utils {

// TODO set in parameters
const double MIN_DEPTH_DISTANCE = 40;   // (millimeters) is the depth camera minimum reliable distance
const double MAX_DEPTH_DISTANCE = 6000; // (millimeters) is the depth camera maximum reliable distance

bool is_depth_valid(const double depth) { return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE); }

matrix44 get_transformation_matrix(const vector3& xFrom,
                                   const vector3& yFrom,
                                   const vector3& centerFrom,
                                   const vector3& xTo,
                                   const vector3& yTo,
                                   const vector3& centerTo)
{
    // sanity checks
    assert(double_equal(xFrom.norm(), 1.0));
    assert(double_equal(yFrom.norm(), 1.0));
    assert(abs(yFrom.dot(xFrom)) <= .01);

    assert(double_equal(xTo.norm(), 1.0));
    assert(double_equal(yTo.norm(), 1.0));
    assert(abs(yTo.dot(xTo)) <= .01);

    Eigen::Affine3d T1 = Eigen::Affine3d::Identity();
    T1.linear() << xFrom, yFrom, xFrom.cross(yFrom); // get coordinate system
    T1.translation() << centerFrom;

    Eigen::Affine3d T2 = Eigen::Affine3d::Identity();
    T2.linear() << xTo, yTo, xTo.cross(yTo); // get next coordinate system
    T2.translation() << centerTo;

    // T = transform to CS2 to CS3
    return T2.matrix() * T1.matrix().inverse();
}

/**
 *      SCREEN COORDINATES
 */

CameraCoordinate2D ScreenCoordinate2D::to_camera_coordinates() const
{
    assert(x() >= 0 and y() >= 0);

    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    const double x = (this->x() - cameraCX) / cameraFX;
    const double y = (this->y() - cameraCY) / cameraFY;

    CameraCoordinate2D cameraPoint(x, y);
    return cameraPoint;
}

matrix22 ScreenCoordinate2D::get_covariance() const
{
    // TODO xy variance should also depend on the placement of the pixel in x and y
    const double xyVariance = 0.1 * 0.1;
    return matrix22({{xyVariance, 0.0}, {0.0, xyVariance}});
}

bool ScreenCoordinate2D::is_in_screen_boundaries() const
{
    static const uint screenSizeX = Parameters::get_camera_1_size_x();
    static const uint screenSizeY = Parameters::get_camera_1_size_y();

    return
            // in screen space
            x() >= 0 and x() <= screenSizeX and y() >= 0 and y() <= screenSizeY;
}

ScreenCoordinateCovariance ScreenCoordinate::get_covariance() const
{
    const double xyVariance = 0.1 * 0.1;
    const matrix22 covariance2D({{xyVariance, 0.0}, {0.0, xyVariance}});

    const double depthQuantization = utils::is_depth_valid(z()) ? get_depth_quantization(z()) : 1000.0;
    // a zero variance will break the kalman gain
    assert(depthQuantization > 0);

    ScreenCoordinateCovariance cov;
    cov << covariance2D, vector2::Zero(), 0.0, 0.0, depthQuantization;
    return cov;
}

WorldCoordinate ScreenCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const
{
    const CameraCoordinate& cameraPoint = this->to_camera_coordinates();
    return cameraPoint.to_world_coordinates(cameraToWorld);
}

CameraCoordinate ScreenCoordinate::to_camera_coordinates() const
{
    assert(x() >= 0 and y() >= 0);
    assert(not double_equal(z(), 0.0));

    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    const double x = (this->x() - cameraCX) * this->z() / cameraFX;
    const double y = (this->y() - cameraCY) * this->z() / cameraFY;

    CameraCoordinate cameraPoint(x, y, z());
    return cameraPoint;
}

bool ScreenCoordinate::is_in_screen_boundaries() const
{
    static const uint screenSizeX = Parameters::get_camera_1_size_x();
    static const uint screenSizeY = Parameters::get_camera_1_size_y();

    return
            // in screen space
            x() >= 0 and x() <= screenSizeX and y() >= 0 and y() <= screenSizeY and
            // in front of the camera
            z() > 0;
}

/**
 *      CAMERA COORDINATES
 */

bool CameraCoordinate2D::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    const double screenX = cameraFX * x() + cameraCX;
    const double screenY = cameraFY * y() + cameraCY;
    if (not std::isnan(screenX) and not std::isnan(screenY))
    {
        screenPoint.x() = screenX;
        screenPoint.y() = screenY;
        return true;
    }
    return false;
}

WorldCoordinate CameraCoordinate::to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const
{
    const vector4 homogenousWorldCoords = cameraToWorld * this->get_homogenous();
    return WorldCoordinate(homogenousWorldCoords.head<3>());
}

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate& screenPoint) const
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    const double screenX = cameraFX * x() / z() + cameraCX;
    const double screenY = cameraFY * y() / z() + cameraCY;
    if (not std::isnan(screenX) and not std::isnan(screenY))
    {
        screenPoint = ScreenCoordinate(screenX, screenY, z());
        return true;
    }
    return false;
}

bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const
{
    const static double cameraFX = Parameters::get_camera_1_focal_x();
    const static double cameraFY = Parameters::get_camera_1_focal_y();
    const static double cameraCX = Parameters::get_camera_1_center_x();
    const static double cameraCY = Parameters::get_camera_1_center_y();

    const double screenX = cameraFX * x() / z() + cameraCX;
    const double screenY = cameraFY * y() / z() + cameraCY;
    if (not std::isnan(screenX) and not std::isnan(screenY))
    {
        screenPoint = ScreenCoordinate2D(screenX, screenY);
        return true;
    }
    return false;
}

/**
 *      WORLD COORDINATES
 */

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate& screenPoint) const
{
    assert(not this->hasNaN());

    const CameraCoordinate& cameraPoint = this->to_camera_coordinates(worldToCamera);
    assert(cameraPoint.get_homogenous()[3] > 0);

    return cameraPoint.to_screen_coordinates(screenPoint);
}

bool WorldCoordinate::to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                            ScreenCoordinate2D& screenPoint) const
{
    if (ScreenCoordinate screenCoordinates; to_screen_coordinates(worldToCamera, screenCoordinates))
    {
        screenPoint.x() = screenCoordinates.x();
        screenPoint.y() = screenCoordinates.y();
        return true;
    }
    return false;
}

vector2 WorldCoordinate::get_signed_distance_2D(const ScreenCoordinate2D& screenPoint,
                                                const WorldToCameraMatrix& worldToCamera) const
{
    if (ScreenCoordinate2D projectedScreenPoint; to_screen_coordinates(worldToCamera, projectedScreenPoint))
    {
        vector2 distance = screenPoint - projectedScreenPoint;
        assert(not distance.hasNaN());
        return distance;
    }
    // high number
    return vector2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
}

double WorldCoordinate::get_distance(const ScreenCoordinate2D& screenPoint,
                                     const WorldToCameraMatrix& worldToCamera) const
{
    const vector2& distance2D = get_signed_distance_2D(screenPoint, worldToCamera);
    if (distance2D.x() >= std::numeric_limits<double>::max() or distance2D.y() >= std::numeric_limits<double>::max())
        // high number
        return std::numeric_limits<double>::max();
    // compute manhattan distance (norm of power 1)
    return distance2D.lpNorm<1>();
}

vector3 WorldCoordinate::get_signed_distance(const ScreenCoordinate& screenPoint,
                                             const CameraToWorldMatrix& cameraToWorld) const
{
    const WorldCoordinate& projectedScreenPoint = screenPoint.to_world_coordinates(cameraToWorld);
    return this->base() - projectedScreenPoint;
}

double WorldCoordinate::get_distance(const ScreenCoordinate& screenPoint,
                                     const CameraToWorldMatrix& cameraToWorld) const
{
    return get_signed_distance(screenPoint, cameraToWorld).lpNorm<1>();
}

/**
 *      CAMERA COORDINATES
 */

CameraCoordinate WorldCoordinate::to_camera_coordinates(const WorldToCameraMatrix& worldToCamera) const
{
    // WorldCoordinate
    vector4 homogenousWorldCoordinates;
    homogenousWorldCoordinates << this->base(), 1.0;

    const vector4& cameraHomogenousCoordinates = worldToCamera * homogenousWorldCoordinates;
    return CameraCoordinate(cameraHomogenousCoordinates);
}

/**
 *      PLANE COORDINATES
 */

PlaneWorldCoordinates PlaneCameraCoordinates::to_world_coordinates(const PlaneCameraToWorldMatrix& cameraToWorld) const
{
    return PlaneWorldCoordinates(cameraToWorld.base() * this->base());
}
PlaneWorldCoordinates PlaneCameraCoordinates::to_world_coordinates_renormalized(
        const PlaneCameraToWorldMatrix& cameraToWorld) const
{
    const PlaneWorldCoordinates& worldCoordinates = this->to_world_coordinates(cameraToWorld).base();
    return PlaneWorldCoordinates(worldCoordinates.get_normal(), worldCoordinates.get_d());
}

PlaneCameraCoordinates PlaneWorldCoordinates::to_camera_coordinates(const PlaneWorldToCameraMatrix& worldToCamera) const
{
    return PlaneCameraCoordinates(worldToCamera.base() * this->base());
}

PlaneCameraCoordinates PlaneWorldCoordinates::to_camera_coordinates_renormalized(
        const PlaneWorldToCameraMatrix& worldToCamera) const
{
    const PlaneCameraCoordinates& cameraCoordinates = this->to_camera_coordinates(worldToCamera).base();
    return PlaneCameraCoordinates(cameraCoordinates.get_normal(), cameraCoordinates.get_d());
}

vector4 PlaneWorldCoordinates::get_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                                   const PlaneWorldToCameraMatrix& worldToCamera) const
{
    const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

    return vector4(angle_distance(cameraPlane.x(), projectedWorldPlane.x()),
                   angle_distance(cameraPlane.y(), projectedWorldPlane.y()),
                   angle_distance(cameraPlane.z(), projectedWorldPlane.z()),
                   cameraPlane.w() - projectedWorldPlane.w());
}

/**
 * \brief Compute a reduced plane form, allowing for better optimization
 */
vector3 get_plane_transformation(const vector4& plane)
{
    const vector3& normal = plane.head(3);
    const double d = plane(3);
    return vector3(atan(normal.y() / normal.x()), asin(normal.z()), d);
}

vector3 PlaneWorldCoordinates::get_reduced_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                                           const PlaneWorldToCameraMatrix& worldToCamera) const
{
    const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

    const vector3& cameraPlaneSimplified = get_plane_transformation(cameraPlane);
    const vector3& worldPlaneSimplified = get_plane_transformation(projectedWorldPlane);

    return vector3(cameraPlaneSimplified.x() - worldPlaneSimplified.x(),
                   cameraPlaneSimplified.y() - worldPlaneSimplified.y(),
                   cameraPlaneSimplified.z() - worldPlaneSimplified.z());
}

} // namespace rgbd_slam::utils