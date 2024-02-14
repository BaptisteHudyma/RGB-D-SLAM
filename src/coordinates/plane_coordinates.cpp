#include "plane_coordinates.hpp"

#include "distance_utils.hpp"
#include "types.hpp"
#include <cmath>
#include <math.h>

namespace rgbd_slam {

/**
 *      PLANE COORDINATES
 */

PlaneWorldCoordinates PlaneCameraCoordinates::to_world_coordinates(
        const PlaneCameraToWorldMatrix& cameraToWorld) const noexcept
{
    return PlaneWorldCoordinates(cameraToWorld.base() * this->get_parametrization());
}

PlaneCameraCoordinates PlaneWorldCoordinates::to_camera_coordinates(
        const PlaneWorldToCameraMatrix& worldToCamera) const noexcept
{
    return PlaneCameraCoordinates(worldToCamera.base() * this->get_parametrization());
}

vector4 PlaneWorldCoordinates::get_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                                   const PlaneWorldToCameraMatrix& worldToCamera) const noexcept
{
    const PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);
    const vector3& cameraNormal = cameraPlane.get_normal();
    const vector3& projectedNormal = projectedWorldPlane.get_normal();

    return vector4(utils::angle_distance(cameraNormal.x(), projectedNormal.x()),
                   utils::angle_distance(cameraNormal.y(), projectedNormal.y()),
                   utils::angle_distance(cameraNormal.z(), projectedNormal.z()),
                   cameraPlane.get_d() - projectedWorldPlane.get_d());
}

/**
 * \brief Compute a reduced plane form, allowing for better optimization
 */
vector3 get_plane_transformation(const PlaneCoordinates& plane) noexcept
{
    const vector3& normal = plane.get_normal();
    const double d = plane.get_d();
    return vector3(atan2(normal.y(), normal.x()), asin(normal.z()), d);
}

vector3 PlaneWorldCoordinates::get_reduced_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                                           const PlaneWorldToCameraMatrix& worldToCamera) const noexcept
{
    const PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

    return cameraPlane.get_d() * cameraPlane.get_normal() -
           projectedWorldPlane.get_d() * projectedWorldPlane.get_normal();
}

matrix34 PlaneWorldCoordinates::get_reduced_signed_distance_jacobian(
        const PlaneWorldToCameraMatrix& worldToCamera) const noexcept
{
    const PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

    const double a = worldToCamera(0, 0);
    const double b = worldToCamera(0, 1);
    const double c = worldToCamera(0, 2);
    const double d = worldToCamera(0, 3);

    const double e = worldToCamera(1, 0);
    const double f = worldToCamera(1, 1);
    const double g = worldToCamera(1, 2);
    const double h = worldToCamera(1, 3);

    const double i = worldToCamera(2, 0);
    const double j = worldToCamera(2, 1);
    const double k = worldToCamera(2, 2);
    const double l = worldToCamera(2, 3);

    const double m = worldToCamera(3, 0);
    const double n = worldToCamera(3, 1);
    const double o = worldToCamera(3, 2);
    const double p = worldToCamera(3, 3);

    const auto& param = projectedWorldPlane.get_parametrization();
    const double theta1 = param(3);
    const double theta2 = param(2);
    const double theta3 = param(1);
    const double theta4 = param(0);

    matrix34 jacobian;
    jacobian.setZero();

    jacobian(0, 0) = -m * theta4 - a * theta1;
    jacobian(0, 1) = -n * theta4 - b * theta1;
    jacobian(0, 2) = -o * theta4 - c * theta1;
    jacobian(0, 3) = -p * theta4 - d * theta1;

    jacobian(1, 0) = -m * theta3 - e * theta1;
    jacobian(1, 1) = -n * theta3 - f * theta1;
    jacobian(1, 2) = -o * theta3 - g * theta1;
    jacobian(1, 3) = -p * theta3 - h * theta1;

    jacobian(2, 0) = -m * theta2 - i * theta1;
    jacobian(2, 1) = -n * theta2 - j * theta1;
    jacobian(2, 2) = -o * theta2 - k * theta1;
    jacobian(2, 3) = -p * theta2 - l * theta1;

    return jacobian;
}

} // namespace rgbd_slam