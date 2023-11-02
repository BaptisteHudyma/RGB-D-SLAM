#include "plane_coordinates.hpp"

#include "../parameters.hpp"
#include "../utils/distance_utils.hpp"
#include "camera_transformation.hpp"
#include "covariances.hpp"
#include "types.hpp"
#include <cmath>
#include <math.h>
#include <stdexcept>

namespace rgbd_slam::utils {

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
    const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);
    const vector3& cameraNormal = cameraPlane.get_normal();
    const vector3& projectedNormal = projectedWorldPlane.get_normal();

    return vector4(angle_distance(cameraNormal.x(), projectedNormal.x()),
                   angle_distance(cameraNormal.y(), projectedNormal.y()),
                   angle_distance(cameraNormal.z(), projectedNormal.z()),
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
    const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

    return cameraPlane.get_d() * cameraPlane.get_normal() -
           projectedWorldPlane.get_d() * projectedWorldPlane.get_normal();
}

} // namespace rgbd_slam::utils