#include "camera_transformation.hpp"
#include "angle_utils.hpp"
#include "logger.hpp"

namespace rgbd_slam::utils {

/**
 * \brief This is used to go from a camera based coordinate system (x right, z forward, y down) to the world
 * coordinate system (x forward, y left, z up) and inverse
 */
static const matrix44 CameraToWorld = get_transformation_matrix(
        get_quaternion_from_euler_angles(EulerAngles(0.0, 90.0 * EulerToRadian, -90.0 * EulerToRadian)),
        vector3::Zero());

CameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position) noexcept
{
    return CameraToWorldMatrix {CameraToWorld * get_transformation_matrix(rotation, position)};
}

CameraToWorldMatrix compute_camera_to_world_transform(const WorldToCameraMatrix& worldToCamera) noexcept
{
    CameraToWorldMatrix cameraToWorld;
    // already contains the camera to world transform, no need to use CameraToWorld again
    cameraToWorld << worldToCamera.inverse();
    return cameraToWorld;
}

CameraToWorldMatrix compute_camera_to_world_transform_no_correction(const quaternion& rotation,
                                                                    const vector3& position) noexcept
{
    outputs::log_error("This function should not be used in any other context than testing !");
    return CameraToWorldMatrix {get_transformation_matrix(rotation, position)};
}

WorldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position) noexcept
{
    return compute_world_to_camera_transform(compute_camera_to_world_transform(rotation, position));
}

WorldToCameraMatrix compute_world_to_camera_transform(const CameraToWorldMatrix& cameraToWorld) noexcept
{
    WorldToCameraMatrix worldToCamera;
    worldToCamera << cameraToWorld.inverse();
    return worldToCamera;
}

WorldToCameraMatrix compute_world_to_camera_transform_no_correction(const quaternion& rotation,
                                                                    const vector3& position) noexcept
{
    outputs::log_error("This function should not be used in any other context than testing !");
    return compute_world_to_camera_transform(compute_camera_to_world_transform_no_correction(rotation, position));
}

PlaneCameraToWorldMatrix compute_plane_camera_to_world_matrix(const CameraToWorldMatrix& cameraToWorld) noexcept
{
    const matrix33& rotationMatrix = cameraToWorld.rotation();
    const vector3& position = cameraToWorld.translation();

    PlaneCameraToWorldMatrix planeCameraToWorld;
    planeCameraToWorld << rotationMatrix, vector3::Zero(), -position.transpose() * rotationMatrix, 1;
    return planeCameraToWorld;
}

PlaneWorldToCameraMatrix compute_plane_world_to_camera_matrix(const WorldToCameraMatrix& worldToCamera) noexcept
{
    // TODO: seems inefficient: double matrix inversion
    const PlaneCameraToWorldMatrix& PlaneCameraToWorldMatrix =
            compute_plane_camera_to_world_matrix(compute_camera_to_world_transform(worldToCamera));
    PlaneWorldToCameraMatrix cameraToWorld;
    cameraToWorld << PlaneCameraToWorldMatrix.inverse();
    return cameraToWorld;
}

} // namespace rgbd_slam::utils
