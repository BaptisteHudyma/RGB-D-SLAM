#include "camera_transformation.hpp"

namespace rgbd_slam::utils {

CameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position) noexcept
{
    CameraToWorldMatrix cameraToWorld;
    cameraToWorld << rotation.toRotationMatrix(), position, 0, 0, 0, 1;
    return cameraToWorld;
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

CameraToWorldMatrix compute_camera_to_world_transform(const WorldToCameraMatrix& worldToCamera) noexcept
{
    CameraToWorldMatrix cameraToWorld;
    cameraToWorld << worldToCamera.inverse();
    return cameraToWorld;
}

PlaneCameraToWorldMatrix compute_plane_camera_to_world_matrix(const CameraToWorldMatrix& cameraToWorld) noexcept
{
    const matrix33& rotationMatrix = cameraToWorld.block<3, 3>(0, 0);
    const vector3& position = cameraToWorld.col(3).head<3>();

    PlaneCameraToWorldMatrix planeCameraToWorld;
    planeCameraToWorld << rotationMatrix, vector3::Zero(), -position.transpose() * rotationMatrix, 1;
    return planeCameraToWorld;
}

PlaneWorldToCameraMatrix compute_plane_world_to_camera_matrix(const WorldToCameraMatrix& worldToCamera) noexcept
{
    const PlaneCameraToWorldMatrix& PlaneCameraToWorldMatrix =
            compute_plane_camera_to_world_matrix(compute_camera_to_world_transform(worldToCamera));
    PlaneWorldToCameraMatrix cameraToWorld;
    cameraToWorld << PlaneCameraToWorldMatrix.inverse();
    return cameraToWorld;
}

} // namespace rgbd_slam::utils
