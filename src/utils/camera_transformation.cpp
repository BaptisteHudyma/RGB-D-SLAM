#include "camera_transformation.hpp"

namespace rgbd_slam::utils {

matrix44 get_transformation_matrix(const quaternion& rotation, const vector3& position) noexcept
{
    matrix44 transfoMatrix;
    transfoMatrix << rotation.toRotationMatrix(), position, 0, 0, 0, 1;
    return transfoMatrix;
}

CameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position) noexcept
{
    return CameraToWorldMatrix(get_transformation_matrix(rotation, position));
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
    const matrix33& rotationMatrix = cameraToWorld.rotation();
    const vector3& position = cameraToWorld.translation();

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
