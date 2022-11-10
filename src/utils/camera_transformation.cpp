#include "camera_transformation.hpp"

namespace rgbd_slam {
    namespace utils {

        const cameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position)
        {
            cameraToWorldMatrix cameraToWorld;
            cameraToWorld << rotation.toRotationMatrix(), position,  0, 0, 0, 1;
            return cameraToWorld;
        }

        const worldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
        {
            return compute_world_to_camera_transform(compute_camera_to_world_transform(rotation, position));
        }

        const worldToCameraMatrix compute_world_to_camera_transform(const cameraToWorldMatrix& cameraToWorld)
        {
            worldToCameraMatrix worldToCamera;
            worldToCamera << cameraToWorld.inverse();
            return worldToCamera;
        }

        const cameraToWorldMatrix compute_camera_to_world_transform(const worldToCameraMatrix& worldToCamera)
        {
            cameraToWorldMatrix cameraToWorld;
            cameraToWorld << worldToCamera.inverse();
            return cameraToWorld;
        }

    }   // utils
}       // rgbd_slam
