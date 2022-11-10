#include "camera_transformation.hpp"
#include <iostream>

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

        planeCameraToWorldMatrix compute_plane_camera_to_world_matrix(const cameraToWorldMatrix& cameraToWorld)
        {
            const matrix33& rotationMatrix = cameraToWorld.block(0,0,3,3);
            const vector3& position = cameraToWorld.col(3).head(3);
            
            planeCameraToWorldMatrix planeCameraToWorld;
            planeCameraToWorld << 
                rotationMatrix, vector3::Zero(),
                -position.transpose()*rotationMatrix, 1;
            return planeCameraToWorld;
        }

        planeWorldToCameraMatrix compute_plane_world_to_camera_matrix(const worldToCameraMatrix& worldToCamera)
        {
            const planeCameraToWorldMatrix& planeCameraToWorldMatrix = compute_plane_camera_to_world_matrix(compute_camera_to_world_transform(worldToCamera));
            planeWorldToCameraMatrix cameraToWorld;
            cameraToWorld << planeCameraToWorldMatrix.inverse();
            return cameraToWorld;
        }

    }   // utils
}       // rgbd_slam
