#ifndef RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP
#define RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP

#include "../types.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
          * \brief Return true is a measurement is in the measurement range
          */
        bool is_depth_valid(const double depth);

        /*
         * \brief Transform a screen point with a depth value to a 3D point
         *
         * \param[in] screenPoint The screen point to convert 
         * \param[in] cameraToWorld Matrix to transform local to world coordinates
         *
         * \return A 3D point in frame coordinates
         */
        const worldCoordinates screen_to_world_coordinates(const screenCoordinates& screenPoint, const cameraToWorldMatrix& cameraToWorld);

        /**
         * \brief Transform a vector in camera space to a vector in world space
         *
         * \param[in] cameraPoint A vector in camera space
         * \param[in] cameraToWorld Matrix to transform local to world coordinates
         *
         * \return A vector in world space
         */
        const vector4 camera_to_world_coordinates(const cameraCoordinates& cameraPoint, const cameraToWorldMatrix& cameraToWorld);

        /**
         * \brief Transform a point from world to screen coordinate system
         *
         * \param[in] position3D Coordinates of the detected point (world coordinates)
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         * \param[out] screenPoint The point screen coordinates, if the function returned true
         *
         * \return True if the screen position is valid
         */
        bool compute_world_to_screen_coordinates(const worldCoordinates& position3D, const worldToCameraMatrix& worldToCamera, screenCoordinates& screenPoint);

        /**
         * \brief Transform a vector in world space to a vector in camera space
         *
         * \param[in] worldCoordinates A vector in world space
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         *
         * \return The input vector transformed to camera space
         */
        const cameraCoordinates world_to_camera_coordinates(const worldCoordinates& worldCoordinates, const worldToCameraMatrix& worldToCamera);

        /**
         * \brief Given a camera pose, returns a transformation matrix to convert a world point (xyz) to camera point (uvd)
         */
        const worldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position);

        const worldToCameraMatrix compute_world_to_camera_transform(const cameraToWorldMatrix& cameraToWorld);

        /**
         * \brief Given a camera pose, returns a transformation matrix to convert a camera point (uvd) to world point (xyz)
         */
        const cameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position);

    }   // utils
}       // rgbd_slam


#endif
