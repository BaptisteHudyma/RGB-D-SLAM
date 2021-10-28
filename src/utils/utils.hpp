#ifndef UTILS_FUNCTIONS_HPP
#define UTILS_FUNCTIONS_HPP

#include "types.hpp"
#include "Pose.hpp"

namespace rgbd_slam {
    namespace utils {

        /*
         * \brief Transform a screen point with a depth value to a 3D point
         *
         * \param[in] screenX X coordinates of the 2D point (double because we can have sub pixel accuracy)
         * \param[in] screenY Y coordinates of the 2D point (double because we can have sub pixel accuracy)
         * \param[in] measuredZ Measured z depth of the point, in meters
         * \param[in] cameraToWorldMatrix Matrix to transform local to world coordinates
         *
         * \return A 3D point in frame coordinates
         */
        const vector3 screen_to_world_coordinates(const double screenX, const double screenY, const double measuredZ, const matrix34& cameraToWorldMatrix);


        /**
         * \brief Transform a point from world to screen coordinate system
         *
         * \param[in] position3D Coordinates of the detected point (world coordinates)
         * \param[in] worldToCameraMatrix Matrix to transform the world to a local coordinate system
         *
         * \return The position of the point in screen coordinates
         */
        const vector2 world_to_screen_coordinates(const vector3& position3D, const matrix34& worldToCameraMatrix);


        /**
         * \brief Given a camera pose, returns a transformation matrix to convert a world point (xyz) to camera point (uvd)
         */
        const matrix34 compute_world_to_camera_transform(const quaternion& rotation, const vector3& position);


        /**
         * \brief Given a camera pose, returns a transformation matrix to convert a camera point (uvd) to world point (xyz)
         */
        const matrix34 compute_camera_to_world_transform(const quaternion& rotation, const vector3& position);


        /**
         * \brief Compute the associated Gaussian error of a screen point when it will be transformed to world point
         */
        const matrix33 get_world_point_covariance(const vector2& screenPoint, const double depth, const matrix33& screenPointError);

    }
}




#endif
