#ifndef RGBDSLAM_UTILS_COVARIANCES_HPP
#define RGBDSLAM_UTILS_COVARIANCES_HPP

#include "coordinates.hpp"

#include "pose.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
         * \brief compute a covariance matrix for a screen point associated with a depth measurement
         *
         * \param[in] ScreenCoordinate The coordinates of this point, in screen space
         *
         * \return A 3x3 covariance matrix. It should be diagonal
         */
        const screenCoordinateCovariance get_screen_point_covariance(const ScreenCoordinate& ScreenCoordinate);

        /**
         * \brief Compute a screen point covariance from a given point
         *
         * \param[in] point The coordinates of this 3D point (world or camera space)
         * \param[in] pointCovariance The covariance associated with this point (world or camera space)
         */
        const screenCoordinateCovariance get_screen_point_covariance(const vector3& point, const matrix33& pointCovariance);

        /**
         * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point. This function will internaly compute the covariance of the screen point.
         *
         * \param[in] screenPoint The 2D point in screen coordinates
         *
         * \return the covariance of the 3D camera point
         */
        const cameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint);

        /**
         * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point
         *
         * \param[in] screenPoint The 2D point in screen coordinates
         * \param[in] screenPointCovariance The covariance matrix associated with a point in screen space
         *
         * \return the covariance of the 3D camera point
         */
        const cameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint, const screenCoordinateCovariance& screenPointCovariance);

        /**
         * \brief Compute the variance of the final pose in X Y Z
         *
         * \param[in] pose The pose to compute the variance of
         * \param[in] matchedPoints A container of matched features (inliers)
         * \param[out] poseVariance If the function returns true, then this is the estimated position variance estimated from matched points
         *
         * \return True if the variance was estimated 
         */
        bool compute_pose_variance(const Pose& pose, const matches_containers::match_point_container& matchedPoints, vector3& poseVariance);

        /**
         * \brief Compute a pose covariance matrix from a pose
         * \param[in] pose The pose to analyze
         */
        const matrix33 compute_pose_covariance(const Pose& pose);

    }   // utils
}       // rgbd_slam


#endif
