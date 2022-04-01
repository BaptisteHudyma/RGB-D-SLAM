#ifndef RGBDSLAM_UTILS_COVARIANCES_HPP
#define RGBDSLAM_UTILS_COVARIANCES_HPP

#include "types.hpp"
#include "pose.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
         * \brief compute a covariance matrix for a screen point associated with a depth measurement
         *
         * \param[in] screenCoordinates The coordinates of this 2D point, in screen space
         * \param[in] depth The depth associated to this 2D point
         *
         * \return A 3x3 covariance matrix. It should be diagonal
         */
        const matrix33 get_screen_point_covariance(const vector2& screenCoordinates, const double depth);

        /**
         * \brief Compute the associated Gaussian error of a screen point when it will be transformed to world point
         *
         * \param[in] screenPoint The 2D point in screen coordinates
         * \param[in] depth The depth associated with this screen point
         * \param[in] screenPointCovariance The covariance matrix associated with a point in screen space
         *
         * \return the covariance of the 3D world point
         */
        const matrix33 get_world_point_covariance(const vector2& screenPoint, const double depth, const matrix33& screenPointCovariance);

        /**
         * \brief Compute the variance of the final pose in X Y Z
         *
         * \param[in] pose The pose to compute the variance of
         * \param[in] matchedPoints A container of matched features (inliers)
         * \param[out] poseVariance If the function returns true, then this is the estimated position variance estimated from matched points
         *
         * \return True if the variance was estimated 
         */
        bool compute_pose_variance(const utils::Pose& pose, const matches_containers::match_point_container& matchedPoints, vector3& poseVariance);

        /**
         * \brief Compute a pose covariance matrix from a pose
         * \param[in] pose The pose to analyze
         */
        const matrix33 compute_pose_covariance(const utils::Pose& pose);

    }   // utils
}       // rgbd_slam


#endif
