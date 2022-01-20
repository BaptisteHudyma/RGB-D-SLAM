#ifndef RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP
#define RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP

#include "Pose.hpp"
#include "types.hpp"
#include "matches_containers.hpp"

namespace rgbd_slam {
    namespace pose_optimization {

        class Pose_Optimization
        {
            public:
                /**
                 * \brief Compute a new observer global pose, to replace the current estimated pose 
                 *
                 * \param[in] currentPose Last observer optimized pose
                 * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                 *
                 * \return An optimized global pose
                 */
                static const utils::Pose  compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints);

            private:
                /**
                 * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
                 *
                 * \param[in] currentPose Last observer optimized pose
                 * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                 *
                 * \return The estimated world translation & rotation of the camera pose 
                 */
                static const utils::Pose get_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints);


                /**
                 * \brief Compute an optimized pose, using a RANSAC methodology
                 *
                 * \param[in] currentPose The current pose of the observer
                 * \param[in] matchedPoints Object container the match between observed screen points and local map points 
                 * \param[out] finalPose The optimized pose, valid if the function returned true
                 * \param[out] inlierMatchedPoints The inlier matched point for the finalPose. Valid i the function returned true
                 *
                 * \return True if a valid transformation and inliers were found
                 */
                static bool compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& finalPose, matches_containers::match_point_container& inlierMatchedPoints); 
        };

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
