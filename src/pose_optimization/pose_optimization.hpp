#ifndef RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP
#define RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP

#include "../types.hpp"
#include "../utils/pose.hpp"
#include "../utils/matches_containers.hpp"

namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Find the transformation between a matches feature sets, using a custom Levenberg Marquardt method
         */
        class Pose_Optimization
        {
            public:
                /**
                 * \brief Compute a new observer global pose, to replace the current estimated pose 
                 *
                 * \param[in] currentPose Last observer optimized pose
                 * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                 * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned true
                 * \param[out] outlierMatchedPoints The outliers matched point for the finalPose. Valid if the function returned true
                 *
                 * \return True if a valid pose was computed 
                 */
                static bool compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose, matches_containers::match_point_container& outlierMatchedPoints); 

            private:
                /**
                 * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
                 *
                 * \param[in] currentPose Last observer optimized pose
                 * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                 * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned true
                 *
                 * \return True if a valid pose was computed 
                 */
                static bool compute_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose);


                /**
                 * \brief Compute an optimized pose, using a RANSAC methodology
                 *
                 * \param[in] currentPose The current pose of the observer
                 * \param[in] matchedPoints Object container the match between observed screen points and local map points 
                 * \param[out] finalPose The optimized pose, valid if the function returned true
                 * \param[out] outlierMatchedPoints The outliers matched point for the finalPose. Valid if the function returned true
                 *
                 * \return True if a valid pose and inliers were found
                 */
                static bool compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& finalPose, matches_containers::match_point_container& outlierMatchedPoints); 

                static bool compute_p3p_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose);
        };

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
