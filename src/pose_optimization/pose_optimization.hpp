#ifndef RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP
#define RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP

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
                 * \param[in] matchedFeatures Object containing the match between observed screen features and reliable map features
                 * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned true
                 * \param[out] featureSets The inliers/outliers matched features for the finalPose. Valid if the function returned true
                 *
                 * \return True if a valid pose was computed 
                 */
                static bool compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::matchContainer& matchedFeatures, utils::Pose& optimizedPose, matches_containers::match_sets& featureSets); 

            private:
                /**
                 * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
                 *
                 * \param[in] currentPose Last observer optimized pose
                 * \param[in] matchedFeatures Object containing the match between observed screen features and reliable map features 
                 * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned true
                 *
                 * \return True if a valid pose was computed 
                 */
                static bool compute_optimized_global_pose(const utils::PoseBase& currentPose, const matches_containers::match_sets& matchedFeatures, utils::PoseBase& optimizedPose);


                /**
                 * \brief Compute an optimized pose, using a RANSAC methodology
                 *
                 * \param[in] currentPose The current pose of the observer
                 * \param[in] matchedFeatures Object container the match between observed screen features and local map features 
                 * \param[out] finalPose The optimized pose, valid if the function returned true
                 * \param[out] featureSets The matched features detected as inlier and outliers. Valid if the function returned true
                 *
                 * \return True if a valid pose and inliers were found
                 */
                static bool compute_pose_with_ransac(const utils::PoseBase& currentPose, const matches_containers::matchContainer& matchedFeatures, utils::PoseBase& finalPose, matches_containers::match_sets& featureSets); 

                static bool compute_p3p_pose(const utils::PoseBase& currentPose, const matches_containers::match_point_container& matchedPoints, utils::PoseBase& optimizedPose);
        };

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
