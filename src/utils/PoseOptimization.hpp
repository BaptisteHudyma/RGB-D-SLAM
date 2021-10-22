#ifndef RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP
#define RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP

#include "Pose.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

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
                static const poseEstimation::Pose  compute_optimized_pose(const poseEstimation::Pose& currentPose, const match_point_container& matchedPoints);

            private:
                /**
                  * \brief Optimize a global pose (orientation/translation) of the observer
*
                  * \param[in] currentPose Last observer optimized pose
                  * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                  *
                  * \return The estimated world translation & rotation of the camera pose 
                  */
                static const poseEstimation::Pose get_optimized_global_pose(const poseEstimation::Pose& currentPose, const match_point_container& matchedPoints);
        };

    }   /* utils */
}   /* rgbd_slam */


#endif
