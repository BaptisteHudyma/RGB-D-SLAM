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
                  * \brief Compute a new observer global pose, to replace the current estimated pose. This function can relaunch optimization as needed to insure that the result is correct
                  *
                  * \param[in] currentPose Last observer optimized pose
                  * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                  *
                  * \return An optimized global pose
                  */
                static const utils::Pose  compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints);

            //private:
                /**
                  * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
*
                  * \param[in] currentPose Last observer optimized pose
                  * \param[in] matchedPoints Object containing the match between observed screen points and reliable map & futur map points 
                  *
                  * \return The estimated world translation & rotation of the camera pose 
                  */
                static const utils::Pose get_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints);
        };

    }   /* pose_optimization */
}   /* rgbd_slam */


#endif
