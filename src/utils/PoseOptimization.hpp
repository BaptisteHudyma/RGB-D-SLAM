#ifndef RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP
#define RGBDSLAM_UTILS_POSE_OPTIMIZATION_HPP

#include "Pose.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        class Pose_Optimization
        {
            public:
                static void compute_optimized_pose(poseEstimation::Pose& currentPose, match_point_container& matchedPoints);
        };

    }   /* utils */
}   /* rgbd_slam */


#endif
