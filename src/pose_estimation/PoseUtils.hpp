#ifndef POSE_UTILS_HPP
#define POSE_UTILS_HPP

#include "Pose.hpp"

namespace poseUtils {

    class Pose_Utils {
        public:
            static poseEstimation::Pose compute_right_camera_pose(const poseEstimation::Pose &leftCamPose, double baseLine);

            static poseEstimation::matrix34 compute_world_to_camera_transform(const poseEstimation::Pose &cameraPose);

            static poseEstimation::matrix34 compute_projection_matrix(const poseEstimation::Pose &cameraPose, const poseEstimation::matrix33 &intrisics);

            inline static poseEstimation::vector2 project_point(const poseEstimation::vector3 &pt, const poseEstimation::matrix34 projectionMatrix);
    };
}


#endif
