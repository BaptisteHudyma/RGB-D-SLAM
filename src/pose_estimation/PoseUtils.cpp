#include "PoseUtils.hpp"

namespace poseEstimation {


    Pose Pose_Utils::compute_right_camera_pose(const Pose& leftCamPose, double baseLine) {
        vector3 pt(baseLine, 0, 0);
        vector3 pos = (leftCamPose.get_orientation_matrix() * pt) + leftCamPose.get_position();
        return Pose(pos, leftCamPose.get_orientation_quaternion());
    }

    matrix34 Pose_Utils::compute_world_to_camera_transform(const Pose& cameraPose) {
        matrix33 world2camRotationMatrix = (cameraPose.get_orientation_matrix()).transpose();
        vector3 world2camTranslation = (-world2camRotationMatrix) * cameraPose.get_position();

        matrix34 world2camMatrix;
        world2camMatrix << world2camRotationMatrix, world2camTranslation;
        return world2camMatrix;
    }

    matrix34 Pose_Utils::compute_projection_matrix(const Pose& cameraPose, const matrix33& intrisics) {
        matrix34 world2camMatrix = Pose_Utils::compute_world_to_camera_transform(cameraPose);
        matrix34 projMatrix = intrisics * world2camMatrix;
        return projMatrix;
    }


    vector2 Pose_Utils::project_point(const vector3& pt, const matrix34& projectionMatrix) {
        vector3 projPt = projectionMatrix * vector4(pt.x(), pt.y(), pt.z(), 1.0);
        const double invZ = 1.0 / projPt.z();
        return vector2(projPt.x() * invZ, projPt.y() * invZ);
    }

}
