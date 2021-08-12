#include "utils.hpp"

namespace rgbd_slam {
namespace utils {

    const vector3 screen_to_world_coordinates(const unsigned int screenX, const unsigned int screenY, const double measuredZ, const matrix34& cameraToWorldMatrix) 
    {
        const double x = (static_cast<double>(screenX) - cx) * measuredZ / fx;
        const double y = (static_cast<double>(screenY) - cy) * measuredZ / fy;

        vector4 worldPoint;
        worldPoint << x, y, measuredZ, 1.0;
        return cameraToWorldMatrix * worldPoint;
    }

    const vector2 world_to_screen_coordinates(const vector3& position3D, const matrix34& worldToCameraMatrix)
    {
        vector4 ptH;
        ptH << position3D, 1.0;
        const vector3& point3D = worldToCameraMatrix * ptH; 

        if (point3D[2] == 0) {
            return vector2(0.0, 0.0);
        }

        const double inverseDepth  = 1.0 / point3D[2];
        const double screenX = fx * point3D[0] * inverseDepth + cx;
        const double screenY = fy * point3D[1] * inverseDepth + cy;

        return vector2(screenX, screenY);
    }

    const matrix34 compute_world_to_camera_transform(const poseEstimation::Pose& cameraPose)
    {
        const matrix33& worldToCamRotMtrx = (cameraPose.get_orientation_matrix()).transpose();
        const vector3& worldToCamTranslation = (-worldToCamRotMtrx) * cameraPose.get_position();
        matrix34 worldToCamMtrx;
        worldToCamMtrx << worldToCamRotMtrx, worldToCamTranslation;
        return worldToCamMtrx;
    }

}
}
