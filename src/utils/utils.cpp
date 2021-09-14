#include "utils.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
namespace utils {

    const quaternion get_underparametrized_quaternion(const double x, const double y, const double z)
    {
        const double w = sqrt(1.0 - pow(x, 2.0) - pow(y, 2.0) - pow(z, 2.0));
        return quaternion(w, x, y, z);
    }

    const vector3 screen_to_world_coordinates(const unsigned int screenX, const unsigned int screenY, const double measuredZ, const matrix34& cameraToWorldMatrix) 
    {
        const double x = (static_cast<double>(screenX) - Parameters::get_camera_center_x()) * measuredZ / Parameters::get_camera_focal_x();
        const double y = (static_cast<double>(screenY) - Parameters::get_camera_center_y()) * measuredZ / Parameters::get_camera_focal_y();

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
        const double screenX = Parameters::get_camera_focal_x() * point3D[0] * inverseDepth + Parameters::get_camera_center_x();
        const double screenY = Parameters::get_camera_focal_y() * point3D[1] * inverseDepth + Parameters::get_camera_center_y();

        return vector2(screenX, screenY);
    }

    const matrix34 compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
    {
        const matrix33& worldToCamRotMtrx = (rotation.toRotationMatrix()).transpose();
        const vector3& worldToCamTranslation = (-worldToCamRotMtrx) * position;
        matrix34 worldToCamMtrx;
        worldToCamMtrx << worldToCamRotMtrx, worldToCamTranslation;
        return worldToCamMtrx;
    }

}
}
