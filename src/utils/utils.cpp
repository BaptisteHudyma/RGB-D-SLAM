#include "utils.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {

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

            if (point3D.z() == 0) {
                return vector2(0.0, 0.0);
            }

            const double inverseDepth  = 1.0 / point3D.z();
            const double screenX = Parameters::get_camera_focal_x() * point3D.x() * inverseDepth + Parameters::get_camera_center_x();
            const double screenY = Parameters::get_camera_focal_y() * point3D.y() * inverseDepth + Parameters::get_camera_center_y();

            return vector2(screenX, screenY);
        }

        const matrix34 compute_camera_to_world_transform(const quaternion& rotation, const vector3& position)
        {
            matrix34 cameraToWorldMatrix;
            cameraToWorldMatrix << rotation.toRotationMatrix(), position;
            return cameraToWorldMatrix;
        }

        const matrix34 compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
        {
            const matrix33& worldToCamRotMtrx = (rotation.toRotationMatrix()).transpose();
            const vector3& worldToCamTranslation = (-worldToCamRotMtrx) * position;
            matrix34 worldToCamMtrx;
            worldToCamMtrx << worldToCamRotMtrx, worldToCamTranslation;
            return worldToCamMtrx;
        }

        const matrix33 get_world_point_covariance(const vector2& screenPoint, const double depth, const matrix33& screenPointError)
        {
            const double cameraFX = Parameters::get_camera_focal_x();
            const double cameraFY = Parameters::get_camera_focal_y();
            const double cameraCX = Parameters::get_camera_center_x();
            const double cameraCY = Parameters::get_camera_center_y();

            const matrix33 jacobian {
                {depth / cameraFX, 0.0, (screenPoint.x() - cameraCX) / cameraFX },
                {0.0, depth / cameraFY, (screenPoint.y() - cameraCY) / cameraFY },
                {0.0, 0.0, 1.0}
            };

            return jacobian * screenPointError * jacobian.transpose();
        }

    }
}
