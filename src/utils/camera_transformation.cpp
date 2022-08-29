#include "camera_transformation.hpp"

#include "../parameters.hpp"

namespace rgbd_slam {
    namespace utils {


        const double MIN_DEPTH_DISTANCE = 40;   // M millimeters is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // N meters is the depth camera maximum reliable distance

        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }

        
        const vector3 screen_to_world_coordinates(const double screenX, const double screenY, const double measuredZ, const matrix44& cameraToWorldMatrix) 
        {
            assert(measuredZ > 0);
            assert(screenX >= 0 and screenY >= 0);

            const double x = (screenX - Parameters::get_camera_1_center_x()) * measuredZ / Parameters::get_camera_1_focal_x();
            const double y = (screenY - Parameters::get_camera_1_center_y()) * measuredZ / Parameters::get_camera_1_focal_y();

            vector4 worldPoint;
            worldPoint << x, y, measuredZ, 1.0;
            return screen_to_world_coordinates(worldPoint, cameraToWorldMatrix).head<3>();
        }

        const vector4 screen_to_world_coordinates(const vector4& vector4d, const matrix44& cameraToWorldMatrix)
        {
            return cameraToWorldMatrix * vector4d;
        }

        bool world_to_screen_coordinates(const vector3& position3D, const matrix44& worldToScreenMatrix, vector2& screenCoordinates)
        {
            assert( not std::isnan(position3D.x()) and not std::isnan(position3D.y()) and not std::isnan(position3D.z()) );

            vector4 ptH;
            ptH << position3D, 1.0;
            const vector4& point4d = world_to_screen_coordinates(ptH, worldToScreenMatrix);
            assert(point4d[3] != 0);
            const vector3& point3D = point4d.head<3>() / point4d[3]; 

            if (point3D.z() <= 0) {
                return false;
            }

            const double inverseDepth  = 1.0 / point3D.z();
            const double screenX = Parameters::get_camera_1_focal_x() * point3D.x() * inverseDepth + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * point3D.y() * inverseDepth + Parameters::get_camera_1_center_y();

            if (not std::isnan(screenX) and not std::isnan(screenY))
            {
                screenCoordinates = vector2(screenX, screenY);
                return true;
            }
            return false;
        }

        const vector4 world_to_screen_coordinates(const vector4& worldVector4, const matrix44& worldToScreenMatrix)
        {
            return worldToScreenMatrix * worldVector4;
        }

        const matrix44 compute_camera_to_world_transform(const quaternion& rotation, const vector3& position)
        {
            matrix44 screenToWorldMatrix;
            screenToWorldMatrix << rotation.toRotationMatrix(), position,  0, 0, 0, 1;
            return screenToWorldMatrix;
        }

        const matrix44 compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
        {
            return compute_world_to_camera_transform(compute_camera_to_world_transform(rotation, position));
        }

        const matrix44 compute_world_to_camera_transform(const matrix44& cameraToWorldMatrix)
        {
            return cameraToWorldMatrix.inverse();
        }

    }   // utils
}       // rgbd_slam
