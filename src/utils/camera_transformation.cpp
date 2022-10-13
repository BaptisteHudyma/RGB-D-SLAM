#include "camera_transformation.hpp"

#include "../parameters.hpp"
#include "types.hpp"
#include <memory>

namespace rgbd_slam {
    namespace utils {


        const double MIN_DEPTH_DISTANCE = 40;   // M millimeters is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // N meters is the depth camera maximum reliable distance

        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }

        
        const worldCoordinates screen_to_world_coordinates(const double screenX, const double screenY, const double measuredZ, const cameraToWorldMatrix& cameraToWorld) 
        {
            assert(measuredZ > 0);
            assert(screenX >= 0 and screenY >= 0);

            const double x = (screenX - Parameters::get_camera_1_center_x()) * measuredZ / Parameters::get_camera_1_focal_x();
            const double y = (screenY - Parameters::get_camera_1_center_y()) * measuredZ / Parameters::get_camera_1_focal_y();

            vector4 worldPoint;
            worldPoint << x, y, measuredZ, 1.0;

            return worldCoordinates(screen_to_world_coordinates(worldPoint, cameraToWorld).head<3>());
        }

        const vector4 screen_to_world_coordinates(const vector4& vector4d, const cameraToWorldMatrix& cameraToWorld)
        {
            return cameraToWorld * vector4d;
        }

        bool compute_world_to_screen_coordinates(const worldCoordinates& position3D, const worldToCameraMatrix& worldToCamera, screenCoordinates& screenCoordinates)
        {
            assert( not std::isnan(position3D.x()) and not std::isnan(position3D.y()) and not std::isnan(position3D.z()) );

            const cameraCoordinates& cameraPoint = world_to_camera_coordinates(position3D, worldToCamera);
            assert(cameraPoint.get_homogenous()[3] != 0);

            if (cameraPoint.z() <= 0) {
                return false;
            }

            const double screenX = Parameters::get_camera_1_focal_x() * cameraPoint.x() / cameraPoint.z() + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * cameraPoint.y() / cameraPoint.z() + Parameters::get_camera_1_center_y();

            if (not std::isnan(screenX) and not std::isnan(screenY))
            {
                screenCoordinates << screenX, screenY;
                return true;
            }
            return false;
        }

        const cameraCoordinates world_to_camera_coordinates(const worldCoordinates& worldCoordinates, const worldToCameraMatrix& worldToCamera)
        {
            //worldCoordinates
            vector4 homogenousWorldCoordinates;
            homogenousWorldCoordinates << worldCoordinates, 1.0;

            const vector4& cameraHomogenousCoordinates = worldToCamera * homogenousWorldCoordinates;
            return cameraCoordinates(cameraHomogenousCoordinates);
        }

        const cameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position)
        {
            cameraToWorldMatrix cameraToWorld;
            cameraToWorld << rotation.toRotationMatrix(), position,  0, 0, 0, 1;
            return cameraToWorld;
        }

        const worldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position)
        {
            return compute_world_to_camera_transform(compute_camera_to_world_transform(rotation, position));
        }

        const worldToCameraMatrix compute_world_to_camera_transform(const cameraToWorldMatrix& cameraToWorld)
        {
            worldToCameraMatrix worldToCamera;
            worldToCamera << cameraToWorld.inverse();
            return worldToCamera;
        }

    }   // utils
}       // rgbd_slam
