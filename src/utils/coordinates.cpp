#include "coordinates.hpp"

#include "../parameters.hpp"

namespace rgbd_slam {
namespace utils {

        const double MIN_DEPTH_DISTANCE = 40;   // M millimeters is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // N meters is the depth camera maximum reliable distance

        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }


        worldCoordinates screenCoordinates::to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const
        {
            assert(z() > 0);
            assert(x() >= 0 and y() >= 0);

            const double x = (this->x() - Parameters::get_camera_1_center_x()) * this->z() / Parameters::get_camera_1_focal_x();
            const double y = (this->y() - Parameters::get_camera_1_center_y()) * this->z() / Parameters::get_camera_1_focal_y();

            const cameraCoordinates cameraPoint(x, y, z());

            const vector4 homogenousWorldCoords = cameraToWorld * cameraPoint.get_homogenous();
            return worldCoordinates(homogenousWorldCoords.head<3>());
        }

        bool worldCoordinates::to_screen_coordinates(const worldToCameraMatrix& worldToCamera, screenCoordinates& screenPoint) const
        {
            assert( not std::isnan(x()) and not std::isnan(y()) and not std::isnan(z()) );

            const cameraCoordinates& cameraPoint = this->to_camera_coordinates(worldToCamera);
            assert(cameraPoint.get_homogenous()[3] != 0);

            if (cameraPoint.z() <= 0) {
                return false;
            }

            const double screenX = Parameters::get_camera_1_focal_x() * cameraPoint.x() / cameraPoint.z() + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * cameraPoint.y() / cameraPoint.z() + Parameters::get_camera_1_center_y();

            if (not std::isnan(screenX) and not std::isnan(screenY))
            {
                screenPoint = screenCoordinates(screenX, screenY, cameraPoint.z());
                return true;
            }
            return false;
        }

        cameraCoordinates worldCoordinates::to_camera_coordinates(const worldToCameraMatrix& worldToCamera) const
        {
            //worldCoordinates
            vector4 homogenousWorldCoordinates;
            homogenousWorldCoordinates << this->base(), 1.0;

            const vector4& cameraHomogenousCoordinates = worldToCamera * homogenousWorldCoordinates;
            return cameraCoordinates(cameraHomogenousCoordinates);
        }

}
}