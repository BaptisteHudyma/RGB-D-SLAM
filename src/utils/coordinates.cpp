#include "coordinates.hpp"

#include "../parameters.hpp"

namespace rgbd_slam {
namespace utils {

        const double MIN_DEPTH_DISTANCE = 40;   // (millimeters) is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // (millimeters) is the depth camera maximum reliable distance

        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }


        WorldCoordinate ScreenCoordinate::to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const
        {
            const CameraCoordinate& cameraPoint = this->to_camera_coordinates();
            return cameraPoint.to_world_coordinates(cameraToWorld);
        }

        CameraCoordinate ScreenCoordinate::to_camera_coordinates() const
        {
            assert(z() > 0);
            assert(x() >= 0 and y() >= 0);

            const double x = (this->x() - Parameters::get_camera_1_center_x()) * this->z() / Parameters::get_camera_1_focal_x();
            const double y = (this->y() - Parameters::get_camera_1_center_y()) * this->z() / Parameters::get_camera_1_focal_y();

            CameraCoordinate cameraPoint(x, y, z());
            return cameraPoint;
        }

        WorldCoordinate CameraCoordinate::to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const
        {
            const vector4 homogenousWorldCoords = cameraToWorld * this->get_homogenous();
            return WorldCoordinate(homogenousWorldCoords.head<3>());
        }

        bool CameraCoordinate::to_screen_coordinates(ScreenCoordinate& screenPoint) const
        {
            const double screenX = Parameters::get_camera_1_focal_x() * x() / z() + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * y() / z() + Parameters::get_camera_1_center_y();

            if (not std::isnan(screenX) and not std::isnan(screenY))
            {
                screenPoint = ScreenCoordinate(screenX, screenY, z());
                return true;
            }
            return false;
        }

        bool WorldCoordinate::to_screen_coordinates(const worldToCameraMatrix& worldToCamera, ScreenCoordinate& screenPoint) const
        {
            assert( not std::isnan(x()) and not std::isnan(y()) and not std::isnan(z()) );

            const CameraCoordinate& cameraPoint = this->to_camera_coordinates(worldToCamera);
            assert(cameraPoint.get_homogenous()[3] != 0);

            if (cameraPoint.z() <= 0) {
                return false;
            }

            return cameraPoint.to_screen_coordinates(screenPoint);
        }

        CameraCoordinate WorldCoordinate::to_camera_coordinates(const worldToCameraMatrix& worldToCamera) const
        {
            //WorldCoordinate
            vector4 homogenousWorldCoordinates;
            homogenousWorldCoordinates << this->base(), 1.0;

            const vector4& cameraHomogenousCoordinates = worldToCamera * homogenousWorldCoordinates;
            return CameraCoordinate(cameraHomogenousCoordinates);
        }

}
}