#include "coordinates.hpp"

#include "../parameters.hpp"
#include "camera_transformation.hpp"

namespace rgbd_slam {
namespace utils {

        const double MIN_DEPTH_DISTANCE = 40;   // (millimeters) is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // (millimeters) is the depth camera maximum reliable distance

        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }


        /**
         *      SCREEN COORDINATES
         */

        CameraCoordinate2D ScreenCoordinate2D::to_camera_coordinates() const
        {
            assert(x() >= 0 and y() >= 0);

            const double x = (this->x() - Parameters::get_camera_1_center_x()) / Parameters::get_camera_1_focal_x();
            const double y = (this->y() - Parameters::get_camera_1_center_y()) / Parameters::get_camera_1_focal_y();

            CameraCoordinate2D cameraPoint(x, y);
            return cameraPoint;
        }


        WorldCoordinate ScreenCoordinate::to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const
        {
            const CameraCoordinate& cameraPoint = this->to_camera_coordinates();
            return cameraPoint.to_world_coordinates(cameraToWorld);
        }

        CameraCoordinate ScreenCoordinate::to_camera_coordinates() const
        {
            assert(x() >= 0 and y() >= 0);
            assert(z() != 0);

            const double x = (this->x() - Parameters::get_camera_1_center_x()) * this->z() / Parameters::get_camera_1_focal_x();
            const double y = (this->y() - Parameters::get_camera_1_center_y()) * this->z() / Parameters::get_camera_1_focal_y();

            CameraCoordinate cameraPoint(x, y, z());
            return cameraPoint;
        }


        /**
         *      CAMERA COORDINATES
         */


        bool CameraCoordinate2D::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const
        {
            const double screenX = Parameters::get_camera_1_focal_x() * x() + Parameters::get_camera_1_center_x();
            const double screenY = Parameters::get_camera_1_focal_y() * y() + Parameters::get_camera_1_center_y();

            if (not std::isnan(screenX) and not std::isnan(screenY))
            {
                screenPoint = ScreenCoordinate2D(screenX, screenY);
                return true;
            }
            return false;
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



        /**
         *      WORLD COORDINATES
         */

        bool WorldCoordinate::to_screen_coordinates(const worldToCameraMatrix& worldToCamera, ScreenCoordinate& screenPoint) const
        {
            assert( not std::isnan(x()) and not std::isnan(y()) and not std::isnan(z()) );

            const CameraCoordinate& cameraPoint = this->to_camera_coordinates(worldToCamera);
            assert(cameraPoint.get_homogenous()[3] > 0);

            return cameraPoint.to_screen_coordinates(screenPoint);
        }

        bool WorldCoordinate::to_screen_coordinates(const worldToCameraMatrix& worldToCamera, ScreenCoordinate2D& screenPoint) const
        {
            ScreenCoordinate screenCoordinates;
            if (to_screen_coordinates(worldToCamera, screenCoordinates))
            {
                screenPoint = ScreenCoordinate2D(screenCoordinates.x(), screenCoordinates.y());
                return true;
            }
            return false;
        }

        vector2 WorldCoordinate::get_signed_distance_2D(const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera) const
        {
            ScreenCoordinate2D projectedScreenPoint; 
            const bool isCoordinatesValid = to_screen_coordinates(worldToCamera, projectedScreenPoint);
            if(isCoordinatesValid)
            {
                vector2 distance = screenPoint - projectedScreenPoint;
                assert (not std::isnan(distance.x()));
                assert (not std::isnan(distance.y()));
                return distance;
            }
            // high number
            return vector2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        }
        
        double WorldCoordinate::get_distance(const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera) const
        {
            const vector2& distance2D = get_signed_distance_2D(screenPoint, worldToCamera);
            if (distance2D.x() >= std::numeric_limits<double>::max() or distance2D.y() >= std::numeric_limits<double>::max())
                // high number
                return std::numeric_limits<double>::max();
            // compute manhattan distance (norm of power 1)
            return distance2D.lpNorm<1>();
        }


        /**
         *      CAMERA COORDINATES
         */

        CameraCoordinate WorldCoordinate::to_camera_coordinates(const worldToCameraMatrix& worldToCamera) const
        {
            //WorldCoordinate
            vector4 homogenousWorldCoordinates;
            homogenousWorldCoordinates << this->base(), 1.0;

            const vector4& cameraHomogenousCoordinates = worldToCamera * homogenousWorldCoordinates;
            return CameraCoordinate(cameraHomogenousCoordinates);
        }


        /**
         *      PLANE COORDINATES
         */


        PlaneWorldCoordinates PlaneCameraCoordinates::to_world_coordinates(const planeCameraToWorldMatrix& cameraToWorld) const
        {
            return PlaneWorldCoordinates(cameraToWorld.base() * this->base());
        }

        PlaneCameraCoordinates PlaneWorldCoordinates::to_camera_coordinates(const planeWorldToCameraMatrix& worldToCamera) const
        {
            return PlaneCameraCoordinates(worldToCamera.base() * this->base());
        }

}
}