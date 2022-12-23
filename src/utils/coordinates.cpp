#include "coordinates.hpp"

#include "../parameters.hpp"
#include "../utils/distance_utils.hpp"
#include "camera_transformation.hpp"
#include <cmath>
#include <math.h>

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

            const static double cameraFX = Parameters::get_camera_1_focal_x();
            const static double cameraFY = Parameters::get_camera_1_focal_y();
            const static double cameraCX = Parameters::get_camera_1_center_x();
            const static double cameraCY = Parameters::get_camera_1_center_y();

            const double x = (this->x() - cameraCX) / cameraFX;
            const double y = (this->y() - cameraCY) / cameraFY;

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
            assert(z() < 0.001 or z() > 0.001);

            const static double cameraFX = Parameters::get_camera_1_focal_x();
            const static double cameraFY = Parameters::get_camera_1_focal_y();
            const static double cameraCX = Parameters::get_camera_1_center_x();
            const static double cameraCY = Parameters::get_camera_1_center_y();

            const double x = (this->x() - cameraCX) * this->z() / cameraFX;
            const double y = (this->y() - cameraCY) * this->z() / cameraFY;

            CameraCoordinate cameraPoint(x, y, z());
            return cameraPoint;
        }


        /**
         *      CAMERA COORDINATES
         */


        bool CameraCoordinate2D::to_screen_coordinates(ScreenCoordinate2D& screenPoint) const
        {
            const static double cameraFX = Parameters::get_camera_1_focal_x();
            const static double cameraFY = Parameters::get_camera_1_focal_y();
            const static double cameraCX = Parameters::get_camera_1_center_x();
            const static double cameraCY = Parameters::get_camera_1_center_y();

            const double screenX = cameraFX * x() + cameraCX;
            const double screenY = cameraFY * y() + cameraCY;

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
            const static double cameraFX = Parameters::get_camera_1_focal_x();
            const static double cameraFY = Parameters::get_camera_1_focal_y();
            const static double cameraCX = Parameters::get_camera_1_center_x();
            const static double cameraCY = Parameters::get_camera_1_center_y();

            const double screenX = cameraFX * x() / z() + cameraCX;
            const double screenY = cameraFY * y() / z() + cameraCY;

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

        vector3 WorldCoordinate::get_signed_distance(const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld) const
        {
            const WorldCoordinate& projectedScreenPoint = screenPoint.to_world_coordinates(cameraToWorld);
            return this->base() - projectedScreenPoint;
        }

        double WorldCoordinate::get_distance(const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld) const
        {
            return get_signed_distance(screenPoint, cameraToWorld).lpNorm<1>();
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

        vector4 PlaneWorldCoordinates::get_signed_distance(const PlaneCameraCoordinates& cameraPlane, const planeWorldToCameraMatrix& worldToCamera) const
        {
            const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

            return vector4(
                angle_distance(cameraPlane.x(), projectedWorldPlane.x()),
                angle_distance(cameraPlane.y(), projectedWorldPlane.y()),
                angle_distance(cameraPlane.z(), projectedWorldPlane.z()),
                cameraPlane.w() - projectedWorldPlane.w()
            );
        }

        /**
         * \brief Compute a reduced plane form, allowing for better optimization
         */
        vector3 get_plane_transformation(const vector4& plane)
        {
            return vector3(
                atan(plane.y() / plane.x()),
                asin(plane.z()),
                plane.w()
            );
        }

        vector3 PlaneWorldCoordinates::get_reduced_signed_distance(const PlaneCameraCoordinates& cameraPlane, const planeWorldToCameraMatrix& worldToCamera) const
        {
            const utils::PlaneCameraCoordinates& projectedWorldPlane = to_camera_coordinates(worldToCamera);

            const vector3& cameraPlaneSimplified = get_plane_transformation(cameraPlane);
            const vector3& worldPlaneSimplified = get_plane_transformation(projectedWorldPlane);

            return vector3(
                angle_distance(cameraPlaneSimplified.x(), worldPlaneSimplified.x()),
                angle_distance(cameraPlaneSimplified.y(), worldPlaneSimplified.y()),
                cameraPlaneSimplified.z() - worldPlaneSimplified.z()
            );
        }


}
}