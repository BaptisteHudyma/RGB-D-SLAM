#include "distance_utils.hpp"

#include "../types.hpp"
#include "coordinates.hpp"

namespace rgbd_slam {
    namespace utils {

        vectorxd get_signed_distance(const vectorxd& pointA, const vectorxd& pointB)
        {
            return (pointA - pointB);
        }

        vector3 get_3D_to_3D_distance_3D(const WorldCoordinate& worldPoint, const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld)
        {
            const WorldCoordinate& projectedScreenPoint = screenPoint.to_world_coordinates(cameraToWorld);

            return get_signed_distance(worldPoint, projectedScreenPoint);
        }

        double get_3D_to_3D_distance(const WorldCoordinate& worldPoint, const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld)
        {
            return get_3D_to_3D_distance_3D(worldPoint, screenPoint, cameraToWorld).lpNorm<1>();
        }

        double angle_distance(const double angleA, const double angleB)
        {
            return atan2(sin(angleA - angleB), cos(angleA - angleB));
        }

        vector4 get_3D_to_2D_plane_distance(const PlaneWorldCoordinates& worldPlane, const PlaneCameraCoordinates& cameraPlane, const planeWorldToCameraMatrix& worldToCamera)
        {
            const utils::PlaneCameraCoordinates& projectedWorldPlane = worldPlane.to_camera_coordinates(worldToCamera);

            return vector4(
                angle_distance(cameraPlane.x(), projectedWorldPlane.x()),
                angle_distance(cameraPlane.y(), projectedWorldPlane.y()),
                angle_distance(cameraPlane.z(), projectedWorldPlane.z()),
                cameraPlane.w() - projectedWorldPlane.w()
            );
        }

    }   // utils
}       // rgbd_slam
