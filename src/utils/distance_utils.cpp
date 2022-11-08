#include "distance_utils.hpp"

#include "../types.hpp"
#include "coordinates.hpp"

namespace rgbd_slam {
    namespace utils {

        vectorxd get_signed_distance(const vectorxd& pointA, const vectorxd& pointB)
        {
            return (pointA - pointB);
        }

        vector2 get_3D_to_2D_distance_2D(const WorldCoordinate& worldPoint, const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera)
        {
            ScreenCoordinate2D projectedScreenPoint; 
            const bool isCoordinatesValid = worldPoint.to_screen_coordinates(worldToCamera, projectedScreenPoint);
            if(isCoordinatesValid)
            {
                const vector2& distance = get_signed_distance(screenPoint, projectedScreenPoint);
                assert (not std::isnan(distance.x()));
                assert (not std::isnan(distance.y()));
                return distance;
            }
            // high number
            return vector2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        }

        double get_3D_to_2D_distance(const WorldCoordinate& worldPoint, const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera)
        {
            const vector2& distance2D = get_3D_to_2D_distance_2D(worldPoint, screenPoint, worldToCamera);
            if (distance2D.x() >= std::numeric_limits<double>::max() or distance2D.y() >= std::numeric_limits<double>::max())
                // high number
                return std::numeric_limits<double>::max();
            // compute manhattan distance (norm of power 1)
            return distance2D.lpNorm<1>();
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

        /**
         * \brief Transform an overconstraint plane to a minimal parametrization
         * \param[in] plane A plane representation in Hessian form 
         */
        vector3 get_transformed_plane(const utils::PlaneCameraCoordinates& plane)
        {
            const vector3& normalizedNormal = plane.head(3);
            return vector3(
                atan(normalizedNormal.y() / normalizedNormal.x()),
                asin(normalizedNormal.z()),
                plane.w()
            );
        }

        vector3 get_3D_to_2D_plane_distance(const PlaneWorldCoordinates& worldPlane, const PlaneCameraCoordinates& cameraPlane, const worldToCameraMatrix& worldToCamera)
        {
            const utils::PlaneCameraCoordinates& projectedWorldPlane = worldPlane.to_camera_coordinates(worldToCamera);
            const vector3& planeProjectionError = get_transformed_plane(cameraPlane) - get_transformed_plane(projectedWorldPlane);
            return planeProjectionError;
        }

    }   // utils
}       // rgbd_slam
