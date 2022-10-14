#include "distance_utils.hpp"

#include "camera_transformation.hpp"
#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        Eigen::VectorXd get_signed_distance(const Eigen::VectorXd& pointA, const Eigen::VectorXd& pointB)
        {
            return (pointA - pointB);
        }

        vector2 get_3D_to_2D_distance_2D(const worldCoordinates& worldPoint, const screenCoordinates& cameraPoint, const worldToCameraMatrix& worldToCamera)
        {
            const vector2 cameraPointAs2D(cameraPoint.x(), cameraPoint.y());
            screenCoordinates projectedScreenPoint; 
            const bool isCoordinatesValid = compute_world_to_screen_coordinates(worldPoint, worldToCamera, projectedScreenPoint);
            if(isCoordinatesValid)
            {
                const vector2 screenPoint(projectedScreenPoint.x(), projectedScreenPoint.y());
                const vector2& distance = get_signed_distance(cameraPointAs2D, screenPoint);
                assert (not std::isnan(distance.x()));
                assert (not std::isnan(distance.y()));
                return distance;
            }
            // high number
            return vector2(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        }

        double get_3D_to_2D_distance(const worldCoordinates& worldPoint, const screenCoordinates& cameraPoint, const worldToCameraMatrix& worldToCamera)
        {
            const vector2& distance2D = get_3D_to_2D_distance_2D(worldPoint, cameraPoint, worldToCamera);
            if (distance2D.x() >= std::numeric_limits<double>::max() or distance2D.y() >= std::numeric_limits<double>::max())
                // high number
                return std::numeric_limits<double>::max();
            // compute manhattan distance (norm of power 1)
            return distance2D.lpNorm<1>();
        }

        vector3 get_3D_to_3D_distance_3D(const worldCoordinates& worldPoint, const screenCoordinates& cameraPoint, const cameraToWorldMatrix& cameraToWorld)
        {
            const worldCoordinates& cameraPointAs3D = screen_to_world_coordinates(cameraPoint, cameraToWorld);

            return get_signed_distance(worldPoint, cameraPointAs3D);
        }

        double get_3D_to_3D_distance(const worldCoordinates& worldPoint, const screenCoordinates& cameraPoint, const cameraToWorldMatrix& cameraToWorld)
        {
            const worldCoordinates& cameraPointAs3D = screen_to_world_coordinates(cameraPoint, cameraToWorld);

            return get_signed_distance(worldPoint, cameraPointAs3D).lpNorm<1>();
        }

    }   // utils
}       // rgbd_slam
