#include "distance_utils.hpp"

#include "camera_transformation.hpp"

namespace rgbd_slam {
    namespace utils {

        double get_distance_euclidean(const Eigen::VectorXd& pointA, const Eigen::VectorXd& pointB)
        {
            return (pointA - pointB).norm();
        }

        double get_distance_manhattan(const Eigen::VectorXd& pointA, const Eigen::VectorXd& pointB)
        {
            return (pointA - pointB).lpNorm<1>();
        }

        double get_3D_to_2D_distance(const vector3& worldPoint, const Eigen::VectorXd& cameraPoint, const matrix44& worldToCameraTransformationMatrix)
        {
            const vector2 cameraPointAs2D(cameraPoint.x(), cameraPoint.y());
            vector2 worldPointAs2D; 
            const bool isCoordinatesValid = utils::world_to_screen_coordinates(worldPoint, worldToCameraTransformationMatrix, worldPointAs2D);
            if(isCoordinatesValid)
            {
                const double distance = get_distance_manhattan(cameraPointAs2D, worldPointAs2D);
                assert (not std::isnan(distance) and distance >= 0);
                return distance;
            }
            // high number
            return std::numeric_limits<double>::max();
        }

        double get_3D_to_3D_distance(const vector3& worldPoint, const vector3& cameraPoint, const matrix44& cameraToWorldTransformationMatrix)
        {
            const vector3& cameraPointAs3D = utils::screen_to_world_coordinates( cameraPoint.x(), cameraPoint.y(), cameraPoint.z(), cameraToWorldTransformationMatrix);

            return get_distance_manhattan(worldPoint, cameraPointAs3D);
        }

    }   // utils
}       // rgbd_slam
