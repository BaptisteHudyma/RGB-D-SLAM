#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include "types.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
         * \brief Compute a distance between two 3D points
         * \return an unsigned euclidean distance
         */
        double get_distance_euclidean(const Eigen::VectorXd& pointA, const Eigen::VectorXd& pointB);

        /**
         * \brief Compute a distance between two 3D points
         * \return an unsigned manhattan distance
         */
        double get_distance_manhattan(const Eigen::VectorXd& pointA, const Eigen::VectorXd& pointB);

        /**
         * \brief Compute a distance between a world point and a camera point, by retroprojecting the world point to camera space.
         * \param[in] worldPoint A 3D point in world space
         * \param[in] cameraPoint A point in camera space. Only the x and y components will be used
         * \param[in] worldToCameraTransformationMatrix A transformation matrix to convert from world to camera space
         * \return an unsigned distance in camera space (pixels)
         */
        double get_3D_to_2D_distance(const vector3& worldPoint, const Eigen::VectorXd& cameraPoint, const matrix44& worldToCameraTransformationMatrix);

        /**
         * \brief Compute a distance between a world point and a 3D point in camera space, by projecting the camera point to world space
         * \param[in] worldPoint A 3D point in world space
         * \param[in] cameraPoint A 3D point in camera space
         * \param[in] cameraToWorldTransformationMatrix A matrix to convert from camera to world space
         * \return The unsigned distance in world space
         */
        double get_3D_to_3D_distance(const vector3& worldPoint, const vector3& cameraPoint, const matrix44& cameraToWorldTransformationMatrix);

    }   // utils
}       // rgbd_slam


#endif
