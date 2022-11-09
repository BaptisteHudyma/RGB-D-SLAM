#ifndef RGBDSLAM_UTILS_DISTANCE_UTILS_HPP
#define RGBDSLAM_UTILS_DISTANCE_UTILS_HPP

#include "coordinates.hpp"

namespace rgbd_slam {
    namespace utils {

        /**
         * \brief Compute a signed distance between two points
         */
        vectorxd get_signed_distance(const vectorxd& pointA, const vectorxd& pointB);

        /**
         * \brief Compute a signed 2D distance between a world point and a screen point, by retroprojecting the world point to screen space.
         * \param[in] worldPoint A 3D point in world space
         * \param[in] screenPoint A point in screen space. Only the x and y components will be used
         * \param[in] worldToCamera A transformation matrix to convert from world to camera space
         * \return a 2D signed distance in camera space (pixels)
         */
        vector2 get_3D_to_2D_distance_2D(const WorldCoordinate& worldPoint, const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera);

        /**
         * \brief Compute a distance between a world point and a screen point, by retroprojecting the world point to screen space.
         * \param[in] worldPoint A 3D point in world space
         * \param[in] screenPoint A point in screen space. Only the x and y components will be used
         * \param[in] worldToCamera A transformation matrix to convert from world to camera space
         * \return an unsigned distance in camera space (pixels)
         */
        double get_3D_to_2D_distance(const WorldCoordinate& worldPoint, const ScreenCoordinate2D& screenPoint, const worldToCameraMatrix& worldToCamera);

        /**
         * \brief Compute a signed distance between a world point and a 3D point in screen space, by projecting the screen point to world space
         * \param[in] worldPoint A 3D point in world space
         * \param[in] screenPoint A 3D point in screen space
         * \param[in] cameraToWorld A matrix to convert from camera to world space
         * \return The 3D signed distance in world space
         */
        vector3 get_3D_to_3D_distance_3D(const WorldCoordinate& worldPoint, const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld);

        /**
         * \brief Compute a distance between a world point and a 3D point in screen space, by projecting the screen point to world space
         * \param[in] worldPoint A 3D point in world space
         * \param[in] screenPoint A 3D point in screen space
         * \param[in] cameraToWorld A matrix to convert from camera to world space
         * \return The unsigned distance in world space
         */
        double get_3D_to_3D_distance(const WorldCoordinate& worldPoint, const ScreenCoordinate& screenPoint, const cameraToWorldMatrix& cameraToWorld);

        /**
         * \brief Compute a distance between two planes, by retroprojecting a world plane to camera space
         * \param[in] worldPlane A plane in world coordinates
         * \param[in] cameraPlane A plane in camera coordinates
         * \param[in] worldToCamera A transformation matrix to convert from world to camera space
         *
         * \return A 3D vector of the error between the two planes. The x and y are angle distances, the z is in millimeters
         */
        vector4 get_3D_to_2D_plane_distance(const PlaneWorldCoordinates& worldPlane, const PlaneCameraCoordinates& cameraPlane, const worldToCameraMatrix& worldToCamera);

    }   // utils
}       // rgbd_slam


#endif
