#ifndef RGBDSLAM_UTILS_COORDINATES_HPP
#define RGBDSLAM_UTILS_COORDINATES_HPP

#include "../types.hpp"

namespace rgbd_slam {
namespace utils {

    struct ScreenCoordinate;
    struct CameraCoordinate;
    struct WorldCoordinate;

    /**
    * \brief Return true is a measurement is in the measurement range
    */
    bool is_depth_valid(const double depth);

    /**
     * \brief Contains a single of coordinate in screen space.
     * Screen space is defined as (x, y) in pixels, and z in distance (millimeters)
     */
    struct ScreenCoordinate : public vector3 {
        /**
         * \brief Scores a 3D coordinate in screenspace (screen x, screen y, depth)
         */
        ScreenCoordinate() {};
        ScreenCoordinate(const vector2& coords, const double depth = 0) : vector3(coords.x(), coords.y(), depth) {};
        ScreenCoordinate(const double x, const double y) : vector3(x, y, 0) {};
        ScreenCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};

        /*
         * \brief Transform a screen point with a value to a 3D world point
         *
         * \param[in] cameraToWorld Matrix to transform local to world coordinates
         *
         * \return A 3D point in frame coordinates
         */
        WorldCoordinate to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const;
    };


    /**
     * \brief Contains a single of coordinate in camera space.
     * Camera space is defined as (x, y, z) in distance (millimeters), relative to the camera center
     */
    struct CameraCoordinate : public vector3 {
        /**
         * \brief Scores a 3D coordinate in camera (x, y, depth). It can be projected to world space using a pose transformation
         */
        CameraCoordinate() {};
        CameraCoordinate(const vector3& coords) : vector3(coords) {};
        CameraCoordinate(const vector4& homegenousCoordinates) : vector3(
            homegenousCoordinates.x()/homegenousCoordinates[3],
            homegenousCoordinates.y()/homegenousCoordinates[3],
            homegenousCoordinates.z()/homegenousCoordinates[3]
        ) {};
        CameraCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};
        vector4 get_homogenous() const { return vector4(x(), y(), z(), 1);};

        WorldCoordinate to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const;

        bool to_screen_coordinates(ScreenCoordinate& screenPoint) const;
    };


    /**
     * \brief Contains a single of coordinate in world space.
     * World space is defined as (x, y, z) in distance (millimeters), relative to the world center
     */
    struct WorldCoordinate : public vector3 {
        /**
         * \brief Scores a 3D coordinate in world space (x, y, depth).
         */
        WorldCoordinate() {};
        WorldCoordinate(const vector3& coords) : vector3(coords) {};
        WorldCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};

        /**
         * \brief Transform a point from world to screen coordinate system
         *
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         * \param[out] screenPoint The point screen coordinates, if the function returned true
         *
         * \return True if the screen position is valid
         */
        bool to_screen_coordinates(const worldToCameraMatrix& worldToCamera, ScreenCoordinate& screenPoint) const;

        /**
         * \brief Transform a vector in world space to a vector in camera space
         *
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         *
         * \return The input vector transformed to camera space
         */
        CameraCoordinate to_camera_coordinates(const worldToCameraMatrix& worldToCamera) const;
    };

}
}

#endif