#ifndef RGBDSLAM_UTILS_COORDINATES_HPP
#define RGBDSLAM_UTILS_COORDINATES_HPP

#include "../types.hpp"

namespace rgbd_slam {
namespace utils {

    struct screenCoordinates;
    struct cameraCoordinates;
    struct worldCoordinates;

    /**
    * \brief Return true is a measurement is in the measurement range
    */
    bool is_depth_valid(const double depth);

    struct screenCoordinates : public vector3 {
        /**
         * \brief Scores a 3D coordinate in screenspace (screen x, screen y, depth)
         */
        screenCoordinates() {};
        screenCoordinates(const vector2& coords, const double depth = 0) : vector3(coords.x(), coords.y(), depth) {};
        screenCoordinates(const double x, const double y) : vector3(x, y, 0) {};
        screenCoordinates(const double x, const double y, const double z) : vector3(x, y, z) {};

        /*
         * \brief Transform a screen point with a value to a 3D world point
         *
         * \param[in] cameraToWorld Matrix to transform local to world coordinates
         *
         * \return A 3D point in frame coordinates
         */
        worldCoordinates to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const;
    };


    struct cameraCoordinates : public vector3 {
        /**
         * \brief Scores a 3D coordinate in camera (x, y, depth). It can be projected to world space using a pose transformation
         */
        cameraCoordinates() {};
        cameraCoordinates(const vector3& coords) : vector3(coords) {};
        cameraCoordinates(const vector4& homegenousCoordinates) : vector3(
            homegenousCoordinates.x()/homegenousCoordinates[3],
            homegenousCoordinates.y()/homegenousCoordinates[3],
            homegenousCoordinates.z()/homegenousCoordinates[3]
        ) {};
        cameraCoordinates(const double x, const double y, const double z) : vector3(x, y, z) {};
        vector4 get_homogenous() const { return vector4(x(), y(), z(), 1);};

        worldCoordinates to_world_coordinates(const cameraToWorldMatrix& cameraToWorld) const;

        bool to_screen_coordinates(screenCoordinates& screenPoint) const;
    };


    struct worldCoordinates : public vector3 {
        /**
         * \brief Scores a 3D coordinate in world space (x, y, depth).
         */
        worldCoordinates() {};
        worldCoordinates(const vector3& coords) : vector3(coords) {};
        worldCoordinates(const double x, const double y, const double z) : vector3(x, y, z) {};

        /**
         * \brief Transform a point from world to screen coordinate system
         *
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         * \param[out] screenPoint The point screen coordinates, if the function returned true
         *
         * \return True if the screen position is valid
         */
        bool to_screen_coordinates(const worldToCameraMatrix& worldToCamera, screenCoordinates& screenPoint) const;

        /**
         * \brief Transform a vector in world space to a vector in camera space
         *
         * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
         *
         * \return The input vector transformed to camera space
         */
        cameraCoordinates to_camera_coordinates(const worldToCameraMatrix& worldToCamera) const;
    };

}
}

#endif