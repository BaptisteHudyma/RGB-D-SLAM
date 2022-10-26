#ifndef RGBDSLAM_UTILS_COORDINATES_HPP
#define RGBDSLAM_UTILS_COORDINATES_HPP

#include "../types.hpp"

namespace rgbd_slam {
namespace utils {

    struct screenCoordinates;
    struct cameraCoordinates;
    struct worldCoordinates;

    struct screenCoordinates : public vector3 {
        /**
         * \brief Scores a 3D coordinate in screenspace (screen x, screen y, depth)
         */
        screenCoordinates() {};
        screenCoordinates(const vector2& coords, const double depth = 0) : vector3(coords.x(), coords.y(), depth) {};
        screenCoordinates(const double x, const double y) : vector3(x, y, 0) {};
        screenCoordinates(const double x, const double y, const double z) : vector3(x, y, z) {};
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
    };
    struct worldCoordinates : public vector3 {
        /**
         * \brief Scores a 3D coordinate in world space (x, y, depth).
         */
        worldCoordinates() {};
        worldCoordinates(const vector3& coords) : vector3(coords) {};
        worldCoordinates(const double x, const double y, const double z) : vector3(x, y, z) {};
    };

}
}

#endif