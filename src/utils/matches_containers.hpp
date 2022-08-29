#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "../types.hpp"

namespace rgbd_slam {
    namespace matches_containers {

        // KeyPoint matching: contains :
        //      - the coordinates of the detected point in screen space
        //      - the coordinates of the matched point in world space
        struct Match {
            Match(const vector3& screenPoint, const vector3& worldPoint, const size_t mapId) :
                _worldPoint(worldPoint),
                _screenPoint(screenPoint),
                _mapPointId(mapId)
            {};

            vector3 _worldPoint;    // coordinates of the local world point
            vector3 _screenPoint;   // Coordinates of the detected screen point
            size_t _mapPointId;     // Id of the world point in the local map 
        };
        typedef std::list<Match> match_point_container;

        // Primitive matching: contains :
        //      - the normal vector of the primitive in screen space
        //      - the normal vector of the primitive in world space
        typedef std::pair<vector3, vector3> primitive_pair;
        typedef std::list<primitive_pair> match_primitive_container;
    }
}

#endif
