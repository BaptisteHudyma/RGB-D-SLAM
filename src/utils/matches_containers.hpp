#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "types.hpp"

namespace rgbd_slam {
    namespace matches_containers {

        // KeyPoint matching: contains :
        //      - the coordinates of the detected point in screen space
        //      - the coordinates of the matched point in world space
        struct Match {
            Match(const vector3& screenPoint, const vector3& worldPoint) :
                _worldPoint(worldPoint),
                _screenPoint(screenPoint)
            {};

            vector3 _worldPoint;
            vector3 _screenPoint;
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
