#ifndef RGBDSLAM_MATCHES_CONTAINERS_HPP
#define RGBDSLAM_MATCHES_CONTAINERS_HPP

#include "types.hpp"

namespace rgbd_slam {
    namespace matches_containers {

        // KeyPoint matching
        typedef std::pair<vector3, vector3> point_pair;
        typedef std::list<point_pair> match_point_container;

    }
}

#endif
