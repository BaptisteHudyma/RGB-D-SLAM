#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "coordinates.hpp"

#include <list>

namespace rgbd_slam {
    namespace matches_containers {

        // KeyPoint matching: contains :
        //      - the coordinates of the detected point in screen space
        //      - the coordinates of the matched point in world space
        struct PointMatch {
            PointMatch(const utils::ScreenCoordinate& screenPoint, const utils::WorldCoordinate& worldPoint, const size_t mapId) :
                _worldPoint(worldPoint),
                _screenPoint(screenPoint),
                _mapPointId(mapId)
            {};

            utils::WorldCoordinate _worldPoint;    // coordinates of the local world point
            utils::ScreenCoordinate _screenPoint;   // Coordinates of the detected screen point
            size_t _mapPointId;     // Id of the world point in the local map 
        };
        typedef std::list<PointMatch> match_point_container;

        // MapPlane matching: contains :
        //      - the normal vector of the plane in camera space
        //      - the normal vector of the plane in world space
        typedef std::pair<utils::PlaneCameraCoordinates, utils::PlaneWorldCoordinates> plane_pair;
        typedef std::list<plane_pair> match_plane_container;
    }
}

#endif
