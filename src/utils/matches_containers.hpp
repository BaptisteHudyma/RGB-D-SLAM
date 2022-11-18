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
        struct PlaneMatch {
            PlaneMatch(const utils::PlaneCameraCoordinates& cameraPlane, const utils::PlaneWorldCoordinates& worldPlane, const size_t mapId) :
                _worldPlane(worldPlane),
                _cameraPlane(cameraPlane),
                _mapPlaneId(mapId)
            {};

            utils::PlaneWorldCoordinates _worldPlane;
            utils::PlaneCameraCoordinates _cameraPlane;
            size_t _mapPlaneId;     // Id of the world plane in the local map 
        };
        typedef std::list<PlaneMatch> match_plane_container;

        // store a set of inliers and a set of outliers for points
        struct point_match_sets {
            match_point_container _inliers;
            match_point_container _outliers;

            void clear()
            {
                _inliers.clear();
                _outliers.clear();
            }

            void swap(point_match_sets& other)
            {
                _inliers.swap(other._inliers);
                _outliers.swap(other._outliers);
            }
        };

        // store a set of inliers and a set of outliers for planes
        struct plane_match_sets {
            match_plane_container _inliers;
            match_plane_container _outliers;

            void clear()
            {
                _inliers.clear();
                _outliers.clear();
            }

            void swap(plane_match_sets& other)
            {
                _inliers.swap(other._inliers);
                _outliers.swap(other._outliers);
            }
        };

        // store a set of inliers and a set of outliers for all features
        struct match_sets {
            point_match_sets _pointSets;
            plane_match_sets _planeSets;

            void clear()
            {
                _pointSets.clear();
                _planeSets.clear();
            }
        };
    }
}

#endif
