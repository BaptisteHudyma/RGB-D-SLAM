#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "coordinates.hpp"

#include <list>

namespace rgbd_slam {
    namespace matches_containers {

        template<class FeatureWorldSpace, class FeatureCameraSpace>
        struct MatchTemplate {
            MatchTemplate(const FeatureCameraSpace& screenfeature, const FeatureWorldSpace& worldFeature, const size_t mapId) :
                _worldFeature(worldFeature),
                _screenFeature(screenfeature),
                _idInMap(mapId)
            {};

            FeatureWorldSpace _worldFeature;    // coordinates of the local world point
            FeatureCameraSpace _screenFeature;   // Coordinates of the detected screen point
            size_t _idInMap;     // Id of the world feature in the local map 
        };

        // KeyPoint matching: contains :
        //      - the coordinates of the detected point in screen space
        //      - the coordinates of the matched point in world space
        typedef MatchTemplate<utils::WorldCoordinate, utils::ScreenCoordinate> PointMatch;
        typedef std::list<PointMatch> match_point_container;

        // MapPlane matching: contains :
        //      - the normal vector of the plane in camera space
        //      - the normal vector of the plane in world space
        typedef MatchTemplate<utils::PlaneWorldCoordinates, utils::PlaneCameraCoordinates> PlaneMatch;
        typedef std::list<PlaneMatch> match_plane_container;


        /**
         * \brief Store a set of inlier and a set of outliers
         */
        template<class Container>
        struct match_sets_template {
            Container _inliers;
            Container _outliers;

            void clear()
            {
                _inliers.clear();
                _outliers.clear();
            }

            void swap(match_sets_template& other)
            {
                _inliers.swap(other._inliers);
                _outliers.swap(other._outliers);
            }
        };

        // store a set of inliers and a set of outliers for points
        typedef match_sets_template<match_point_container> point_match_sets; 
        // store a set of inliers and a set of outliers for planes
        typedef match_sets_template<match_plane_container> plane_match_sets; 

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
