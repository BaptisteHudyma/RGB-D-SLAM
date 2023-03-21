#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "coordinates.hpp"

#include <list>

namespace rgbd_slam {
    namespace matches_containers {

        template<class FeatureCameraSpace, class FeatureWorldSpace, class WorldFeatureCovariance, class ProjectedWorldFeatureCovariance>
        struct MatchTemplate {
            MatchTemplate(const FeatureCameraSpace& screenfeature, const FeatureWorldSpace& worldFeature, const WorldFeatureCovariance& worldfeatureCovariance, const ProjectedWorldFeatureCovariance& projectedWorldfeatureCovariance, const size_t mapId) :
                _screenFeature(screenfeature),
                _worldFeature(worldFeature),
                _worldFeatureCovariance(worldfeatureCovariance),
                _projectedWorldfeatureCovariance(projectedWorldfeatureCovariance),
                _idInMap(mapId)
            {};

            FeatureCameraSpace _screenFeature;   // Coordinates of the detected screen point
            FeatureWorldSpace _worldFeature;    // coordinates of the local world point
            WorldFeatureCovariance _worldFeatureCovariance; //projected to screen space
            ProjectedWorldFeatureCovariance _projectedWorldfeatureCovariance;
            size_t _idInMap;     // Id of the world feature in the local map 
        };

        // KeyPoint matching: contains :
        //      - the coordinates of the detected point in screen space
        //      - the coordinates of the matched point in world space
        //      - the covariance of the world plane projected in screen space
        typedef MatchTemplate<utils::ScreenCoordinate, utils::WorldCoordinate, vector3, vector2> PointMatch;
        typedef std::list<PointMatch> match_point_container;

        // MapPlane matching: contains :
        //      - the normal vector of the plane in camera space
        //      - the normal vector of the plane in world space
        //      - the covariance of the world plane projected in camera space
        typedef MatchTemplate<utils::PlaneCameraCoordinates, utils::PlaneWorldCoordinates, void*, void*> PlaneMatch;
        typedef std::list<PlaneMatch> match_plane_container;

        struct matchContainer {
            match_point_container _points;
            match_plane_container _planes;

            void clear()
            {
                _points.clear();
                _planes.clear();
            }

            void swap(matchContainer& other)
            {
                _points.swap(other._points);
                _planes.swap(other._planes);
            }

            size_t size() const { return _points.size() + _planes.size(); };
        };


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

            void swap(match_sets& other)
            {
                _pointSets.swap(other._pointSets);
                _planeSets.swap(other._planeSets);
            }
        };
    }
}

#endif
