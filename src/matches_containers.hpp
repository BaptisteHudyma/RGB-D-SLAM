#ifndef RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP
#define RGBDSLAM_UTILS_MATCHESCONTAINERS_HPP

#include "coordinates.hpp"
#include <list>

namespace rgbd_slam::matches_containers {

template<class FeatureCameraSpace, class FeatureWorldSpace, class WorldFeatureCovariance> struct MatchTemplate
{
    MatchTemplate(const FeatureCameraSpace& screenfeature,
                  const FeatureWorldSpace& worldFeature,
                  const WorldFeatureCovariance& worldfeatureCovariance,
                  const size_t mapId) :
        _screenFeature(screenfeature),
        _worldFeature(worldFeature),
        _worldFeatureCovariance(worldfeatureCovariance),
        _idInMap(mapId) {};

    FeatureCameraSpace _screenFeature;              // Coordinates of the detected screen point
    FeatureWorldSpace _worldFeature;                // coordinates of the local world point
    WorldFeatureCovariance _worldFeatureCovariance; // Covariance of this feature in world space
    size_t _idInMap;                                // Id of the world feature in the local map
};

// KeyPoint matching: contains :
//      - the coordinates of the detected point in screen space
//      - the coordinates of the matched point in world space
//      - the diagonal of the covariance of the world point in world space
using PointMatch = MatchTemplate<utils::ScreenCoordinate, utils::WorldCoordinate, vector3>;
using match_point_container = std::list<PointMatch>;

// MapPlane matching: contains :
//      - the normal vector of the plane in camera space
//      - the normal vector of the plane in world space
//      - the covariance of the world plane in world space
using PlaneMatch = MatchTemplate<utils::PlaneCameraCoordinates, utils::PlaneWorldCoordinates, matrix44>;
using match_plane_container = std::list<PlaneMatch>;

struct matchContainer
{
    match_point_container _points;
    match_plane_container _planes;

    void clear()
    {
        _points.clear();
        _planes.clear();
    }

    void swap(matchContainer& other) noexcept
    {
        _points.swap(other._points);
        _planes.swap(other._planes);
    }

    size_t size() const { return _points.size() + _planes.size(); };
};

/**
 * \brief Store a set of inlier and a set of outliers
 */
template<class Container> struct match_sets_template
{
    Container _inliers;
    Container _outliers;

    void clear()
    {
        _inliers.clear();
        _outliers.clear();
    }

    void swap(match_sets_template& other) noexcept
    {
        _inliers.swap(other._inliers);
        _outliers.swap(other._outliers);
    }
};

// store a set of inliers and a set of outliers for points
using point_match_sets = match_sets_template<match_point_container>;
// store a set of inliers and a set of outliers for planes
using plane_match_sets = match_sets_template<match_plane_container>;

// store a set of inliers and a set of outliers for all features
struct match_sets
{
    point_match_sets _pointSets;
    plane_match_sets _planeSets;

    void clear()
    {
        _pointSets.clear();
        _planeSets.clear();
    }

    void swap(match_sets& other) noexcept
    {
        _pointSets.swap(other._pointSets);
        _planeSets.swap(other._planeSets);
    }
};

} // namespace rgbd_slam::matches_containers

#endif
