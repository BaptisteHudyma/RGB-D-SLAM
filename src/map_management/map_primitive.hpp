#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "../features/primitives/shape_primitives.hpp"
#include "../parameters.hpp"
#include "../tracking/kalman_filter.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/coordinates.hpp"
#include "../matches_containers.hpp"
#include "../utils/random.hpp"
#include "covariances.hpp"
#include "distance_utils.hpp"
#include "feature_map.hpp"
#include "polygon.hpp"
#include "types.hpp"
#include <bits/ranges_algo.h>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace rgbd_slam::map_management {

using DetectedPlaneType = features::primitives::Plane;
using DetectedPlaneObject = features::primitives::plane_container;
using PlaneMatchType = matches_containers::PlaneMatch;
using TrackedPlaneObject = void*; // TODO implement

class Plane
{
  public:
    Plane()
    {
        _parametrization.setZero();
        _covariance.setZero();

        build_kalman_filter();
    }

    utils::PlaneWorldCoordinates get_parametrization() const { return _parametrization; }
    matrix44 get_covariance() const { return _covariance; };
    utils::WorldPolygon get_boundary_polygon() const { return _boundaryPolygon; };

    /**
     * \brief Update this plane coordinates using a new detection
     * \param[in] newDetectionParameters The detected plane parameters
     * \param[in] newDetectionCovariance The covariance of the newly detected feature
     * \return The update score (distance between old and new parametrization)
     */
    double track(const CameraToWorldMatrix& cameraToWorld,
                 const DetectedPlaneType& matchedFeature,
                 const utils::PlaneWorldCoordinates& newDetectionParameters,
                 const matrix44& newDetectionCovariance)
    {
        assert(utils::is_covariance_valid(newDetectionCovariance));
        assert(utils::is_covariance_valid(_covariance));

        const std::pair<vector4, matrix44>& res = _kalmanFilter->get_new_state(
                _parametrization, _covariance, newDetectionParameters, newDetectionCovariance);
        const utils::PlaneWorldCoordinates& newEstimatedParameters = res.first;
        const matrix44& newEstimatedCovariance = res.second;
        const double score = (_parametrization - newEstimatedParameters).norm();

        // covariance update
        _covariance = newEstimatedCovariance;

        // parameters update
        _parametrization =
                utils::PlaneWorldCoordinates(newEstimatedParameters.get_normal(), newEstimatedParameters.get_d());
        _parametrization.head(3).normalize();
        assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));

        // merge the boundary polygon (after optimization) with the observed polygon
        update_boundary_polygon(cameraToWorld, matchedFeature.get_boundary_polygon());

        // static sanity checks
        assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
        assert(not _covariance.hasNaN());
        assert(utils::is_covariance_valid(_covariance));
        assert(not _parametrization.hasNaN());
        return score;
    }

    utils::PlaneWorldCoordinates _parametrization; // parametrization of this plane in world space
    matrix44 _covariance;                          // covariance of this plane in world space
    utils::WorldPolygon _boundaryPolygon;          // polygon describing the boundary of the plane, in plane space

  private:
    /**
     * \brief Update the current boundary polygon with the one from the detected plane
     * \param[in] cameraToWorld The matrix to convert from caera to world space
     * \param[in] detectedPolygon The boundary polygon of the matched feature, to project to this plane space
     */
    void update_boundary_polygon(const CameraToWorldMatrix& cameraToWorld, const utils::CameraPolygon& detectedPolygon)
    {
        // correct the projection of the boundary polygon to correspond to the parametrization
        const vector3& worldPolygonNormal = _parametrization.get_normal();
        const vector3& worldPolygonCenter = _parametrization.get_center();
        _boundaryPolygon = _boundaryPolygon.project(worldPolygonNormal, worldPolygonCenter);
        assert(_boundaryPolygon.get_center().isApprox(worldPolygonCenter));

        // convert detected polygon to world space, it is supposed to be aligned with the world polygon
        const utils::WorldPolygon& projectedPolygon = detectedPolygon.to_world_space(cameraToWorld);

        // merge the projected observed polygon with optimized parameters with the current world polygon
        _boundaryPolygon.merge(projectedPolygon);
    }

    /**
     * \brief Build the parameter kalman filter
     */
    static void build_kalman_filter()
    {
        if (_kalmanFilter == nullptr)
        {
            const matrix44 systemDynamics = matrix44::Identity(); // planes are not supposed to move, so no dynamics
            const matrix44 outputMatrix = matrix44::Identity();   // we need all positions

            const double parametersProcessNoise = 0; // TODO set in parameters
            const matrix44 processNoiseCovariance =
                    matrix44::Identity() * parametersProcessNoise; // Process noise covariance

            _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<4, 4>>(
                    systemDynamics, outputMatrix, processNoiseCovariance);
        }
    }
    // shared kalman filter, between all planes
    inline static std::unique_ptr<tracking::SharedKalmanFilter<4, 4>> _kalmanFilter = nullptr;
};

class MapPlane :
    public Plane,
    public IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>
{
  public:
    MapPlane() : IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>()
    {
        assert(_id > 0);
    }

    MapPlane(const size_t id) :
        IMapFeature<DetectedPlaneObject, DetectedPlaneType, PlaneMatchType, TrackedPlaneObject>(id)
    {
        assert(_id > 0);
    }

    virtual ~MapPlane() = default;

    int find_match(const DetectedPlaneObject& detectedFeatures,
                   const WorldToCameraMatrix& worldToCamera,
                   const vectorb& isDetectedFeatureMatched,
                   std::list<PlaneMatchType>& matches,
                   const bool shouldAddToMatches = true,
                   const bool useAdvancedSearch = false) const override
    {
        const PlaneWorldToCameraMatrix& planeCameraToWorld = utils::compute_plane_world_to_camera_matrix(worldToCamera);
        // project plane in camera space
        const utils::PlaneCameraCoordinates& projectedPlane =
                get_parametrization().to_camera_coordinates_renormalized(planeCameraToWorld);

        const utils::CameraPolygon& projectedPolygon = _boundaryPolygon.to_camera_space(worldToCamera);

        // minimum plane overlap
        static double planeMinimalOverlap = Parameters::get_minimum_plane_overlap_for_match();
        const double areaSimilarityThreshold = (useAdvancedSearch ? planeMinimalOverlap / 2 : planeMinimalOverlap);

        double greatestSimilarity = 0.0;
        int selectedIndex = UNMATCHED_FEATURE_INDEX;

        // search best match score
        const int detectedPlaneSize = static_cast<int>(detectedFeatures.size());
        for (int planeIndex = 0; planeIndex < detectedPlaneSize; ++planeIndex)
        {
            if (isDetectedFeatureMatched[planeIndex])
                // Does not allow multiple removal of a single match
                // TODO: change this
                continue;

            assert(planeIndex >= 0 and planeIndex < detectedPlaneSize);
            const features::primitives::Plane& shapePlane = detectedFeatures[planeIndex];

            // if distance between planes is too great or angle between normals is further than a threshold, reject
            if (not shapePlane.is_distance_similar(projectedPlane) or not shapePlane.is_normal_similar(projectedPlane))
                continue;

            // compute a similarity score: compute the inter area of the map plane and the detected plane, divide it by
            // the detected plane area. Considers that the detected plane area should be lower than the map plane area
            const double detectedPlaneArea = shapePlane.get_boundary_polygon().area();
            const double interArea = shapePlane.get_boundary_polygon().inter_area(projectedPolygon);
            // similarity is greater than the greatest similarity, and overlap is greater than threshold
            if (interArea > greatestSimilarity and interArea / detectedPlaneArea > areaSimilarityThreshold)
            {
                selectedIndex = planeIndex;
                greatestSimilarity = interArea;
            }
        }

        if (selectedIndex == UNMATCHED_FEATURE_INDEX)
            return UNMATCHED_FEATURE_INDEX;

        if (shouldAddToMatches)
        {
            const features::primitives::Plane& shapePlane = detectedFeatures[selectedIndex];
            matches.emplace_back(shapePlane.get_parametrization(), get_parametrization(), get_covariance(), _id);
        }

        return selectedIndex;
    }

    bool add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                        TrackedPlaneObject& trackedFeatures,
                        const uint dropChance = 1000) const override
    {
        // silence warning for unused parameters
        (void)worldToCamera;
        (void)trackedFeatures;
        (void)dropChance;
        return false;
    }

    void draw(const WorldToCameraMatrix& worldToCamMatrix, cv::Mat& debugImage, const cv::Scalar& color) const override
    {
        // display the boundary of the plane
        _boundaryPolygon.display(worldToCamMatrix, color, debugImage);
    }

    bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const override
    {
        return _boundaryPolygon.to_camera_space(worldToCamMatrix).is_visible_in_screen_space();
    }

    void write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const override
    {
        mapWriter->add_polygon(_boundaryPolygon.get_unprojected_boundary());
    }

  protected:
    bool update_with_match(const DetectedPlaneType& matchedFeature,
                           const matrix33& poseCovariance,
                           const CameraToWorldMatrix& cameraToWorld) override
    {
        assert(_matchIndex >= 0);

        // compute projection matrices
        const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
        const matrix44& planeParameterCovariance = utils::compute_plane_covariance(
                matchedFeature.get_parametrization(), matchedFeature.get_point_cloud_covariance());

        // project to world coordinates
        const matrix44 worldCovariance = utils::get_world_plane_covariance(
                matchedFeature.get_parametrization(), planeCameraToWorld, planeParameterCovariance, poseCovariance);
        const utils::PlaneWorldCoordinates& projectedPlaneCoordinates =
                matchedFeature.get_parametrization().to_world_coordinates_renormalized(planeCameraToWorld);

        // update this plane with the other one's parameters
        track(cameraToWorld, matchedFeature, projectedPlaneCoordinates, worldCovariance);
        return true;
    }

    void update_no_match() override
    {
        // do nothing
    }
};

class StagedMapPlane : public MapPlane, public IStagedMapFeature<DetectedPlaneType>
{
  public:
    StagedMapPlane(const matrix33& poseCovariance,
                   const CameraToWorldMatrix& cameraToWorld,
                   const DetectedPlaneType& detectedFeature) :
        MapPlane()
    {
        // compute plane transition matrix and plane parameter covariance
        const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
        const matrix44& planeParameterCovariance = utils::compute_plane_covariance(
                detectedFeature.get_parametrization(), detectedFeature.get_point_cloud_covariance());

        // set parameters in world coordinates
        _parametrization = detectedFeature.get_parametrization().to_world_coordinates_renormalized(planeCameraToWorld);
        _covariance = utils::get_world_plane_covariance(
                detectedFeature.get_parametrization(), planeCameraToWorld, planeParameterCovariance, poseCovariance);
        _boundaryPolygon = detectedFeature.get_boundary_polygon().to_world_space(cameraToWorld);

        assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
    }

    bool should_remove_from_staged() const override { return _failedTrackingCount >= 2; }

    bool should_add_to_local_map() const override { return _successivMatchedCount >= 1; }
};

class LocalMapPlane : public MapPlane, public ILocalMapFeature<StagedMapPlane>
{
  public:
    LocalMapPlane(const StagedMapPlane& stagedPlane) : MapPlane(stagedPlane._id)
    {
        // new map point, new color
        set_color();

        _matchIndex = stagedPlane._matchIndex;
        _successivMatchedCount = stagedPlane._successivMatchedCount;

        _parametrization = stagedPlane.get_parametrization();
        _covariance = stagedPlane.get_covariance();
        _boundaryPolygon = stagedPlane._boundaryPolygon;

        assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
        assert(utils::is_covariance_valid(_covariance));
    }

    bool is_lost() const override
    {
        const static size_t maximumUnmatchBeforeremoval = Parameters::get_maximum_unmatched_before_removal();
        return _failedTrackingCount >= maximumUnmatchBeforeremoval;
    }
};

} // namespace rgbd_slam::map_management

#endif
