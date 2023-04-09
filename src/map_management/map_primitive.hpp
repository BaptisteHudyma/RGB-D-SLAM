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
        _centroid.setZero();

        build_kalman_filter();
    }

    utils::PlaneWorldCoordinates get_parametrization() const { return _parametrization; }
    utils::WorldCoordinate get_centroid() const { return _centroid; }
    matrix44 get_covariance() const { return _covariance; };
    utils::Polygon get_boundary_polygon() const { return _boundaryPolygon; };

    /**
     * \brief Update this plane coordinates using a new detection
     * \param[in] newDetectionParameters The detected plane parameters
     * \param[in] newDetectionCovariance The covariance of the newly detected feature
     * \param[in] detectedCentroid The centroid of the detected plane
     * \return The update score (distance between old and new parametrization)
     */
    double track(const utils::PlaneWorldCoordinates& newDetectionParameters,
                 const matrix44& newDetectionCovariance,
                 const utils::WorldCoordinate& detectedCentroid)
    {
        assert(utils::is_covariance_valid(newDetectionCovariance));
        assert(utils::is_covariance_valid(_covariance));

        const std::pair<vector4, matrix44>& res = _kalmanFilter->get_new_state(
                _parametrization, _covariance, newDetectionParameters, newDetectionCovariance);
        const vector4& newEstimatedParameters = res.first;
        const matrix44& newEstimatedCovariance = res.second;
        const double score = (_parametrization - newEstimatedParameters).norm();

        // covariance update
        _covariance = newEstimatedCovariance;

        // parameters update
        _parametrization = newEstimatedParameters;
        _parametrization.head(3).normalize();

        // update centroid (low pass filter)
        _centroid << (_centroid * 0.9 + detectedCentroid * 0.1);

        // static sanity checks
        assert(utils::is_covariance_valid(_covariance));
        assert(not _parametrization.hasNaN());
        assert(not _centroid.hasNaN());
        return score;
    }

    /**
     * \brief Update the current boundary polygon with the one from the detected plane
     * \param[in] detectedFeatureParameters the matched feature parameters, projected to world coordinates
     * \param[in] detectedFeatureCenter the matched feature centroid, projected to world coordinates
     * \param[in] detectedPolygon The boundary polygon of the matched feature, to project to this plane space
     */
    void update_boundary_polygon(const utils::PlaneWorldCoordinates& detectedFeatureParameters,
                                 const utils::WorldCoordinate& detectedFeatureCenter,
                                 const utils::Polygon& detectedPolygon)
    {
        const std::pair<vector3, vector3>& detectedPlaneVectors =
                utils::get_plane_coordinate_system(detectedFeatureParameters.head(3));
        const vector3& detectedPlaneCenter = detectedFeatureCenter;
        const vector3& uVecDetection = detectedPlaneVectors.first;
        const vector3& vVecDetection = detectedPlaneVectors.second;

        const std::pair<vector3, vector3>& nextPlaneVectors =
                utils::get_plane_coordinate_system(_parametrization.head(3));
        const vector3& nextPlaneCenter = _centroid;
        const vector3& uVecNext = nextPlaneVectors.first;
        const vector3& vVecNext = nextPlaneVectors.second;

        _boundaryPolygon.merge(detectedPolygon.project(
                detectedPlaneCenter, uVecDetection, vVecDetection, nextPlaneCenter, uVecNext, vVecNext));
    }

    utils::PlaneWorldCoordinates _parametrization; // parametrization of this plane in world space
    matrix44 _covariance;                          // covariance of this plane in world space
    utils::WorldCoordinate _centroid;              // centroid of the detected plane
    utils::Polygon _boundaryPolygon;               // polygon describing the boundary of the plane, in plane space

  private:
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
                get_parametrization().to_camera_coordinates(planeCameraToWorld);
        const utils::CameraCoordinate& planeCentroid = get_centroid().to_camera_coordinates(worldToCamera);

        // get the plane cooridnate system
        const std::pair<vector3, vector3>& detectedPlaneVectors =
                utils::get_plane_coordinate_system(projectedPlane.head(3));
        const vector3& detectedPlaneCenter = planeCentroid;
        const vector3& uVecDetection = detectedPlaneVectors.first;
        const vector3& vVecDetection = detectedPlaneVectors.second;

        const double areaSimilarityThreshold = (useAdvancedSearch ? 0.6 : 0.8);

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

            // if angle between normals is further than a threshold, reject match
            if (not shapePlane.is_normal_similar(projectedPlane))
                continue;

            // project the plane boundary to detected plane space
            const std::pair<vector3, vector3>& nextPlaneVectors =
                    utils::get_plane_coordinate_system(shapePlane.get_normal());
            const vector3& nextPlaneCenter = shapePlane.get_centroid();
            const vector3& uVecNext = nextPlaneVectors.first;
            const vector3& vVecNext = nextPlaneVectors.second;

            const utils::Polygon& projectedBoundary = _boundaryPolygon.project(
                    detectedPlaneCenter, uVecDetection, vVecDetection, nextPlaneCenter, uVecNext, vVecNext);

            // compute a similarity score: compute the inter area of the map plane and the detected plane, divide it by
            // the detected plane area. Considers that the detected plane area should be lower than the map plane area
            const double detectedPlaneArea = shapePlane.get_boundary_polygon().area();
            const double interArea = shapePlane.get_boundary_polygon().inter_area(projectedBoundary);
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
        // project plane in camera space
        const utils::PlaneCameraCoordinates& projectedPlane = get_parametrization().to_camera_coordinates(
                utils::compute_plane_world_to_camera_matrix(worldToCamMatrix));
        const vector3& normal = projectedPlane.head(3).normalized();
        const vector3& center = _centroid.to_camera_coordinates(worldToCamMatrix);

        // find arbitrary othogonal vectors of the normal
        const std::pair<vector3, vector3>& res = utils::get_plane_coordinate_system(normal);
        const vector3& uVec = res.first;
        const vector3& vVec = res.second;

        // display the boundary of the plane
        _boundaryPolygon.display(center, uVec, vVec, color, debugImage);
    }

    bool is_visible(const WorldToCameraMatrix& worldToCamMatrix) const override
    {
        static const uint screenSizeX = Parameters::get_camera_1_size_x();
        static const uint screenSizeY = Parameters::get_camera_1_size_y();

        // project plane to camera space
        const utils::PlaneCameraCoordinates& projectedPlane = get_parametrization().to_camera_coordinates(
                utils::compute_plane_world_to_camera_matrix(worldToCamMatrix));
        const vector3& normal = projectedPlane.head(3).normalized();
        const vector3& center = _centroid.to_camera_coordinates(worldToCamMatrix);

        // get plane coordinate system
        const std::pair<vector3, vector3>& planeVectors = utils::get_plane_coordinate_system(normal);
        const vector3& uVec = planeVectors.first;
        const vector3& vVec = planeVectors.second;

        // if any point of the boundary is in the camera frame, return true
        return std::ranges::any_of(_boundaryPolygon.get_boundary(), [center, uVec, vVec](const vector2& p) {
            // project to camera coordinates
            const utils::CameraCoordinate& cameraPoint = utils::get_point_from_plane_coordinates(p, center, uVec, vVec);
            // project to screen coordinates
            if (utils::ScreenCoordinate2D projection; cameraPoint.to_screen_coordinates(projection))
            {
                // at least one boundary point is visible
                return (projection.x() >= 0 and projection.x() <= screenSizeX and projection.y() >= 0 and
                        projection.y() <= screenSizeY);
            }
            return false;
        });
    }

  protected:
    bool update_with_match(const DetectedPlaneType& matchedFeature,
                           const matrix33& poseCovariance,
                           const CameraToWorldMatrix& cameraToWorld) override
    {
        assert(_matchIndex >= 0);

        const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

        const matrix44& planeParameterCovariance = utils::compute_plane_covariance(
                matchedFeature.get_parametrization(), matchedFeature.get_point_cloud_covariance(), poseCovariance);
        const matrix44 worldCovariance = planeCameraToWorld * planeParameterCovariance * planeCameraToWorld.transpose();

        // project to world coordinates
        const utils::PlaneWorldCoordinates& projectedPlaneCoordinates =
                matchedFeature.get_parametrization().to_world_coordinates_renormalized(planeCameraToWorld);
        const utils::WorldCoordinate& projectedPlaneCenter =
                matchedFeature.get_centroid().to_world_coordinates(cameraToWorld);

        // update this plane with the other one parameters
        track(projectedPlaneCoordinates, worldCovariance, projectedPlaneCenter);
        // merge the boundary polygon (after optimization)
        update_boundary_polygon(projectedPlaneCoordinates, projectedPlaneCenter, matchedFeature.get_boundary_polygon());
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
        const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
        _parametrization = detectedFeature.get_parametrization().to_world_coordinates_renormalized(planeCameraToWorld);
        _centroid = detectedFeature.get_centroid().to_world_coordinates(cameraToWorld);

        const matrix44& planeParameterCovariance = utils::compute_plane_covariance(
                detectedFeature.get_parametrization(), detectedFeature.get_point_cloud_covariance(), poseCovariance);
        _covariance = planeCameraToWorld * planeParameterCovariance * planeCameraToWorld.transpose();
        _boundaryPolygon = detectedFeature.get_boundary_polygon();

        assert(utils::double_equal(_parametrization.head(3).norm(), 1.0));
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
        _centroid = stagedPlane.get_centroid();
        _covariance = stagedPlane.get_covariance();
        _boundaryPolygon = stagedPlane._boundaryPolygon;

        assert(utils::double_equal(_parametrization.head(3).norm(), 1.0));
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
