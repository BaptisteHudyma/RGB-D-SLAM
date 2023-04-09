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
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
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

    /**
     * \brief Return the number of pixels in this plane mask
     */
    uint get_contained_pixels() const
    {
        const static uint cellSize = Parameters::get_depth_map_patch_size();
        const static uint pixelPerCell = cellSize * cellSize;
        return cv::countNonZero(_shapeMask) * pixelPerCell;
    }

    utils::PlaneWorldCoordinates get_parametrization() const { return _parametrization; }
    utils::WorldCoordinate get_centroid() const { return _centroid; }
    cv::Mat get_mask() const { return _shapeMask; }
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

        // update centroid
        _centroid = detectedCentroid;

        // static sanity checks
        assert(utils::is_covariance_valid(_covariance));
        assert(not _parametrization.hasNaN());
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
    cv::Mat _shapeMask;                            // mask of the detected plane
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
        const vector6& descriptor =
                features::primitives::Plane::compute_descriptor(projectedPlane, planeCentroid, get_contained_pixels());
        const double similarityThreshold = useAdvancedSearch ? 0.2 : 0.4;

        double smallestSimilarity = std::numeric_limits<double>::max();
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

            // compute a similarity score
            const double descriptorSimilarity = shapePlane.get_similarity(descriptor);
            if (descriptorSimilarity < smallestSimilarity)
            {
                selectedIndex = planeIndex;
                smallestSimilarity = descriptorSimilarity;
            }
        }

        if (selectedIndex == UNMATCHED_FEATURE_INDEX or smallestSimilarity >= similarityThreshold)
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
        assert(not get_mask().empty());

        const double maskAlpha = 0.3;
        const cv::Size& debugImageSize = debugImage.size();

        cv::Mat planeMask;
        cv::Mat planeColorMask;
        // Resize with no interpolation
        cv::resize(get_mask() * 255, planeMask, debugImageSize, 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(planeMask, planeColorMask, cv::COLOR_GRAY2BGR);
        assert(planeMask.size == debugImage.size);
        assert(planeMask.size == debugImage.size);
        assert(planeColorMask.type() == debugImage.type());

        // merge with debug image
        planeColorMask.setTo(color, planeMask);
        cv::Mat maskedInput;
        cv::Mat ImaskedInput;
        // get masked original image, with the only visible part being the plane part
        debugImage.copyTo(maskedInput, planeMask);
        // get masked original image, with the only visible part being the non plane part
        debugImage.copyTo(ImaskedInput, 255 - planeMask);

        cv::addWeighted(maskedInput, (1 - maskAlpha), planeColorMask, maskAlpha, 0.0, maskedInput);

        debugImage = maskedInput + ImaskedInput;

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
        // TODO
        (void)worldToCamMatrix;
        return true;
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

        _shapeMask = matchedFeature.get_shape_mask().clone();
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
        _shapeMask = detectedFeature.get_shape_mask();

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
        _shapeMask = stagedPlane.get_mask();
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
