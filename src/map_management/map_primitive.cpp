#include "map_primitive.hpp"

#include "camera_transformation.hpp"
#include "logger.hpp"
#include "parameters.hpp"
#include "distance_utils.hpp"

namespace rgbd_slam::map_management {

/**
 *  MapPlane
 */

int MapPlane::find_match(const DetectedPlaneObject& detectedFeatures,
                         const WorldToCameraMatrix& worldToCamera,
                         const vectorb& isDetectedFeatureMatched,
                         std::list<PlaneMatchType>& matches,
                         const bool shouldAddToMatches,
                         const bool useAdvancedSearch) const noexcept
{
    const PlaneWorldToCameraMatrix& planeCameraToWorld = utils::compute_plane_world_to_camera_matrix(worldToCamera);
    // project plane in camera space
    const PlaneCameraCoordinates& projectedPlane = get_parametrization().to_camera_coordinates(planeCameraToWorld);

    const utils::CameraPolygon& projectedPolygon = _boundaryPolygon.to_camera_space(worldToCamera);
    const double projectedArea = projectedPolygon.get_area();

    // minimum plane overlap
    static double planeMinimalOverlap = parameters::matching::minimumPlaneOverlapToConsiderMatch;
    const double areaSimilarityThreshold = (useAdvancedSearch ? planeMinimalOverlap / 2 : planeMinimalOverlap);

    double greatestSimilarity = 0.0;
    int selectedIndex = UNMATCHED_FEATURE_INDEX;

    if (projectedArea <= 0.0)
        return selectedIndex;

    // search best match score
    const int detectedPlaneSize = static_cast<int>(detectedFeatures.size());
    for (int planeIndex = 0; planeIndex < detectedPlaneSize; ++planeIndex)
    {
        if (isDetectedFeatureMatched[planeIndex])
            // Does not allow multiple removal of a single match
            // TODO: change this
            continue;

        const features::primitives::Plane& shapePlane = detectedFeatures[planeIndex];

        // if distance between planes is too great or angle between normals is further than a threshold, reject
        if (not shapePlane.is_distance_similar(projectedPlane) or not shapePlane.is_normal_similar(projectedPlane))
            continue;

        // compute a similarity score: compute the inter area of the map plane and the detected plane, divide it by
        // the detected plane area. Considers that the detected plane area should be lower than the map plane area
        const utils::CameraPolygon& detectedPolygon = shapePlane.get_boundary_polygon();
        // TODO: this metric fails as the plane becomes bigger
        const double newPlaneArea = detectedPolygon.get_area(); // max area of the two potential planes
        const double interArea = detectedPolygon.inter_area(projectedPolygon);
        // similarity is greater than the greatest similarity, and overlap is greater than threshold
        if (interArea > greatestSimilarity and interArea / newPlaneArea >= areaSimilarityThreshold)
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

bool MapPlane::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                              TrackedPlaneObject& trackedFeatures,
                              const uint dropChance) const noexcept
{
    // TODO: track 2D points
    // silence warning for unused parameters
    (void)worldToCamera;
    (void)trackedFeatures;
    (void)dropChance;
    return false;
}

void MapPlane::draw(const WorldToCameraMatrix& worldToCamMatrix,
                    cv::Mat& debugImage,
                    const cv::Scalar& color) const noexcept
{
    if (not is_matched())
        return;

    // display the boundary of the plane
    _boundaryPolygon.display(worldToCamMatrix, color, debugImage);
}

bool MapPlane::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const noexcept
{
    return _boundaryPolygon.to_camera_space(worldToCamMatrix).is_visible_in_screen_space();
}

void MapPlane::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const noexcept
{
    if (mapWriter != nullptr)
    {
        mapWriter->add_polygon(_boundaryPolygon.get_unprojected_boundary(), _boundaryPolygon.get_normal());
    }
    else
    {
        outputs::log_error("mapWriter is null");
        exit(-1);
    }
}

bool MapPlane::update_with_match(const DetectedPlaneType& matchedFeature,
                                 const matrix33& poseCovariance,
                                 const CameraToWorldMatrix& cameraToWorld) noexcept
{
    if (_matchIndex < 0)
    {
        outputs::log_error("Tries to call the function update_with_match with no associated match");
        return false;
    }

    // compute projection matrices
    try
    {
        const PlaneCameraCoordinates& matchedFeatureParams = matchedFeature.get_parametrization();
        const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
        const matrix44& planeParameterCovariance =
                utils::compute_plane_covariance(matchedFeatureParams, matchedFeature.get_point_cloud_covariance());

        if (not utils::is_covariance_valid(planeParameterCovariance))
        {
            outputs::log_error(
                    "MapPlane: Covariance of the detected plane is invalid after projecting it from point cloud "
                    "covariance");
            return false;
        }

        // project to world coordinates
        const matrix44 worldCovariance = utils::get_world_plane_covariance(
                matchedFeatureParams, cameraToWorld, planeCameraToWorld, planeParameterCovariance, poseCovariance);
        if (not utils::is_covariance_valid(worldCovariance))
        {
            outputs::log_error(
                    "MapPlane: Covariance of the detected plane is invalid after projecting it to world space");
            return false;
        }

        const PlaneWorldCoordinates& projectedPlaneCoordinates =
                matchedFeatureParams.to_world_coordinates(planeCameraToWorld);

        // update this plane with the other one's parameters
        track(cameraToWorld, matchedFeature, projectedPlaneCoordinates, worldCovariance);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void MapPlane::update_no_match() noexcept
{
    // do nothing
}

/**
 *  StagedMapPlane
 */

StagedMapPlane::StagedMapPlane(const matrix33& poseCovariance,
                               const CameraToWorldMatrix& cameraToWorld,
                               const DetectedPlaneType& detectedFeature) :
    MapPlane()
{
    // compute plane transition matrix and plane parameter covariance
    const PlaneCameraCoordinates& detectedFeatureParams = detectedFeature.get_parametrization();
    const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
    const matrix44& planeParameterCovariance =
            utils::compute_plane_covariance(detectedFeatureParams, detectedFeature.get_point_cloud_covariance());

    // set parameters in world coordinates
    _parametrization = detectedFeatureParams.to_world_coordinates(planeCameraToWorld);

    _covariance = utils::get_world_plane_covariance(
            detectedFeatureParams, cameraToWorld, planeCameraToWorld, planeParameterCovariance, poseCovariance);
    _boundaryPolygon = detectedFeature.get_boundary_polygon().to_world_space(cameraToWorld);

    if (not utils::double_equal(_parametrization.get_normal().norm(), 1.0))
    {
        throw std::invalid_argument("parametrization of detected feature as an invalid normal vector");
    }
}

bool StagedMapPlane::should_remove_from_staged() const noexcept { return _failedTrackingCount >= 2; }

bool StagedMapPlane::should_add_to_local_map() const noexcept { return _successivMatchedCount >= 2; }

/**
 *  LocalMapPlane
 */

LocalMapPlane::LocalMapPlane(const StagedMapPlane& stagedPlane) : MapPlane(stagedPlane._id)
{
    // new map point, new color
    set_color();

    _matchIndex = stagedPlane._matchIndex;
    _successivMatchedCount = stagedPlane._successivMatchedCount;

    _parametrization = stagedPlane.get_parametrization();
    _covariance = stagedPlane.get_covariance();
    _boundaryPolygon = stagedPlane._boundaryPolygon;

    if (not utils::is_covariance_valid(_covariance))
    {
        throw std::invalid_argument("covariance of stagedPlane is an invalid covariance matrix");
    }
    if (not utils::double_equal(_parametrization.get_normal().norm(), 1.0))
    {
        throw std::invalid_argument("parametrization of stagedPlane as an invalid normal vector");
    }
}

bool LocalMapPlane::is_lost() const noexcept
{
    return _failedTrackingCount >= parameters::mapping::planeUnmatchedCountToLoose;
}

} // namespace rgbd_slam::map_management