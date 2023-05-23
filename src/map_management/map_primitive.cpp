#include "map_primitive.hpp"

namespace rgbd_slam::map_management {

/**
 *  Plane
 */

Plane::Plane()
{
    _covariance.setZero();

    build_kalman_filter();
}

double Plane::track(const CameraToWorldMatrix& cameraToWorld,
                    const DetectedPlaneType& matchedFeature,
                    const utils::PlaneWorldCoordinates& newDetectionParameters,
                    const matrix44& newDetectionCovariance)
{
    assert(_kalmanFilter != nullptr);
    assert(utils::is_covariance_valid(newDetectionCovariance));
    assert(utils::is_covariance_valid(_covariance));

    const std::pair<vector4, matrix44>& res = _kalmanFilter->get_new_state(_parametrization.get_parametrization(),
                                                                           _covariance,
                                                                           newDetectionParameters.get_parametrization(),
                                                                           newDetectionCovariance);
    const utils::PlaneWorldCoordinates newEstimatedParameters(res.first);
    const matrix44& newEstimatedCovariance = res.second;
    const double score = (_parametrization.get_parametrization() - newEstimatedParameters.get_parametrization()).norm();

    // covariance update
    _covariance = newEstimatedCovariance;

    // parameters update
    _parametrization = utils::PlaneWorldCoordinates(newEstimatedParameters);

    // merge the boundary polygon (after optimization) with the observed polygon
    update_boundary_polygon(cameraToWorld, matchedFeature.get_boundary_polygon());

    // static sanity checks
    assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
    assert(not _covariance.hasNaN());
    assert(utils::is_covariance_valid(_covariance));
    assert(not _parametrization.hasNaN());
    return score;
}

void Plane::update_boundary_polygon(const CameraToWorldMatrix& cameraToWorld,
                                    const utils::CameraPolygon& detectedPolygon)
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

void Plane::build_kalman_filter()
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

/**
 *  MapPlane
 */

int MapPlane::find_match(const DetectedPlaneObject& detectedFeatures,
                         const WorldToCameraMatrix& worldToCamera,
                         const vectorb& isDetectedFeatureMatched,
                         std::list<PlaneMatchType>& matches,
                         const bool shouldAddToMatches,
                         const bool useAdvancedSearch) const
{
    const PlaneWorldToCameraMatrix& planeCameraToWorld = utils::compute_plane_world_to_camera_matrix(worldToCamera);
    // project plane in camera space
    const utils::PlaneCameraCoordinates& projectedPlane =
            get_parametrization().to_camera_coordinates(planeCameraToWorld);

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
        const double detectedPlaneArea = shapePlane.get_boundary_polygon().get_area();
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

bool MapPlane::add_to_tracked(const WorldToCameraMatrix& worldToCamera,
                              TrackedPlaneObject& trackedFeatures,
                              const uint dropChance) const
{
    // silence warning for unused parameters
    (void)worldToCamera;
    (void)trackedFeatures;
    (void)dropChance;
    return false;
}

void MapPlane::draw(const WorldToCameraMatrix& worldToCamMatrix, cv::Mat& debugImage, const cv::Scalar& color) const
{
    // display the boundary of the plane
    _boundaryPolygon.display(worldToCamMatrix, color, debugImage);
}

bool MapPlane::is_visible(const WorldToCameraMatrix& worldToCamMatrix) const
{
    return _boundaryPolygon.to_camera_space(worldToCamMatrix).is_visible_in_screen_space();
}

void MapPlane::write_to_file(std::shared_ptr<outputs::IMap_Writer> mapWriter) const
{
    assert(mapWriter != nullptr);
    mapWriter->add_polygon(_boundaryPolygon.get_unprojected_boundary());
}

bool MapPlane::update_with_match(const DetectedPlaneType& matchedFeature,
                                 const matrix33& poseCovariance,
                                 const CameraToWorldMatrix& cameraToWorld)
{
    assert(_matchIndex >= 0);

    // compute projection matrices
    const utils::PlaneCameraCoordinates& matchedFeatureParams = matchedFeature.get_parametrization();
    const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
    const matrix44& planeParameterCovariance =
            utils::compute_plane_covariance(matchedFeatureParams, matchedFeature.get_point_cloud_covariance());

    // project to world coordinates
    const matrix44 worldCovariance = utils::get_world_plane_covariance(
            matchedFeatureParams, planeCameraToWorld, planeParameterCovariance, poseCovariance);
    const utils::PlaneWorldCoordinates& projectedPlaneCoordinates =
            matchedFeatureParams.to_world_coordinates(planeCameraToWorld);

    // update this plane with the other one's parameters
    track(cameraToWorld, matchedFeature, projectedPlaneCoordinates, worldCovariance);
    return true;
}

void MapPlane::update_no_match()
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
    const utils::PlaneCameraCoordinates& detectedFeatureParams = detectedFeature.get_parametrization();
    const PlaneCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);
    const matrix44& planeParameterCovariance =
            utils::compute_plane_covariance(detectedFeatureParams, detectedFeature.get_point_cloud_covariance());

    // set parameters in world coordinates
    _parametrization = detectedFeatureParams.to_world_coordinates(planeCameraToWorld);

    _covariance = utils::get_world_plane_covariance(
            detectedFeatureParams, planeCameraToWorld, planeParameterCovariance, poseCovariance);
    _boundaryPolygon = detectedFeature.get_boundary_polygon().to_world_space(cameraToWorld);

    assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
}

bool StagedMapPlane::should_remove_from_staged() const { return _failedTrackingCount >= 2; }

bool StagedMapPlane::should_add_to_local_map() const { return _successivMatchedCount >= 1; }

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

    assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
    assert(utils::is_covariance_valid(_covariance));
}

bool LocalMapPlane::is_lost() const
{
    const static size_t maximumUnmatchBeforeremoval = Parameters::get_maximum_unmatched_before_removal();
    return _failedTrackingCount >= maximumUnmatchBeforeremoval;
}

} // namespace rgbd_slam::map_management