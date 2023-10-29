#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

#include "../features/keypoints/keypoint_handler.hpp"
#include "../features/primitives/shape_primitives.hpp"
#include "../features/lines/line_detection.hpp"
#include "../outputs/map_writer.hpp"
#include "../matches_containers.hpp"
#include "../utils/pose.hpp"
#include "coordinates.hpp"
#include "feature_map.hpp"
#include "map_point2d.hpp"
#include "map_point.hpp"
#include "map_primitive.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

namespace rgbd_slam::map_management {

struct DetectedFeatureContainer
{
    DetectedFeatureContainer(const features::keypoints::Keypoint_Handler& newKeypointObject,
                             const features::lines::line_container& newdDetectedLines,
                             const features::primitives::plane_container& newDetectedPlanes) :
        keypointObject(newKeypointObject),
        detectedLines(newdDetectedLines),
        detectedPlanes(newDetectedPlanes),
        id(++idAllocator)
    {
    }

    const features::keypoints::Keypoint_Handler keypointObject;
    const features::lines::line_container detectedLines;
    const features::primitives::plane_container detectedPlanes;
    const size_t id; // unique id to differenciate from other detections

  private:
    inline static size_t idAllocator = 0;
};

/**
 * \brief Maintain a local (around the camera) map.
 * Handle the feature association and tracking in local space.
 * Can return matched features, and update the global map when features are estimated to be reliable.
 */
class Local_Map
{
  public:
    Local_Map();
    ~Local_Map();

    /**
     * \brief Return an object containing the tracked keypoint features in screen space (2D), with the associated global
     * ids
     * \param[in] lastpose The last known pose of the observer
     */
    [[nodiscard]] features::keypoints::KeypointsWithIdStruct get_tracked_keypoints_features(
            const utils::Pose& lastpose) const noexcept;

    /**
     * \brief Find all matches for the given detected features
     * \param[in] currentPose The pose of the observer
     * \param[in] detectedFeatures An object that contains the detected features
     */
    [[nodiscard]] matches_containers::matchContainer find_feature_matches(
            const utils::Pose& currentPose, const DetectedFeatureContainer& detectedFeatures) noexcept;

    /**
     * \brief Update the local and global map. Add new points to staged and map container
     *
     * \param[in] optimizedPose The clean true pose of the observer, after optimization
     * \param[in] detectedFeatures An object that contains all the detected features
     * \param[in] outlierMatchedPoints A container for all the wrongly associated points detected in the pose
     * optimization process. They should be marked as invalid matches
     * \param[in] outlierMatchedPlanes A container for all the wrongly associated planes detected in the pose
     * optimization process. They should be marked as invalid matches
     */
    void update(const utils::Pose& optimizedPose,
                const DetectedFeatureContainer& detectedFeatures,
                const matches_containers::match_point_container& outlierMatchedPoints,
                const matches_containers::match_plane_container& outlierMatchedPlanes) noexcept;

    /**
     * \brief Update the local map when no pose could be estimated. Consider all features as unmatched
     */
    void update_no_pose() noexcept;

    /**
     * \brief Add features to staged map
     * \param[in] poseCovariance The pose covariance of the observer, after optimization
     * \param[in] cameraToWorld The matrix to go from camera to world space
     * \param[in] detectedFeatures Contains the detected features
     * \param[in] addAllFeatures If false, will add all non matched features, if true, add all features regardless of
     * the match status
     */
    void add_features_to_map(const matrix33& poseCovariance,
                             const CameraToWorldMatrix& cameraToWorld,
                             const DetectedFeatureContainer& detectedFeatures,
                             const bool addAllFeatures) noexcept;

    /**
     * \brief Hard clean the local and staged map
     */
    void reset() noexcept;

    /**
     * \brief Compute a debug image to display the keypoints & planes
     *
     * \param[in] camPose Pose of the camera in world coordinates
     * \param[in] shouldDisplayStaged If true, will also display the content of the staged keypoint map
     * \param[in] shouldDisplayPlaneMasks If true, will also display the planes in local map
     * \param[in, out] debugImage Output image
     */
    void get_debug_image(const utils::Pose& camPose,
                         const bool shouldDisplayStaged,
                         const bool shouldDisplayPlaneMasks,
                         cv::Mat& debugImage) const noexcept;

  protected:
    /**
     * \brief Clean the local map so it stays local, and update the global map with the good features
     */
    void update_local_to_global() noexcept;

    /**
     * \brief draw the top information band on the debug image
     */
    void draw_image_head_band(cv::Mat& debugImage) const noexcept;

    /**
     * \brief Mark all the outliers detected during optimization as unmatched
     * \param[in] outlierMatchedPoints A container of the wrong matches detected after the optimization process
     */
    void mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints) noexcept;
    /**
     * \brief Mark all the outliers detected during optimization as unmatched
     * \param[in] outlierMatchedPlanes A container of the wrong matches detected after the optimization process
     */
    void mark_outliers_as_unmatched(const matches_containers::match_plane_container& outlierMatchedPlanes) noexcept;

  private:
    // Define types
    using localPoint2DMap = Feature_Map<LocalMapPoint2D,
                                        StagedMapPoint2D,
                                        DetectedKeypointsObject,
                                        DetectedPoint2DType,
                                        PointMatch2DType,
                                        TrackedPointsObject,
                                        UpgradedPoint2DType>;
    using localPointMap = Feature_Map<LocalMapPoint,
                                      StagedMapPoint,
                                      DetectedKeypointsObject,
                                      DetectedPointType,
                                      PointMatchType,
                                      TrackedPointsObject,
                                      UpgradedPointType>;
    using localPlaneMap = Feature_Map<LocalMapPlane,
                                      StagedMapPlane,
                                      DetectedPlaneObject,
                                      DetectedPlaneType,
                                      PlaneMatchType,
                                      TrackedPlaneObject,
                                      UpgradedPlaneType>;

    size_t _detectedFeatureId; // store the if of the detected feature
    localPoint2DMap _localPoint2DMap;
    localPointMap _localPointMap;
    localPlaneMap _localPlaneMap;

    // local shape plane map container
    using plane_map_container = std::unordered_map<size_t, MapPlane>;

    std::shared_ptr<outputs::IMap_Writer> _mapWriter = nullptr;

    // Remove copy operators
    Local_Map(const Local_Map& map) = delete;
    void operator=(const Local_Map& map) = delete;
};

} // namespace rgbd_slam::map_management

#endif
