#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

// TODO: remove when map is ready
#include "../features/lines/line_detection.hpp"
// \TODO

#include "outputs/map_writer.hpp"
#include "matches_containers.hpp"
#include "utils/pose.hpp"

#include "map_features/map_point2d.hpp"
#include "map_features/map_point.hpp"
#include "map_features/map_primitive.hpp"

namespace rgbd_slam::map_management {

/**
 * \brief Contains sets of detected features
 */
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
    [[nodiscard]] matches_containers::match_container find_feature_matches(
            const utils::Pose& currentPose, const DetectedFeatureContainer& detectedFeatures) noexcept;

    /**
     * \brief Update the local and global map. Add new points to staged and map container
     *
     * \param[in] optimizedPose The clean true pose of the observer, after optimization
     * \param[in] detectedFeatures An object that contains all the detected features
     * \param[in] outlierMatched A container for all the wrongly associated features detected in the pose
     * optimization process. They should be marked as invalid matches
     */
    void update(const utils::Pose& optimizedPose,
                const DetectedFeatureContainer& detectedFeatures,
                const matches_containers::match_container& outlierMatched);

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
                             const bool addAllFeatures);

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

    void show_statistics(const double meanFrameTreatmentDuration,
                         const uint frameCount,
                         const bool shouldDisplayDetails = false) const noexcept;

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
     * \param[in] outlierMatched A container of the wrong matches detected after the optimization process
     */
    void mark_outliers_as_unmatched(const matches_containers::match_container& outlierMatched) noexcept;

  private:
    size_t _detectedFeatureId; // store the if of the detected feature
    localPoint2DMap _localPoint2DMap;
    localPointMap _localPointMap;
    localPlaneMap _localPlaneMap;

    std::shared_ptr<outputs::IMap_Writer> _mapWriter = nullptr;

    // Remove copy operators
    Local_Map(const Local_Map& map) = delete;
    void operator=(const Local_Map& map) = delete;

    // perf measurments
    double find2DPointMatchDuration = 0.0;
    double findPointMatchDuration = 0.0;
    double findPlaneMatchDuration = 0.0;

    double mapUpdateDuration = 0.0;
    double mapAddFeaturesDuration = 0.0;
};

} // namespace rgbd_slam::map_management

#endif
