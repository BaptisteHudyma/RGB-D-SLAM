#ifndef RGBDSLAM_RGBDSLAM_HPP
#define RGBDSLAM_RGBDSLAM_HPP

#include "features/keypoints/keypoint_detection.hpp"
#include "features/lines/line_detection.hpp"
#include "features/primitives/depth_map_transformation.hpp"
#include "features/primitives/primitive_detection.hpp"
#include "map_management/local_map.hpp"
#include "tracking/motion_model.hpp"
#include "utils/pose.hpp"
#include <memory>
#include <opencv2/line_descriptor.hpp>

namespace rgbd_slam {

/**
 * \brief Main Simultaneous localisation and tracking class.
 * This is the input of the program, the only needed interaction from a user view point.
 */
class RGBD_SLAM
{
  public:
    /**
     * \param[in] startPose the initial pose
     * \param[in] imageWidth The width of the depth images (fixed)
     * \param[in] imageHeight The height of the depth image (fixed)
     */
    RGBD_SLAM(const utils::Pose& startPose, const uint imageWidth = 640, const uint imageHeight = 480);

    /**
     * \brief Convert the given depth image to the rectified version. IE: align it with the RGB image
     * \param[in, out] depthImage the distorded depth image
     */
    void rectify_depth(cv::Mat_<float>& depthImage) noexcept;

    /**
     * \brief Estimates a new pose from the given images
     *
     * \param[in] inputRgbImage Raw RGB image
     * \param[in] inputDepthImage Raw depth Image, in millimeters
     *
     * \return The new estimated pose
     */
    [[nodiscard]] utils::Pose track(const cv::Mat& inputRgbImage, const cv::Mat_<float>& inputDepthImage) noexcept;

    /**
     * \brief Compute a debug image
     *
     * \param[in] camPose Current pose of the observer
     * \param[in] originalRGB Raw rgb image. Will be used as a base for the final image
     * \param[in] elapsedTime Time since the last call (used for FPS count)
     * \param[in] shouldDisplayStagedFeatures Display the features that are not map features yet
     */
    [[nodiscard]] cv::Mat get_debug_image(const utils::Pose& camPose,
                                          const cv::Mat& originalRGB,
                                          const double elapsedTime,
                                          const bool shouldDisplayStagedFeatures = false) const noexcept;

    /**
     * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
     */
    void show_statistics(double meanFrameTreatmentDuration) const noexcept;

  protected:
    [[nodiscard]] map_management::DetectedFeatureContainer detect_features(const utils::Pose& predictedPose,
                                                                           const cv::Mat& grayImage,
                                                                           const cv::Mat_<float>& depthImage,
                                                                           const matrixf& cloudArrayOrganized) noexcept;

    /**
     * \brief Compute a new pose from the keypoints points between two following images. It uses only the keypoints with
     * an associated depth
     *
     * \param[in] grayImage The input image from the camera, as a gray image
     * \param[in] depthImage The associated depth image, already corrected with camera parameters
     * \param[in] cloudArrayOrganized Organized depth image as a connected cloud
     *
     * \return The new estimated pose from points positions
     */
    [[nodiscard]] utils::Pose compute_new_pose(const cv::Mat& grayImage,
                                               const cv::Mat_<float>& depthImage,
                                               const matrixf& cloudArrayOrganized) noexcept;

    void compute_lines(const cv::Mat& grayImage, const cv::Mat_<float>& depthImage, cv::Mat& outImage) noexcept;

    void set_color_vector() noexcept;

  private:
    const uint _width;
    const uint _height;

    std::unique_ptr<features::primitives::Depth_Map_Transformation> _depthOps = nullptr;

    size_t _computeKeypointCount;

    /* Detectors */
    std::unique_ptr<features::primitives::Primitive_Detection> _primitiveDetector = nullptr;

    std::unique_ptr<map_management::Local_Map> _localMap = nullptr;
    std::unique_ptr<features::keypoints::Key_Point_Extraction> _pointDetector = nullptr;
    std::unique_ptr<features::lines::Line_Detection> _lineDetector = nullptr;

    utils::Pose _currentPose;
    tracking::Motion_Model _motionModel;

    bool _isTrackingLost;      // True is the tracking of last frame failed
    uint _failedTrackingCount; // number of consecutive lost tracking

    bool _isFirstTrackingCall = true; // first call to the tracking function (prevent erroneous error messages)

    // debug
    uint _totalFrameTreated = 0;
    double _meanDepthMapTreatmentDuration = 0;

    // remove copy constructors as we have dynamically instantiated members
    RGBD_SLAM(const RGBD_SLAM& rgbdSlam) = delete;
    RGBD_SLAM& operator=(const RGBD_SLAM& rgbdSlam) = delete;
};

} // namespace rgbd_slam

#endif
