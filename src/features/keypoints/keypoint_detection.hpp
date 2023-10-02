#ifndef RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_DETECTION_HPP
#define RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_DETECTION_HPP

#include "keypoint_handler.hpp"

namespace rgbd_slam::features::keypoints {

/**
 * \brief A class to detect and track keypoints.
 * This class will detect keypoints in gray images, compute descriptors and track those points using optical
 * flow and/or descriptor matching
 */
class Key_Point_Extraction
{
  public:
    Key_Point_Extraction();

    /**
     * \brief compute the keypoints in the gray image, using optical flow and/or generic feature detectors
     *
     * \param[in] grayImage The input image from camera
     * \param[in] depthImage The input depth image from camera
     * \param[in] lastKeypointsWithIds The keypoints of the previous detection step, that will be tracked
     * with optical flow \param[in] forceKeypointDetection Force the detection of keypoints in the image
     *
     * \return An object that contains the detected keypoints
     */
    [[nodiscard]] Keypoint_Handler compute_keypoints(const cv::Mat& grayImage,
                                                     const cv::Mat_<float>& depthImage,
                                                     const KeypointsWithIdStruct& lastKeypointsWithIds,
                                                     const bool forceKeypointDetection = false) noexcept;

    /**
     * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
     *
     * \param[in] meanFrameTreatmentDuration The mean duration in seconds that this program used to treat
     * one frame \param[in] frameCount Total of frame treated by the program
     */
    void show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const noexcept;

  protected:
    /**
     * \brief Compute the current frame keypoints from optical flow. There is need for matching with this
     * configuration
     *
     * \param[in] imagePreviousPyramide The pyramid representation of the previous image
     * \param[in] imageCurrentPyramide The pyramid representation of the current image to analyze
     * \param[in] lastKeypointsWithIds The keypoints detected in imagePrevious
     * \param[in] pyramidDepth The chosen depth of the image pyramids
     * \param[in] windowSize The chosen size of the optical flow window
     * \param[in] maxDistanceThreshold a distance threshold, in pixels
     * \param[out] keypointStruct The keypoints tracked by optical flow
     */
    static void get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide,
                                                const std::vector<cv::Mat>& imageCurrentPyramide,
                                                const KeypointsWithIdStruct& lastKeypointsWithIds,
                                                const uint pyramidDepth,
                                                const uint windowSize,
                                                const double maxDistanceThreshold,
                                                KeypointsWithIdStruct& keypointStruct) noexcept;

    /**
     * \brief Compute new key point, with an optional mask to exclude detection zones
     * \param[in] grayImage The image in which we want to detect waypoints
     * \param[in] mask The mask which we do not want to detect waypoints
     * \param[in] minimumPointsForValidity The minimum number of points under which we will use the precise
     * detector \return An array of points in the input image
     */
    [[nodiscard]] std::vector<cv::Point2f> detect_keypoints(const cv::Mat& grayImage,
                                                            const cv::Mat_<uchar>& mask,
                                                            const uint minimumPointsForValidity) const noexcept;

    /**
     * \brief Perform keypoint detection on the image, divided in smaller patches.
     * \param[in] grayImage Image in which to detect keypoints
     * \param[in] mask Mask of the image in which to detect keypoints. No keypoints will be detected in this
     * area \param[in] featureDetector The feature detector to use \param[out] frameKeypoints the detected
     * keypoints in the image
     */
    void perform_keypoint_detection(const cv::Mat& grayImage,
                                    const cv::Mat_<uchar>& mask,
                                    const cv::Ptr<cv::FeatureDetector>& featureDetector,
                                    std::vector<cv::KeyPoint>& frameKeypoints) const noexcept;

    [[nodiscard]] cv::Mat_<uchar> compute_key_point_mask(
            const cv::Size imageSize, const std::vector<cv::Point2f>& keypointContainer) const noexcept;

  private:
    cv::Ptr<cv::FeatureDetector> _featureDetector;
    cv::Ptr<cv::FeatureDetector> _advancedFeatureDetector;
    std::vector<cv::Rect> _detectionWindows;

    std::vector<cv::Mat> _lastFramePyramide;

    double _meanPointExtractionDuration;
};

} // namespace rgbd_slam::features::keypoints

#endif
