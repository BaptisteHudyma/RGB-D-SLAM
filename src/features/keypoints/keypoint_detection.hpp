#ifndef RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_DETECTION_HPP
#define RGBDSLAM_FEATURES_KEYPOINTS_KEYPOINTS_DETECTION_HPP

#include "keypoint_handler.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            /**
             * \brief A class to detect and track keypoints.
             * This class will detect keypoints in gray images, compute descriptors and track those points using optical flow and/or descriptor matching
             */
            class Key_Point_Extraction 
            {
                public:

                    /**
                     * \param[in] minHessian The "precision" of the keypoint descriptor
                     */
                    Key_Point_Extraction(const uint minHessian = 25);

                    /**
                     * \brief compute the keypoints in the gray image, using optical flow and/or generic feature detectors 
                     *
                     * \param[in] grayImage The input image from camera
                     * \param[in] depthImage The input depth image from camera
                     * \param[in] lastKeypointsWithIds The keypoints of the previous detection step, that will be tracked with optical flow
                     * \param[in] forceKeypointDetection Force the detection of keypoints in the image
                     *
                     * \return An object that contains the detected keypoints
                     */
                    const Keypoint_Handler compute_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage, const KeypointsWithIdStruct& lastKeypointsWithIds, const bool forceKeypointDetection = false);


                    /**
                     * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
                     *
                     * \param[in] meanFrameTreatmentDuration The mean duration in seconds that this program used to treat one frame
                     * \param[in] frameCount Total of frame treated by the program
                     */
                    void show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const;

                protected:

                    /**
                     * \brief Compute the current frame keypoints from optical flow. There is need for matching with this configuration
                     *
                     * \param[in] imagePreviousPyramide The pyramid representation of the previous image
                     * \param[in] imageCurrentPyramide The pyramid representation of the current image to analyze
                     * \param[in] lastKeypointsWithIds The keypoints detected in imagePrevious
                     * \param[in] pyramidDepth The chosen depth of the image pyramids
                     * \param[in] windowSize The chosen size of the optical flow window 
                     * \param[in] errorThreshold an error Threshold for optical flow, in pixels
                     * \param[in] maxDistanceThreshold a distance threshold, in pixels
                     * \param[out] keypointStruct The keypoints tracked by optical flow
                     */
                    static void get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide, const std::vector<cv::Mat>& imageCurrentPyramide, const KeypointsWithIdStruct& lastKeypointsWithIds, const uint pyramidDepth, const uint windowSize, const double errorThreshold, const double maxDistanceThreshold, KeypointsWithIdStruct& keypointStruct);

                    /**
                     * \brief Compute new key point, with an optional mask to exclude detection zones
                     *
                     * \param[in] grayImage The image in which we want to detect waypoints
                     * \param[in] mask The mask which we do not want to detect waypoints
                     * \param[in] minimumPointsForValidity The minimum number of points under which we will use the precise detector
                     *
                     * \return An array of points in the input image
                     */
                    const std::vector<cv::Point2f> detect_keypoints(const cv::Mat& grayImage, const cv::Mat& mask, const uint minimumPointsForValidity) const;

                    const cv::Mat compute_key_point_mask(const cv::Size imageSize, const std::vector<cv::Point2f>& keypointContainer) const;

                private:
                    cv::Ptr<cv::FeatureDetector> _featureDetector;
                    cv::Ptr<cv::FeatureDetector> _advancedFeatureDetector;
                    cv::Ptr<cv::DescriptorExtractor> _descriptorExtractor;

                    std::vector<cv::Mat> _lastFramePyramide;

                    double _meanPointExtractionDuration;

            };

        }
    }
}

#endif
