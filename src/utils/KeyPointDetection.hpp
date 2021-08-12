#ifndef KEY_POINT_EXTRATION_HPP
#define KEY_POINT_EXTRATION_HPP

#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <list>

#include "types.hpp"
#include "PoseUtils.hpp"

namespace rgbd_slam {
namespace utils {

#define MINIMUM_KEY_POINT_FOR_KNN 6

    /**
     * \brief A class to detect and store keypoints
     *
     */
    class Key_Point_Extraction 
    {
        public:

            /**
              * \param[in] maxMatchDistance Maximum distance to consider that a match of two points is valid
              * \param[in] minHessian Resolution of the point detection
              */
            Key_Point_Extraction(double maxMatchDistance, unsigned int minHessian);

            /**
             * \brief detect and compute the key point matches with the local map
             *
             * \param[in] camPose Pose of the camera in world coordinates
             * \param[in] grayImage The input image from camera
             * \param[in] depthImage The input depth image 
             *
             * \return A container with the matched points (local map and newly detected)
             */
            const matched_point_container detect_and_match_points(const poseEstimation::Pose& camPose, const cv::Mat& grayImage, const cv::Mat& depthImage);

            /**
             * \brief Compute a debug image
             *
             * \param[in] camPose Pose of the camera in world coordinates
             * \param[in, out] debugImage Output image
             */

            void get_debug_image(const poseEstimation::Pose& camPose, cv::Mat& debugImage);

            /**
             * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
             *
             * \param[in] meanFrameTreatmentTime The mean time in seconds that this program used to treat one frame
             * \param[in] frameCount Total of frame treated by the program
             */
            void show_statistics(double meanFrameTreatmentTime, unsigned int frameCount);

        protected:


            /**
             * \brief get a container with the filtered point matches. Uses _maxMatchDistance to estimate good matches. 
             *
             * \param[in] thisFrameKeypoints This frame detected and filtered keypoints
             * \param[in] thisFrameDescriptors The associated descriptors
             *
             * \return The matched points
             */
            const matched_point_container get_good_matches(keypoint_container& thisFrameKeypoints, const cv::Mat& thisFrameDescriptors); 

            /**
             * \brief Get the filtered keypoints. Remove keypoints without depth informations 
             *
             * \param[in] depthImage The associated depth image, already rectified
             * \param[in] kp Container for raw keypoints
             * \param[out] cleanedPoints Container with the final clean keypoints
             */
            void get_cleaned_keypoint(const poseEstimation::Pose& camPose, const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints);

        private:
            cv::Ptr<cv::FeatureDetector> _featureDetector;
            cv::Ptr<cv::DescriptorExtractor> _descriptorExtractor;

            cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

            const double _maxMatchDistance;

            keypoint_container _lastFrameKeypoints;
            cv::Mat _lastFrameDescriptors;

            double _meanPointExtractionTime;

    };

}
}

#endif
