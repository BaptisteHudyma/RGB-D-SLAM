#ifndef RGBDSLAM_RGBDSLAM_HPP
#define RGBDSLAM_RGBDSLAM_HPP


#include <opencv2/line_descriptor.hpp>

#include <Eigen/Dense>

#include "features/primitives/depth_map_transformation.hpp"
#include "features/primitives/primitive_detection.hpp"
#include "features/keypoints/keypoint_detection.hpp"

#include "map_management/local_map.hpp"

#include "utils/pose.hpp"
#include "utils/motion_model.hpp"

#include "../third_party/line_segment_detector.hpp"


namespace rgbd_slam {

    class RGBD_SLAM {
        public:
            typedef std::vector<cv::Vec4f> line_vector;

            /**
             * \param[in] startPose the initial pose
             * \param[in] imageWidth The width of the depth images (fixed)
             * \param[in] imageHeight The height of the depth image (fixed)
             */
            explicit RGBD_SLAM(const utils::Pose &startPose, const uint imageWidth = 640, const uint imageHeight = 480);
            ~RGBD_SLAM();

            /**
             * \brief Estimates a new pose from the given images
             *
             * \param[in] inputRgbImage Raw RGB image
             * \param[in] inputDepthImage Raw depth Image, in millimeters
             * \param[in] shouldDetectLines Should we use line detection on the RGB image ?
             *
             * \return The new estimated pose 
             */
            const utils::Pose track(const cv::Mat& inputRgbImage, const cv::Mat& inputDepthImage, const bool shouldDetectLines = false);

            /**
             * \brief Compute a debug image
             *
             * \param[in] camPose Current pose of the observer
             * \param[in] originalRGB Raw rgb image. Will be used as a base for the final image
             * \param[out] debugImage Output image
             * \param[in] elapsedTime Time since the last call (used for FPS count)
             * \param[in] shouldDisplayStagedPoints Display the points that are not map points yet
             * \param[in] shouldDisplayPrimitiveMasks Display the detected primitive masks
             */
            void get_debug_image(const utils::Pose& camPose, const cv::Mat originalRGB, cv::Mat& debugImage, const double elapsedTime, const bool shouldDisplayStagedPoints = false, const bool shouldDisplayPrimitiveMasks = false);

            /**
             * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
             */
            void show_statistics(double meanFrameTreatmentDuration) const;

        protected:

            /**
             * \brief Compute a new pose from the keypoints points between two following images. It uses only the keypoints with an associated depth
             *
             * \param[in, out] grayImage The input image from the camera, as a gray image
             * \param[in] depthImage The associated depth image, already corrected with camera parameters
             * \param[in] cloudArrayOrganized Organized depth image as a connected cloud
             *
             * \return The new estimated pose from points positions
             */
            const utils::Pose compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage, const Eigen::MatrixXf& cloudArrayOrganized);

            void compute_lines(const cv::Mat& grayImage, const cv::Mat& depthImage, cv::Mat& outImage);

            void set_color_vector();

        private:
            const uint _width;
            const uint _height;

            features::primitives::Depth_Map_Transformation* _depthOps;

            size_t _computeKeypointCount;

            /* Detectors */
            features::primitives::Primitive_Detection* _primitiveDetector;
            cv::LSD* _lineDetector;

            map_management::Local_Map* _localMap;
            features::keypoints::Key_Point_Extraction* _pointDetector;

            cv::Mat _kernel;

            utils::Pose _currentPose;
            utils::Motion_Model _motionModel;

            // debug
            uint _totalFrameTreated;
            double _meanDepthMapTreatmentDuration;
            double _meanPoseOptimizationDuration;

            double _meanPrimitiveTreatmentDuration;
            double _meanLineTreatmentDuration;
            double _meanFindMatchTime;
            double _meanPoseOptimizationFromFeatures;
            double _meanLocalMapUpdateDuration;

        private:
            // remove copy constructors as we have dynamically instantiated members
            RGBD_SLAM(const RGBD_SLAM& rgbdSlam) = delete;
            RGBD_SLAM& operator=(const RGBD_SLAM& rgbdSlam) = delete;
    };


}


#endif
