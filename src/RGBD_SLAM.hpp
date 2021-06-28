#ifndef POINTS_TRACKING_HPP
#define POINTS_TRACKING_HPP


#include <opencv2/xfeatures2d.hpp>
#include <opencv2/line_descriptor.hpp>

#include <Eigen/Dense>

#include "Parameters.hpp"
#include "DepthOperations.hpp"

#include "GeodesicOperations.hpp"
#include "PrimitiveDetection.hpp"
#include "PlaneSegment.hpp"
#include "MonocularDepthMap.hpp"
#include "DepthMapSegmentation.hpp"
#include "LineSegmentDetector.hpp"
#include "RGB_Slam.hpp"

#include "Pose.hpp"
#include "MotionModel.hpp"
#include "Pose_Optimisation.hpp"

namespace primitiveDetection {


    class RGBD_SLAM {
        public:
            typedef std::map<unsigned int, poseEstimation::vector3> keypoint_container;
            typedef std::vector<cv::Vec4f> line_vector;
            typedef std::list<std::unique_ptr<primitiveDetection::Primitive>> primitive_container; 

            /**
             * \param[in] dataPath Path of the file that contains the camera intrinsics
             * \param[in] imageWidth The width of the depth images (fixed)
             * \param[in] imageHeight The height of the depth image (fixed)
             * \param[in] minHessian Minimum Hessian parameter for point detection
             * \param[in] maxMatchDistance Maximum distance between the two closest point matches candidates (0 - 1)
             */
            RGBD_SLAM(const std::stringstream& dataPath, unsigned int imageWidth = 640, unsigned int imageHeight = 480, unsigned int minHessian = 300, double maxMatchDistance = 0.7);

            /**
             * \brief Estimates a new pose from the given images
             *
             * \param[in] rgbImage Raw RGB image
             * \param[in] depthImage Raw depth Image
             * \param[in] detectLines Should we use line detection on the RGB image ?
             *
             * \return The new estimated pose 
             */
            poseEstimation::Pose track(const cv::Mat& rgbImage, const cv::Mat& depthImage, bool detectLines = false);

            /**
             * \brief Compute a debug image
             *
             * \param[in] originalRGB Raw rgb image. Will be used as a base for the final image
             * \param[out] debugImage Output image
             * \param[in] elapsedTime Time since the last call (used for FPS count)
             * \param[in] showPrimitiveMasks Display the detected primitive masks
             */
            void get_debug_image(const cv::Mat originalRGB, cv::Mat& debugImage, double elapsedTime, bool showPrimitiveMasks = true);

            /**
             * \brief Shox the time statistics for certain parts of the program. Kind of a basic profiler
             */
            void show_statistics();

        protected:

            /**
             * \brief Compute a new pose from the keypoints points between two following images. It uses only the keypoints with an associated depth
             *
             * \param[in, out] grayImage The input image from the camera, as a gray image
             * \param[in] depthImage The associated depth image, already corrected with camera parameters
             *
             * \return The new estimated pose from points positions
             */
            const poseEstimation::Pose compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage);

            /**
             * \brief get a container with the filtered point matches. Uses _maxMatchDistance to estimate good matches. 
             *
             * \param[in] thisFrameKeypoints This frame detected and filtered keypoints
             * \param[in] thisFrameDescriptors The associated descriptors
             *
             * \return The matched points
             */
            const poseEstimation::matched_point_container get_good_matches(keypoint_container& thisFrameKeypoints, cv::Mat& thisFrameDescriptors); 

            /**
             * \brief Use for debug.
             * \return Returns a string with the human readable version of Eigen LevenbergMarquardt output status
             */
            const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status); 

            /**
             * \brief Get the filtered keypoints. Remove keypoints without depth informations 
             *
             * \param[in] depthImage The associated depth image, already rectified
             * \param[in] kp Container for raw keypoints
             * \param[out] cleanedPoints Container with the final clean keypoints
             */
            void get_cleaned_keypoint(const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints);

            void set_color_vector();

        private:
            const unsigned int _width;
            const unsigned int _height;

            Depth_Operations* _depthOps;

            /* Detectors */
            Primitive_Detection* _primitiveDetector;
            cv::LSD* _lineDetector;


            cv::Mat _kernel;

            const double _maxMatchDistance;

            //keep track of the primitives tracked last frame
            primitive_container _previousFramePrimitives;

            cv::Ptr<cv::xfeatures2d::SURF> _featureDetector;
            cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

            std::map<int, int> _previousAssociatedIds;

            keypoint_container _lastFrameKeypoints;
            cv::Mat _lastFrameDescriptors;

            poseEstimation::Pose _currentPose;
            poseEstimation::Motion_Model _motionModel;

            // display
            cv::Mat_<uchar> _segmentationOutput;
            std::vector<cv::Vec3b> _colorCodes;

            // debug
            unsigned int _totalFrameTreated;
            double _maxTreatTime;
            double _meanMatTreatmentTime;
            double _meanTreatmentTime;
            double _meanLineTreatment;
            double _meanPoseTreatmentTime;
    };


}


#endif
