#ifndef POINTS_TRACKING_HPP
#define POINTS_TRACKING_HPP


#include <opencv2/line_descriptor.hpp>

#include <Eigen/Dense>

#include "DepthOperations.hpp"

#include "GeodesicOperations.hpp"
#include "PrimitiveDetection.hpp"
#include "PlaneSegment.hpp"
#include "LineSegmentDetector.hpp"
#include "KeyPointDetection.hpp"
#include "local_map.hpp"

#include "Pose.hpp"
#include "MotionModel.hpp"
#include "Pose_Optimisation.hpp"

namespace rgbd_slam {

    class RGBD_SLAM {
        public:
            typedef std::vector<cv::Vec4f> line_vector;
            typedef std::list<std::unique_ptr<primitiveDetection::Primitive>> primitive_container; 

            /**
             * \param[in] dataPath Path of the file that contains the camera intrinsics
             * \param[in] imageWidth The width of the depth images (fixed)
             * \param[in] imageHeight The height of the depth image (fixed)
             * \param[in] minHessian Minimum Hessian parameter for point detection
             * \param[in] maxMatchDistance Maximum distance between the two closest point matches candidates (0 - 1)
             */
            RGBD_SLAM(const std::stringstream& dataPath, unsigned int imageWidth = 640, unsigned int imageHeight = 480);

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
            void get_debug_image(const poseEstimation::Pose& camPose, const cv::Mat originalRGB, cv::Mat& debugImage, double elapsedTime, bool showPrimitiveMasks = true);

            /**
             * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
             */
            void show_statistics(double frameMeanTreatmentTime);

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

            void set_color_vector();

        private:
            const unsigned int _width;
            const unsigned int _height;

            primitiveDetection::Depth_Operations* _depthOps;

            /* Detectors */
            primitiveDetection::Primitive_Detection* _primitiveDetector;
            cv::LSD* _lineDetector;

            map_management::Local_Map* _localMap;
            utils::Key_Point_Extraction* _pointMatcher;

            cv::Mat _kernel;

            double _meanPoseOptimisationIterations;


            //keep track of the primitives tracked last frame
            primitive_container _previousFramePrimitives;

            std::map<int, int> _previousAssociatedIds;

            poseEstimation::Pose _currentPose;
            poseEstimation::Motion_Model _motionModel;

            // display
            cv::Mat_<uchar> _segmentationOutput;
            std::vector<cv::Vec3b> _colorCodes;

            // debug
            unsigned int _totalFrameTreated;
            double _meanMatTreatmentTime;
            double _meanTreatmentTime;
            double _meanLineTreatment;
            double _meanPoseTreatmentTime;
    };


}


#endif
