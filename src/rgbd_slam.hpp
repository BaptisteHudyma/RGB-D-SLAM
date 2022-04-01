#ifndef RGBDSLAM_RGBDSLAM_HPP
#define RGBDSLAM_RGBDSLAM_HPP


#include <opencv2/line_descriptor.hpp>

#include <Eigen/Dense>

#include "depth_map_transformation.hpp"

#include "geodesic_operations.hpp"
#include "primitive_detection.hpp"
#include "line_segment_detector.hpp"
#include "keypoint_detection.hpp"
#include "local_map.hpp"

#include "pose.hpp"
#include "motion_model.hpp"

namespace rgbd_slam {

    class RGBD_SLAM {
        public:
            typedef std::vector<cv::Vec4f> line_vector;
            typedef std::list<features::primitives::primitive_uniq_ptr> primitive_container; 

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
             * \param[in] inputDepthImage Raw depth Image
             * \param[in] detectLines Should we use line detection on the RGB image ?
             *
             * \return The new estimated pose 
             */
            const utils::Pose track(const cv::Mat& inputRgbImage, const cv::Mat& inputDepthImage, const bool detectLines = false);

            /**
             * \brief Compute a debug image
             *
             * \param[in] camPose Current pose of the observer
             * \param[in] originalRGB Raw rgb image. Will be used as a base for the final image
             * \param[out] debugImage Output image
             * \param[in] elapsedTime Time since the last call (used for FPS count)
             * \param[in] showPrimitiveMasks Display the detected primitive masks
             */
            void get_debug_image(const utils::Pose& camPose, const cv::Mat originalRGB, cv::Mat& debugImage, const double elapsedTime, const bool showStagedPoints = false, const bool showPrimitiveMasks = false);

            /**
             * \brief Show the time statistics for certain parts of the program. Kind of a basic profiler
             */
            void show_statistics(double meanFrameTreatmentTime) const;

        protected:

            /**
             * \brief Compute a new pose from the keypoints points between two following images. It uses only the keypoints with an associated depth
             *
             * \param[in, out] grayImage The input image from the camera, as a gray image
             * \param[in] depthImage The associated depth image, already corrected with camera parameters
             *
             * \return The new estimated pose from points positions
             */
            const utils::Pose compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage);

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
            features::keypoints::Key_Point_Extraction* _pointMatcher;

            cv::Mat _kernel;


            //keep track of the primitives tracked last frame
            primitive_container _previousFramePrimitives;

            std::unordered_map<int, uint> _previousAssociatedIds;

            utils::Pose _currentPose;
            utils::Motion_Model _motionModel;

            // display
            cv::Mat_<uchar> _segmentationOutput;
            std::vector<cv::Vec3b> _colorCodes;

            // debug
            uint _totalFrameTreated;
            double _meanMatTreatmentTime;
            double _meanTreatmentTime;
            double _meanLineTreatment;
            double _meanPoseTreatmentTime;

        private:
            // remove copy constructors as we have dynamically instantiated members
            RGBD_SLAM(const RGBD_SLAM& rgbdSlam) = delete;
            RGBD_SLAM& operator=(const RGBD_SLAM& rgbdSlam) = delete;
    };


}


#endif
