#ifndef POINTS_TRACKING_HPP
#define POINTS_TRACKING_HPP


#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>

#include "Pose.hpp"
#include "MotionModel.hpp"
#include "Pose_Optimisation.hpp"

namespace poseEstimation {

    class Points_Tracking {
        public:
            Points_Tracking();

            const Pose compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage);

        protected:
            const matched_point_container get_good_matches(const cv::Mat& depthImage, std::vector<cv::KeyPoint>& thisFrameKeypoints, cv::Mat& thisFrameDescriptors); 

            const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status); 

        private:

            cv::Ptr<cv::xfeatures2d::SURF> _featureDetector;
            cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

            std::vector<cv::KeyPoint> _lastFrameKeypoints;
            cv::Mat _lastFrameDescriptors;
            cv::Mat _lastDepthImage;

            Pose _currentPose;
            Motion_Model _motionModel;

    };


}


#endif
