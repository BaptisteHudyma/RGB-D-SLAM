#ifndef POINTS_TRACKING_HPP
#define POINTS_TRACKING_HPP


#include <opencv2/xfeatures2d.hpp>

#include <Eigen/Dense>

#include "Pose.hpp"
#include "MotionModel.hpp"
#include "Pose_Optimisation.hpp"

namespace poseEstimation {

    typedef std::map<unsigned int, vector3> keypoint_container;

    class Points_Tracking {
        public:
            Points_Tracking(int minHessian);

            const Pose compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage);

        protected:
            const matched_point_container get_good_matches(keypoint_container& thisFrameKeypoints, cv::Mat& thisFrameDescriptors); 

            const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status); 

            void get_cleaned_keypoint(const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints);

        private:
            
            cv::Ptr<cv::xfeatures2d::SURF> _featureDetector;
            cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

            keypoint_container _lastFrameKeypoints;
            cv::Mat _lastFrameDescriptors;

            Pose _currentPose;
            Motion_Model _motionModel;

    };


}


#endif
