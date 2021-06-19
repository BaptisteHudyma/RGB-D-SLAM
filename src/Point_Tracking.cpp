
#include "Pose_Optimisation.hpp"

#include "Point_Tracking.hpp"

#include <iostream>

namespace poseEstimation {


    Points_Tracking::Points_Tracking(int minHessian) 
    {
        //init motion model
        _motionModel.reset();


        // Create feature extractor and matcher
        _featureDetector = cv::xfeatures2d::SURF::create( minHessian );
        _featuresMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    void Points_Tracking::get_cleaned_keypoint(const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints) 
    {
        cv::Rect lastRect(cv::Point(), depthImage.size());
        unsigned int i = 0;
        for (const cv::KeyPoint& keypoint : kp) {
            const cv::Point2f& pt = keypoint.pt;
            if (lastRect.contains(pt)) {
                const float depth = depthImage.at<float>(pt.y, pt.x);
                if (depth > 0) {
                    cleanedPoints.emplace( i, vector3(pt.x, pt.y, depth) );
                }
            }
            ++i;
        }
    }


    const Pose Points_Tracking::compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage) 
    {
        std::vector<cv::KeyPoint> frameKeypoints;
        cv::Mat frameDescriptors;

        _featureDetector->detectAndCompute(grayImage, cv:: noArray(), frameKeypoints, frameDescriptors); 

        keypoint_container cleanedKp;
        get_cleaned_keypoint(depthImage, frameKeypoints, cleanedKp);

        if (_lastFrameKeypoints.size() <= 0) {
            //first call, or tracking lost
            _lastFrameKeypoints.swap(cleanedKp);
            _lastFrameDescriptors = frameDescriptors;
            return _currentPose;
        }
        else if (frameKeypoints.size() <= 2) {
            std::cout << "Not enough features detected for knn matching" << std::endl;
            return _currentPose;
        }

        //get and refine pose
        Pose refinedPose = _motionModel.predict_next_pose(_currentPose);

        const matched_point_container matchedPoints = this->get_good_matches(cleanedKp, frameDescriptors);

        if (matchedPoints.size() > 5) {

            vector3 pos = refinedPose.get_position();
            quaternion orient = refinedPose.get_orientation_quaternion();

            Eigen::VectorXd input(7);
            input[0] = 0;
            input[1] = 0;
            input[2] = 0;
            input[3] = 1;
            input[4] = 0;
            input[5] = 0;
            input[6] = 0;

            Pose_Functor pf(Pose_Estimator(input.size(), matchedPoints));
            Eigen::LevenbergMarquardt<Pose_Functor, double> lm( pf );
            lm.parameters.epsfcn = 1e-5;
            lm.parameters.maxfev= 1000;

            Eigen::LevenbergMarquardtSpace::Status endStatus = lm.minimize(input);
            const std::string message = get_human_readable_end_message(endStatus);

            quaternion endRotation(input[3], input[4], input[5], input[6]);
            endRotation.normalize();
            vector3 endTranslation(input[0], input[1], input[2]);

            refinedPose.update(endTranslation, endRotation);

            std::cout << matchedPoints.size() << " pts " << endTranslation.transpose() << " | " << endRotation.coeffs().transpose() << " in " << lm.iter << " iters. Result " << endStatus << " (" << message << ")" << std::endl;

        }
        else
            std::cout << "Not enough points for pose estimation" << std::endl;

        /*
           for (const cv::KeyPoint& pt : frameKeypoints) {
           cv::circle(segRgb, cv::Point(pt.pt.x, pt.pt.y), 0, cv::Scalar(0, 255, 255), 1);
           }
         */


        //update motion model with refined pose
        _motionModel.update_model(refinedPose);


        _lastFrameKeypoints.swap(cleanedKp);
        _lastFrameDescriptors = frameDescriptors;

        _currentPose = refinedPose;

        return refinedPose;
    }

    const matched_point_container
        Points_Tracking::get_good_matches(keypoint_container& thisFrameKeypoints, cv::Mat& thisFrameDescriptors)
        {
            std::vector< std::vector<cv::DMatch> > knnMatches;
            _featuresMatcher->knnMatch(_lastFrameDescriptors, thisFrameDescriptors, knnMatches, 2);

            matched_point_container matchedPoints;
            for (size_t i = 0; i < knnMatches.size(); i++)
            {
                const std::vector<cv::DMatch>& match = knnMatches[i];
                //closest point closest by multiplier than second closest
                if (match[0].distance < 0.9f * match[1].distance)
                {
                    int trainIdx = match[0].trainIdx;
                    int queryIdx = match[0].queryIdx;

                    if (_lastFrameKeypoints.contains(trainIdx) and thisFrameKeypoints.contains(queryIdx)) {
                        point_pair newMatch = std::make_pair(
                                _lastFrameKeypoints[trainIdx],
                                thisFrameKeypoints[queryIdx]
                                );
                        matchedPoints.push_back(newMatch);
                    }
                }
            }
            return matchedPoints;
        }


    const std::string Points_Tracking::get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status) {
        switch(status) {
            case Eigen::LevenbergMarquardtSpace::Status::NotStarted :
                return "not started";
            case Eigen::LevenbergMarquardtSpace::Status::Running :
                return "running";
            case Eigen::LevenbergMarquardtSpace::Status::ImproperInputParameters :
                return "improper input parameters";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall :
                return "relative reduction too small";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall :
                return "relative error too small";
            case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall :
                return "relative error and reduction too small";
            case Eigen::LevenbergMarquardtSpace::Status::CosinusTooSmall :
                return "cosinus too small";
            case Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation :
                return "too many function evaluation";
            case Eigen::LevenbergMarquardtSpace::Status::FtolTooSmall :
                return "xtol too small";
            case Eigen::LevenbergMarquardtSpace::Status::XtolTooSmall :
                return "ftol too small";
            case Eigen::LevenbergMarquardtSpace::Status::GtolTooSmall :
                return "gtol too small";
            case Eigen::LevenbergMarquardtSpace::Status::UserAsked :
                return "user asked";
            default:
                return "error: empty message";
        }
        return std::string("");
    }


} /* namespace poseEstimation */
