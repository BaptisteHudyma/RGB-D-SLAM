
#include "Pose_Optimisation.hpp"

#include "Point_Tracking.hpp"

#include <iostream>

namespace poseEstimation {


    Points_Tracking::Points_Tracking() {
        //init motion model
        _motionModel.reset();


        // Create feature extractor and matcher
        int minHessian = 400;
        _featureDetector = cv::xfeatures2d::SURF::create( minHessian );
        _featuresMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }



    const Pose Points_Tracking::compute_new_pose (const cv::Mat& grayImage, const cv::Mat& depthImage) {

        std::vector<cv::KeyPoint> frameKeypoints;
        cv::Mat frameDescriptors;

        _featureDetector->detectAndCompute(grayImage, cv:: noArray(), frameKeypoints, frameDescriptors); 


        if (_lastFrameKeypoints.size() <= 0) {
            //first call, or tracking lost
            _lastFrameKeypoints.swap(frameKeypoints);
            _lastFrameDescriptors = frameDescriptors;
            _lastDepthImage = depthImage.clone();
            return _currentPose;
        }

        //get and refine pose
        Pose refinedPose = _motionModel.predict_next_pose(_currentPose);

        const matched_point_container matchedPoints = this->get_good_matches(depthImage, frameKeypoints, frameDescriptors);

        if (matchedPoints.size() > 5) {

            vector3 pos = refinedPose.get_position();
            quaternion orient = refinedPose.get_orientation_quaternion();

            Eigen::VectorXd input(7);
            input[0] = pos.x(); 
            input[1] = pos.y(); 
            input[2] = pos.z();
            input[3] = orient.x();
            input[4] = orient.y();
            input[5] = orient.z();
            input[6] = orient.w();

            Pose_Functor pf(Pose_Estimator(input.size(), matchedPoints));
            Eigen::LevenbergMarquardt<Pose_Functor, double> lm( pf );
            lm.parameters.xtol = 1e-10;
            lm.parameters.ftol = lm.parameters.xtol / 2.0;
            lm.parameters.epsfcn = 0;

            Eigen::LevenbergMarquardtSpace::Status endStatus = lm.minimize(input);
            const std::string message = get_human_readable_end_message(endStatus);
            std::cout << matchedPoints.size() << " pts " << input.transpose() << " in " << lm.iter << " iters. Result " << endStatus << " (" << message << ")" << std::endl;

            refinedPose.set_parameters(vector3(input[0], input[1], input[2]), quaternion(input[3], input[4], input[5], input[6]));
            

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


        _lastFrameKeypoints.swap(frameKeypoints);
        _lastFrameDescriptors = frameDescriptors;
        _lastDepthImage = depthImage.clone();

        _currentPose = refinedPose;

        return refinedPose;
    }




    const matched_point_container
        Points_Tracking::get_good_matches(const cv::Mat& depthImage, std::vector<cv::KeyPoint>& thisFrameKeypoints, cv::Mat& thisFrameDescriptors)
        {
            std::vector< std::vector<cv::DMatch> > knnMatches;
            _featuresMatcher->knnMatch(_lastFrameDescriptors, thisFrameDescriptors, knnMatches, 2);
            cv::Rect rect(cv::Point(), depthImage.size());

            matched_point_container matchedPoints;
            for (size_t i = 0; i < knnMatches.size(); i++)
            {
                const std::vector<cv::DMatch>& match = knnMatches[i];
                //closest point closest by multiplicator than second closest
                if (match[0].distance < 0.9f * match[1].distance)
                {
                    int trainIdx = match[0].trainIdx;
                    int queryIdx = match[0].queryIdx;

                    const cv::Point2f& thisPt = thisFrameKeypoints[queryIdx].pt;
                    const cv::Point2f& lastPt = _lastFrameKeypoints[trainIdx].pt;

                    if (rect.contains(thisPt) and rect.contains(lastPt)) {
                        float depth = depthImage.at<float>(thisPt.x, thisPt.y);
                        float prevDepth = _lastDepthImage.at<float>(lastPt.x, lastPt.y);
                        if (depth > 0 and prevDepth > 0) {

                             point_pair newMatch = std::make_pair(
                                        //for now, do not add depth
                                        vector3(lastPt.x, lastPt.y, prevDepth),
                                        vector3(thisPt.x, thisPt.y, depth)
                                        );
                            matchedPoints.push_back(newMatch);

                        }
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
