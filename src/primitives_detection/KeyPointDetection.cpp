#include "KeyPointDetection.hpp"

// circle
#include <opencv2/opencv.hpp>

// cout cerr
#include <iostream>


#define CLASS_ERR "<Key_Point_Extraction> "


Key_Point_Extraction::Key_Point_Extraction(double maxMatchDistance, unsigned int minHessian) :
    _maxMatchDistance(maxMatchDistance)

{
    if (_maxMatchDistance <= 0 or _maxMatchDistance > 1) {
        std::cerr << CLASS_ERR << "Maximum matching distance disparity must be between 0 and 1" << std::endl;
        exit(-1);
    }

    // Create feature extractor and matcher
    _featureDetector = cv::xfeatures2d::SURF::create( minHessian );
    _featuresMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    //profiling
    _meanPointExtractionTime = 0.0;
}

const matched_point_container Key_Point_Extraction::detect_and_match_points(const cv::Mat& grayImage, const cv::Mat& depthImage) 
{
    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;

    double t1 = cv::getTickCount();
    _featureDetector->detectAndCompute(grayImage, cv:: noArray(), frameKeypoints, frameDescriptors); 
    _meanPointExtractionTime += (cv::getTickCount() - t1) / (double)cv::getTickFrequency();

    keypoint_container cleanedKp;
    get_cleaned_keypoint(depthImage, frameKeypoints, cleanedKp);

    if (_lastFrameKeypoints.size() <= 0) {
        //first call, or tracking lost
        _lastFrameKeypoints.swap(cleanedKp);
        _lastFrameDescriptors = frameDescriptors;

        matched_point_container matchedPoints;
        return matchedPoints;
    }
    else if (frameKeypoints.size() <= 3) {
        std::cout << "Not enough features detected for knn matching" << std::endl;

        matched_point_container matchedPoints;
        return matchedPoints;
    }

    // keep track of last frame points
    _lastFrameKeypoints.swap(cleanedKp);
    _lastFrameDescriptors = frameDescriptors;

    // return good matches
    return this->get_good_matches(cleanedKp, frameDescriptors);
}


void Key_Point_Extraction::get_cleaned_keypoint(const cv::Mat& depthImage, const std::vector<cv::KeyPoint>& kp, keypoint_container& cleanedPoints) 
{
    cv::Rect lastRect(cv::Point(), depthImage.size());
    unsigned int i = 0;
    for (const cv::KeyPoint& keypoint : kp) {
        const cv::Point2f& pt = keypoint.pt;
        if (lastRect.contains(pt)) {
            const float depth = depthImage.at<float>(pt.y, pt.x);
            if (depth > 0) {
                cleanedPoints.emplace( i, poseEstimation::vector3(pt.x, pt.y, depth) );
            }
            else {
                cleanedPoints.emplace( i, poseEstimation::vector3(pt.x, pt.y, 0) );
            }
        }
        ++i;
    }
}

const matched_point_container Key_Point_Extraction::get_good_matches(keypoint_container& thisFrameKeypoints, cv::Mat& thisFrameDescriptors)
{
    std::vector< std::vector<cv::DMatch> > knnMatches;
    _featuresMatcher->knnMatch(_lastFrameDescriptors, thisFrameDescriptors, knnMatches, 2);

    matched_point_container matchedPoints;
    for (size_t i = 0; i < knnMatches.size(); i++)
    {
        const std::vector<cv::DMatch>& match = knnMatches[i];
        //check if point is a good match by checking it's distance to the second best matched point
        if (match[0].distance < _maxMatchDistance * match[1].distance)
        {
            int trainIdx = match[0].trainIdx;   //last frame key point
            int queryIdx = match[0].queryIdx;   //this frame key point

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



void Key_Point_Extraction::get_debug_image(cv::Mat& debugImage) 
{
    if (_lastFrameKeypoints.size() > 0) {
        for (const std::pair<unsigned int, poseEstimation::vector3> pair : _lastFrameKeypoints) {
            const poseEstimation::vector3 point = pair.second;
            if (point.z() <= 0) {
                cv::circle(debugImage, cv::Point(point.x(), point.y()), 4, cv::Scalar(0, 255, 255), 1);
            }
            else {
                // no depth
                cv::circle(debugImage, cv::Point(point.x(), point.y()), 4, cv::Scalar(255, 255, 0), 1);
            }
        }
    }
}

double get_percent_of_elapsed_time(double treatmentTime, double totalTimeElapsed) 
{
    if (totalTimeElapsed <= 0)
        return 0;
    return std::round(treatmentTime / totalTimeElapsed * 10000) / 100;
}

void Key_Point_Extraction::show_statistics(double meanFrameTreatmentTime, unsigned int frameCount) {
    if (frameCount > 0) { 
        double meanPointExtractionTime = _meanPointExtractionTime / frameCount;
        std::cout << "Mean point extraction time is " << meanPointExtractionTime << " seconds (" << get_percent_of_elapsed_time(meanPointExtractionTime, meanFrameTreatmentTime) << "%)" << std::endl;
    }
}



