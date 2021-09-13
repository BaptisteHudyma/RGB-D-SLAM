#include "KeyPointDetection.hpp"
#include "parameters.hpp"

// circle
#include <opencv2/opencv.hpp>

// cout cerr
#include <iostream>

#include "utils.hpp"

// Error display
#define CLASS_ERR "<Key_Point_Extraction> "

namespace rgbd_slam {
    namespace utils {


        Keypoint_Handler::Keypoint_Handler(std::vector<cv::KeyPoint>& inKeypoints, cv::Mat& inDescriptors, const cv::Mat& depthImage, const double maxMatchDistance) :
            _maxMatchDistance(maxMatchDistance)
        {
            if (_maxMatchDistance <= 0) {
                std::cerr << CLASS_ERR << "Maximum matching distance must be > 0" << std::endl;
                exit(-1);
            }
            // knn matcher
            _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));

            _keypoints.swap(inKeypoints);
            _descriptors = inDescriptors;

            // Fill depth values
            cv::Rect imageBoundaries(cv::Point(), depthImage.size());
            _depths = std::vector<double>(_keypoints.size());
            unsigned int i = 0;
            for(const cv::KeyPoint& keypoint : _keypoints) {
                const cv::Point2f& pt = keypoint.pt;
                assert(pt.x > 0 and pt.y > 0);

                if (imageBoundaries.contains(pt)) {
                    // convert to meters
                    _depths[i] = (depthImage.at<const float>(pt.y, pt.x) / 1000 );
                }
                ++i;
            }
            assert(_keypoints.size() == _depths.size());
        }

        int Keypoint_Handler::get_match_index(const Point& mapPoint) const
        {
            std::vector< std::vector<cv::DMatch> > knnMatches;
            _featuresMatcher->knnMatch(mapPoint._descriptor, _descriptors, knnMatches, 2);

            //check the farthest neighbors
            if (knnMatches[0].size() > 1) {
                const std::vector<cv::DMatch>& match = knnMatches[0];
                //check if point is a good match by checking it's distance to the second best matched point
                if (match[0].distance < _maxMatchDistance * match[1].distance) {
                    int id = match[0].trainIdx;
                    return id;   //this frame key point
                }
                return -1;
            }
            else if (knnMatches[0].size() == 1) {
                int id = knnMatches[0][0].trainIdx;
                return id;   //this frame key point
            }
            return -1;
        }



        /*
         * Keypoint extraction
         */


        Key_Point_Extraction::Key_Point_Extraction(const unsigned int minHessian) 
        {
            // Create feature extractor and matcher
            _featureDetector = cv::FastFeatureDetector::create( minHessian );
            _descriptorExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

            //profiling
            _meanPointExtractionTime = 0.0;
        }

        const Keypoint_Handler Key_Point_Extraction::detect_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage) 
        {
            std::vector<cv::KeyPoint> frameKeypoints;

            //detect keypoints
            double t1 = cv::getTickCount();
            _featureDetector->detect(grayImage, frameKeypoints); 
            _meanPointExtractionTime += (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());

            cv::Mat frameDescriptors;
            _descriptorExtractor->compute(grayImage, frameKeypoints, frameDescriptors);

            return Keypoint_Handler(frameKeypoints, frameDescriptors, depthImage, Parameters::get_maximum_match_distance()); 
        }



        double get_percent_of_elapsed_time(const double treatmentTime, const double totalTimeElapsed) 
        {
            if (totalTimeElapsed <= 0)
                return 0;
            return std::round(treatmentTime / totalTimeElapsed * 10000.0) / 100.0;
        }

        void Key_Point_Extraction::show_statistics(const double meanFrameTreatmentTime, const unsigned int frameCount) const {
            if (frameCount > 0) { 
                double meanPointExtractionTime = _meanPointExtractionTime / frameCount;
                std::cout << "Mean point extraction time is " << meanPointExtractionTime << " seconds (" << get_percent_of_elapsed_time(meanPointExtractionTime, meanFrameTreatmentTime) << "%)" << std::endl;
            }
        }


    }
}

