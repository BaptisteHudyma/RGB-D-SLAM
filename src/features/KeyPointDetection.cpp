#include "KeyPointDetection.hpp"
#include "parameters.hpp"

#include "utils.hpp"

// circle
#include <opencv2/opencv.hpp>

// cout cerr
#include <iostream>

// Error display
#define CLASS_ERR "<Key_Point_Extraction> "

namespace rgbd_slam {
namespace features {
namespace keypoints {



        Keypoint_Handler::Keypoint_Handler(std::vector<cv::KeyPoint>& inKeypoints, cv::Mat& inDescriptors, const cv::Mat& depthImage, const double maxMatchDistance) :
            _maxMatchDistance(maxMatchDistance)
        {
            if (_maxMatchDistance <= 0) {
                std::cerr << CLASS_ERR << "Maximum matching distance must be > 0" << std::endl;
                exit(-1);
            }
            // knn matcher
            _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));

            _descriptors = inDescriptors;

            const float cellSize = static_cast<float>(Parameters::get_search_matches_cell_size());
            _searchSpaceCellRadius = std::ceil(Parameters::get_search_matches_distance() / cellSize);

            _cellCountX = std::ceil(depthImage.cols / cellSize);
            _cellCountY = std::ceil(depthImage.rows / cellSize);

            _searchSpaceIndexContainer.resize(_cellCountY * _cellCountX);

            // Fill depth values, add points to image boxes
            cv::Rect imageBoundaries(cv::Point(), depthImage.size());

            const unsigned int keyPointSize = inKeypoints.size();
            _depths = std::vector<double>(keyPointSize, 0.0);
            _keypoints = std::vector<vector2>(keyPointSize);

            unsigned int pointIndex = 0;
            for(const cv::KeyPoint& keypoint : inKeypoints) {
                const cv::Point2f pt = keypoint.pt;
                const vector2 vectorKeypoint(pt.x, pt.y); 

                _keypoints[pointIndex] = vectorKeypoint; 

                const unsigned int searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint));
                _searchSpaceIndexContainer[searchSpaceIndex].push_back(pointIndex);

                if (imageBoundaries.contains(pt)) {
                    // convert to meters
                    _depths[pointIndex] = (depthImage.at<const float>(pt.y, pt.x)) * 0.001;
                }
                ++pointIndex;
            }
        }


        unsigned int Keypoint_Handler::get_search_space_index(const int_pair& searchSpaceIndex) const
        {
            return get_search_space_index(searchSpaceIndex.second, searchSpaceIndex.first);
        }
        unsigned int Keypoint_Handler::get_search_space_index(const unsigned int x, const unsigned int y) const 
        {
            return y * _cellCountY + x;
        }


        typedef std::pair<int, int> int_pair;
        const int_pair Keypoint_Handler::get_search_space_coordinates(const vector2& pointToPlace) const
        {
            const double cellSize = static_cast<double>(Parameters::get_search_matches_cell_size());
            const int_pair cellCoordinates(
                    std::max(0.0, std::min(floor(pointToPlace.y() / cellSize), _cellCountY - 1.0)),
                    std::max(0.0, std::min(floor(pointToPlace.x() / cellSize), _cellCountX - 1.0))
                    );
            return cellCoordinates;
        }

        const cv::Mat Keypoint_Handler::compute_key_point_mask(const vector2& pointToSearch, const std::vector<bool>& isKeyPointMatchedContainer) const
        {
            const int_pair& searchSpaceCoordinates = get_search_space_coordinates(pointToSearch);

            const unsigned int startY = std::max(0, searchSpaceCoordinates.first - _searchSpaceCellRadius);
            const unsigned int startX = std::max(0, searchSpaceCoordinates.second - _searchSpaceCellRadius);

            const unsigned int endY = std::min(_cellCountY, searchSpaceCoordinates.first + _searchSpaceCellRadius + 1);
            const unsigned int endX = std::min(_cellCountX, searchSpaceCoordinates.second + _searchSpaceCellRadius + 1);

            // Squared search diameter, to compare distance without sqrt
            const float squaredSearchDiameter = pow(Parameters::get_search_matches_distance(), 2);

            cv::Mat keyPointMask(cv::Mat::zeros(1, _keypoints.size(), CV_8UC1));
            for (unsigned int i = startY; i < endY; ++i)
            {
                for (unsigned int j = startX; j < endX; ++j)
                {
                    const index_container& keypointIndexContainer = _searchSpaceIndexContainer[get_search_space_index(j, i)]; 
                    for(int keypointIndex : keypointIndexContainer)
                    {
                        if (not isKeyPointMatchedContainer[keypointIndex])
                        {
                            const vector2& keypoint = get_keypoint(keypointIndex);
                            const double squarredDistance = 
                                pow(keypoint.x() - pointToSearch.x(), 2.0) + 
                                pow(keypoint.y() - pointToSearch.y(), 2.0);

                            if (squarredDistance <= squaredSearchDiameter)
                                keyPointMask.at<uint8_t>(0, keypointIndex) = 1;
                        }
                    }
                }
            }

            return keyPointMask;
        }


        int Keypoint_Handler::get_match_index(const vector2& projectedMapPoint, const cv::Mat& mapPointDescriptor, const std::vector<bool>& isKeyPointMatchedContainer) const
        {
            if (_keypoints.size() <= 0)
                return -1;

            const cv::Mat& keyPointMask = compute_key_point_mask(projectedMapPoint, isKeyPointMatchedContainer);

            std::vector< std::vector<cv::DMatch> > knnMatches;
            _featuresMatcher->knnMatch(mapPointDescriptor, _descriptors, knnMatches, 2, keyPointMask);

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
            cv::Mat frameDescriptors;

            //detect keypoints
            double t1 = cv::getTickCount();
            _featureDetector->detect(grayImage, frameKeypoints); 

            if (frameKeypoints.size() > 0)
            {
                std::vector<cv::Point2f> framePoints;
                framePoints.reserve(frameKeypoints.size());
                cv::KeyPoint::convert(frameKeypoints, framePoints);

                cv::Size winSize  = cv::Size(3, 3);
                cv::Size zeroZone = cv::Size(-1, -1);
                cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
                cv::cornerSubPix(grayImage, framePoints, winSize, zeroZone, termCriteria);

                frameKeypoints.clear();
                frameKeypoints.reserve(framePoints.size());
                cv::KeyPoint::convert(framePoints, frameKeypoints);

                _descriptorExtractor->compute(grayImage, frameKeypoints, frameDescriptors);
            }
            _meanPointExtractionTime += (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());

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
}
