#include "KeyPointDetection.hpp"
#include "parameters.hpp"

#include "utils.hpp"

// circle
#include <opencv2/opencv.hpp>

const float BORDER_SIZE = 1.; // Border of an image, in which points will be ignored

namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            /**
             * \brief checks if a point is in an image, with border
             */
            bool is_in_border(const cv::Point2f &pt, const cv::Mat &im) 
            {

                return BORDER_SIZE <= pt.x && pt.x < im.cols - BORDER_SIZE && BORDER_SIZE <= pt.y && pt.y < im.rows - BORDER_SIZE;
            } 


            Keypoint_Handler::Keypoint_Handler(std::vector<cv::KeyPoint>& inKeypoints, cv::Mat& inDescriptors, const cv::Mat& depthImage, const double maxMatchDistance) :
                _maxMatchDistance(maxMatchDistance)
            {
                if (_maxMatchDistance <= 0) {
                    utils::log_error("Maximum matching distance must be > 0");
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
                        // Depths are in millimeters
                        _depths[pointIndex] = (depthImage.at<const float>(pt.y, pt.x));
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
                _advancedFeatureDetector = cv::FastFeatureDetector::create( minHessian / 2 );
                _descriptorExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

                //profiling
                _meanPointExtractionTime = 0.0;
            }

            const std::vector<cv::Point2f> Key_Point_Extraction::detect_keypoints(const cv::Mat& grayImage, const cv::Mat& mask, const size_t minimumPointsForValidity)
            {
                std::vector<cv::Point2f> framePoints;
                std::vector<cv::KeyPoint> frameKeypoints;
                _featureDetector->detect(grayImage, frameKeypoints, mask); 

                if (frameKeypoints.size() <=  minimumPointsForValidity)
                {
                    // Not enough keypoints detected: restart with a more precise detector
                    frameKeypoints.clear();
                    _advancedFeatureDetector->detect(grayImage, frameKeypoints, mask); 
                }

                if (frameKeypoints.size() >  minimumPointsForValidity)
                {
                    // Refine keypoints positions
                    framePoints.reserve(frameKeypoints.size());
                    cv::KeyPoint::convert(frameKeypoints, framePoints);

                    const cv::Size winSize  = cv::Size(3, 3);
                    const cv::Size zeroZone = cv::Size(-1, -1);
                    const cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
                    cv::cornerSubPix(grayImage, framePoints, winSize, zeroZone, termCriteria);
                }

                return framePoints;
            }

            const cv::Mat Key_Point_Extraction::compute_key_point_mask(const cv::Size imageSize, const std::vector<cv::Point2f> keypointContainer)
            {
                const size_t radiusOfAreaAroundPoint = Parameters::get_keypoint_mask_diameter();  // in pixels
                const cv::Scalar fillColor(0, 0, 0);
                cv::Mat mask = cv::Mat::ones(imageSize, CV_8UC1);
                for (const cv::Point2f& point : keypointContainer)
                {
#if 1
                    cv::circle(mask, point, radiusOfAreaAroundPoint, fillColor, -1);
#else
                    const int areaXmin = point.x - radiusOfAreaAroundPoint;
                    const int areaXmax = point.x + radiusOfAreaAroundPoint;
                    const int areaYmin = point.y - radiusOfAreaAroundPoint;
                    const int areaYmax = point.y + radiusOfAreaAroundPoint;
                    cv::rectangle(mask, cv::Point(areaXmin, areaYmin), cv::Point(areaXmax, areaYmax), fillColor, -1);
#endif
                }
                return mask;
            }

            const Keypoint_Handler Key_Point_Extraction::compute_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage, const bool forceKeypointDetection) 
            {
                cv::Mat frameDescriptors;

                //detect keypoints
                double t1 = cv::getTickCount();

                /*
                 * OPTICAL FLOW
                 */

                // load parameters
                const size_t pyramidWindowSize = Parameters::get_optical_flow_pyramid_windown_size();
                const size_t pyramidDepth = Parameters::get_optical_flow_pyramid_depth();
                const size_t maxError = Parameters::get_optical_flow_max_error();
                const size_t maxDistance = Parameters::get_optical_flow_max_distance();

                const cv::Size pyramidSize = cv::Size(pyramidWindowSize, pyramidWindowSize);   // must be bigger than the size used in calcOpticalFlow

                // build pyramid
                std::vector<cv::Mat> newImagePyramide;
                cv::buildOpticalFlowPyramid(grayImage, newImagePyramide, pyramidSize, pyramidDepth);
                if (_lastFramePyramide.size() > 0)
                {
                    if (_lastKeypoints.size() > 0) {
                        const Key_Point_Extraction::KeypointsWithStatusStruct& keypointsWithStatus = get_keypoints_from_optical_flow(_lastFramePyramide, newImagePyramide, _lastKeypoints, pyramidDepth, pyramidWindowSize, maxError, maxDistance);

                        _lastKeypoints = keypointsWithStatus._keypoints;
                    }
                    else
                    {
                        utils::log_error("No keypoints available to use optical flow algorithm");
                    }
                }
                //else: No optical flow for the first frame
                _lastFramePyramide = newImagePyramide;

                /*
                 * KEY POINT DETECTION
                 *      Use keypoint detection when low on keypoints, or when requested
                 */   

                // detect keypoint if: it is requested OR not enough points were detected
                const size_t minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();
                const bool shouldDetectKeypoints = forceKeypointDetection or _lastKeypoints.size() < minimumPointsForOptimization;
                if (shouldDetectKeypoints and _lastKeypoints.size() < Parameters::get_maximum_point_count_per_frame())
                {
                    // create a mask at current keypoint location
                    const cv::Mat& keypointMask = compute_key_point_mask(grayImage.size(), _lastKeypoints);

                    // get new keypoints
                    const std::vector<cv::Point2f>& keypoints = detect_keypoints(grayImage, keypointMask, minimumPointsForOptimization);

                    // merge keypoints
                    _lastKeypoints.insert(_lastKeypoints.end(), keypoints.begin(), keypoints.end());
                }

                /**
                 *  DESCRIPTORS
                 */

                std::vector<cv::KeyPoint> frameKeypoints;
                frameKeypoints.reserve(_lastKeypoints.size());
                cv::KeyPoint::convert(_lastKeypoints, frameKeypoints);

                // Compute descriptors
                // Caution: the frameKeypoints list is mutable by this function, and should be use instead of _lastKeypoints
                _descriptorExtractor->compute(grayImage, frameKeypoints, frameDescriptors);

                _meanPointExtractionTime += (cv::getTickCount() - t1) / static_cast<double>(cv::getTickFrequency());

                return Keypoint_Handler(frameKeypoints, frameDescriptors, depthImage, Parameters::get_maximum_match_distance()); 
            }


            const Key_Point_Extraction::KeypointsWithStatusStruct Key_Point_Extraction::get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide, const std::vector<cv::Mat>& imageCurrentPyramide, const std::vector<cv::Point2f>& keypointsPrevious, const size_t pyramidDepth, const size_t windowSize, const double errorThreshold, const double maxDistanceThreshold)
            {
                KeypointsWithStatusStruct keypointStruct;
                keypointStruct._isValid = false;

                // START of optical flow
                if (imagePreviousPyramide.size() <= 0 or imageCurrentPyramide.size() <= 0 or errorThreshold < 0 or keypointsPrevious.size() <= 0)
                {
                    utils::log_error("OpticalFlow: invalid parameters");
                    return keypointStruct;
                }

                // Calculate optical flow
                std::vector<uchar> statusContainer;
                std::vector<float> errorContainer;
                std::vector<cv::Point2f> forwardPoints;

                const size_t previousKeyPointCount = keypointsPrevious.size();
                // Reserve room for the status, error and points
                statusContainer.reserve(previousKeyPointCount);
                errorContainer.reserve(previousKeyPointCount);
                forwardPoints.reserve(previousKeyPointCount);

                const cv::Size windowSizeObject = cv::Size(windowSize, windowSize);
                const cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

                // Get forward points: optical flow from previous to current image to extract new keypoints
                cv::calcOpticalFlowPyrLK(imagePreviousPyramide, imageCurrentPyramide, keypointsPrevious, forwardPoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

                // Contains the keypoints from this frame, without outliers
                std::vector<cv::Point2f> backwardKeypoints;
                // contains the ids of the good waypoints
                std::vector<unsigned int> keypointIds;

                backwardKeypoints.reserve(previousKeyPointCount);
                keypointIds.reserve(previousKeyPointCount);

                // set output structure
                keypointStruct._keypoints.reserve(previousKeyPointCount);
                keypointStruct._status.reserve(previousKeyPointCount);


                // Remove outliers from current waypoint list by creating a new one
                for(uint keypointIndex = 0; keypointIndex < previousKeyPointCount; ++keypointIndex)
                {
                    if(statusContainer[keypointIndex] != 1) {
                        // point was not associated
                        keypointStruct._status.push_back(false);
                        continue;
                    }

                    if (errorContainer[keypointIndex] > errorThreshold)
                    {
                        // point error is too great
                        keypointStruct._status.push_back(false);
                        continue;
                    }

                    if (not is_in_border(forwardPoints[keypointIndex], imageCurrentPyramide.at(0)))
                    {
                        // point not in image borders
                        keypointStruct._status.push_back(false);
                        continue;
                    }

                    keypointStruct._keypoints.push_back(forwardPoints[keypointIndex]);
                    backwardKeypoints.push_back(keypointsPrevious[keypointIndex]);
                    keypointIds.push_back(keypointIndex);
                    keypointStruct._status.push_back(true);
                }

                if (keypointStruct._keypoints.size() <= 0)
                {
                    utils::log("No new points detected for backtracking", std::source_location::current());
                    return keypointStruct;
                }

                // Backward tracking: go from this frame inliers to the last frame inliers
                cv::calcOpticalFlowPyrLK(imageCurrentPyramide, imagePreviousPyramide, keypointStruct._keypoints, backwardKeypoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

                // mark outliers as false and visualize
                const size_t keypointSize = keypointStruct._keypoints.size();
                for(uint i = 0; i < keypointSize; i++)
                {
                    const size_t keypointIndex = keypointIds[i];
                    if(statusContainer[i] != 1) {
                        keypointStruct._status[keypointIndex] = false;
                        continue;
                    }

                    if (cv::norm(keypointsPrevious[keypointIndex] - backwardKeypoints[i]) > maxDistanceThreshold) 
                    {
                        keypointStruct._status[keypointIndex] = false;
                        continue;
                    }
                }
                keypointStruct._isValid = true;

                return keypointStruct;
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
