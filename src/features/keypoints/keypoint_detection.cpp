#include "keypoint_detection.hpp"

#include "parameters.hpp"
#include "logger.hpp"

// circle
#include <opencv2/opencv.hpp>


namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            /*
             * Keypoint extraction
             */

            Key_Point_Extraction::Key_Point_Extraction(const uint minHessian) :
                // Create feature extractor and matcher
                _featureDetector(cv::FastFeatureDetector::create(minHessian)),
                _advancedFeatureDetector(cv::FastFeatureDetector::create(minHessian * 0.5)),
                _descriptorExtractor(cv::xfeatures2d::BriefDescriptorExtractor::create())
            {
                assert(not _featureDetector.empty() );
                assert(not _advancedFeatureDetector.empty() );
                assert(not _descriptorExtractor.empty() );

                //profiling
                _meanPointExtractionTime = 0.0;
            }

            const std::vector<cv::Point2f> Key_Point_Extraction::detect_keypoints(const cv::Mat& grayImage, const cv::Mat& mask, const uint minimumPointsForValidity) const
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
                    cv::KeyPoint::convert(frameKeypoints, framePoints);

                    const cv::Size winSize  = cv::Size(3, 3);
                    const cv::Size zeroZone = cv::Size(-1, -1);
                    const cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
                    cv::cornerSubPix(grayImage, framePoints, winSize, zeroZone, termCriteria);
                }

                return framePoints;
            }

            const cv::Mat Key_Point_Extraction::compute_key_point_mask(const cv::Size imageSize, const std::vector<cv::Point2f>& keypointContainer) const
            {
                const uint radiusOfAreaAroundPoint = Parameters::get_keypoint_mask_diameter();  // in pixels
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

            const Keypoint_Handler Key_Point_Extraction::compute_keypoints(const cv::Mat& grayImage, const cv::Mat& depthImage, const KeypointsWithIdStruct& lastKeypointsWithIds, const bool forceKeypointDetection) 
            {
                assert(lastKeypointsWithIds._keypoints.size() == lastKeypointsWithIds._ids.size());

                KeypointsWithIdStruct newKeypointsObject;
                cv::Mat keypointDescriptors;

                //detect keypoints
                int64 t1 = cv::getTickCount();

                /*
                 * OPTICAL FLOW
                 */

                // load parameters
                const uint pyramidWindowSize = Parameters::get_optical_flow_pyramid_windown_size();
                const uint pyramidDepth = Parameters::get_optical_flow_pyramid_depth();
                const uint maxError = Parameters::get_optical_flow_max_error();
                const uint maxDistance = Parameters::get_optical_flow_max_distance();
                const uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();
                const uint maximumPointsForLocalMap = Parameters::get_maximum_point_count_per_frame();
                const double maximumMatchDistance = Parameters::get_maximum_match_distance();

                const cv::Size pyramidSize = cv::Size(pyramidWindowSize, pyramidWindowSize);   // must be >= than the size used in calcOpticalFlow

                // build pyramid
                std::vector<cv::Mat> newImagePyramide;
                cv::buildOpticalFlowPyramid(grayImage, newImagePyramide, pyramidSize, pyramidDepth);
                // TODO: when the optical flow will not show so much drift, maybe we could remove the tracked keypoint redetection
                if (not forceKeypointDetection and _lastFramePyramide.size() > 0)
                {
                    if (lastKeypointsWithIds._keypoints.size() > 0) {
                        newKeypointsObject = get_keypoints_from_optical_flow(_lastFramePyramide, newImagePyramide, lastKeypointsWithIds, pyramidDepth, pyramidWindowSize, maxError, maxDistance);

                        // TODO: add descriptors to handle short term rematching of lost optical flow features
                    }
                    else
                    {
                        utils::log_error("No keypoints available to use optical flow algorithm");
                    }
                }
                //else: No optical flow for the first frame
                _lastFramePyramide = newImagePyramide;

                const size_t opticalFlowTrackedPointCount = newKeypointsObject._keypoints.size();
                assert(opticalFlowTrackedPointCount == newKeypointsObject._ids.size());

                /*
                 * KEY POINT DETECTION
                 *      Use keypoint detection when low on keypoints, or when requested
                 */   

                // detect keypoint if: it is requested OR not enough points were detected
                std::vector<cv::Point2f> detectedKeypoints;
                const bool shouldDetectKeypoints = opticalFlowTrackedPointCount < minimumPointsForOptimization and opticalFlowTrackedPointCount < maximumPointsForLocalMap;
                if (forceKeypointDetection or shouldDetectKeypoints)
                {
                    // create a mask at current keypoint location
                    const cv::Mat& keypointMask = compute_key_point_mask(grayImage.size(), newKeypointsObject._keypoints);

                    // get new keypoints
                    detectedKeypoints = detect_keypoints(grayImage, keypointMask, minimumPointsForOptimization);
                }

                cv::Mat detectedKeypointDescriptors;
                if (detectedKeypoints.size() > 0)
                {
                    /**
                     *  DESCRIPTORS
                     */
                    std::vector<cv::KeyPoint> frameKeypoints;
                    cv::KeyPoint::convert(detectedKeypoints, frameKeypoints);

                    // Compute descriptors
                    // Caution: the frameKeypoints list is mutable by this function
                    //          The bad points will be removed by the compute descriptor function
                    _descriptorExtractor->compute(grayImage, frameKeypoints, detectedKeypointDescriptors);

                    // convert back to keypoint list
                    detectedKeypoints.clear();
                    cv::KeyPoint::convert(frameKeypoints, detectedKeypoints);

                    if (keypointDescriptors.rows > 0)
                        cv::vconcat(detectedKeypointDescriptors, keypointDescriptors, keypointDescriptors);
                    else
                        keypointDescriptors = detectedKeypointDescriptors;
                }

                const double deltaTime = static_cast<double>(cv::getTickCount() - t1);
                _meanPointExtractionTime += deltaTime / static_cast<double>(cv::getTickFrequency());

                // Update last keypoint struct
                return Keypoint_Handler(detectedKeypoints, keypointDescriptors, newKeypointsObject, depthImage, maximumMatchDistance);
            }


            KeypointsWithIdStruct Key_Point_Extraction::get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide, const std::vector<cv::Mat>& imageCurrentPyramide, const KeypointsWithIdStruct& lastKeypointsWithIds, const uint pyramidDepth, const uint windowSize, const double errorThreshold, const double maxDistanceThreshold)
            {
                assert(lastKeypointsWithIds._keypoints.size() == lastKeypointsWithIds._ids.size());

                KeypointsWithIdStruct keypointStruct;

                // START of optical flow
                const std::vector<cv::Point2f>& lastKeypoints = lastKeypointsWithIds._keypoints;
                if (imagePreviousPyramide.empty() or imageCurrentPyramide.empty() or errorThreshold < 0 or lastKeypoints.empty())
                {
                    utils::log_error("OpticalFlow: invalid parameters");
                    return keypointStruct;
                }

                // Calculate optical flow
                std::vector<uchar> statusContainer;
                std::vector<float> errorContainer;
                std::vector<cv::Point2f> forwardPoints;

                const size_t previousKeyPointCount = lastKeypoints.size();
                const cv::Size windowSizeObject = cv::Size(windowSize, windowSize);
                const cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

                // Get forward points: optical flow from previous to current image to extract new keypoints
                cv::calcOpticalFlowPyrLK(imagePreviousPyramide, imageCurrentPyramide, lastKeypoints, forwardPoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

                // contains the ids of the good waypoints
                std::vector<size_t> keypointIndexContainer;
                keypointIndexContainer.reserve(previousKeyPointCount);

                // set output structure
                std::vector<cv::Point2f> newKeypoints;
                newKeypoints.reserve(previousKeyPointCount);

                // Remove outliers from current waypoint list by creating a new one
                for(size_t keypointIndex = 0; keypointIndex < previousKeyPointCount; ++keypointIndex)
                {
                    if(statusContainer[keypointIndex] != 1) {
                        // point was not associated
                        continue;
                    }
                    if (errorContainer[keypointIndex] > errorThreshold)
                    {
                        // point error is too great
                        continue;
                    }
                    if (not is_in_border(forwardPoints[keypointIndex], imageCurrentPyramide.at(0)))
                    {
                        // point not in image borders
                        continue;
                    }

                    newKeypoints.push_back(forwardPoints[keypointIndex]);
                    keypointIndexContainer.push_back(keypointIndex);
                }

                if (newKeypoints.empty())
                {
                    utils::log("No new points detected for backtracking", std::source_location::current());
                    return keypointStruct;
                }

                // Contains the keypoints from this frame, without outliers
                std::vector<cv::Point2f> backwardKeypoints;

                // Backward tracking: go from this frame inliers to the last frame inliers
                cv::calcOpticalFlowPyrLK(imageCurrentPyramide, imagePreviousPyramide, newKeypoints, backwardKeypoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

                // mark outliers as false and visualize
                const size_t keypointSize = backwardKeypoints.size();
                keypointStruct._ids.reserve(keypointSize);
                keypointStruct._keypoints.reserve(keypointSize);
                for(size_t i = 0; i < keypointSize; ++i)
                {
                    const size_t keypointIndex = keypointIndexContainer[i];
                    if(statusContainer[i] != 1) {
                        continue;
                    }
                    // check distance of the backpropagated point to the original point
                    if (cv::norm(lastKeypoints[keypointIndex] - backwardKeypoints[i]) > maxDistanceThreshold) {
                        continue;
                    }

                    keypointStruct._keypoints.push_back(forwardPoints[keypointIndex]);
                    keypointStruct._ids.push_back(lastKeypointsWithIds._ids[keypointIndex]);
                }
                return keypointStruct;
            }



            double get_percent_of_elapsed_time(const double treatmentTime, const double totalTimeElapsed) 
            {
                if (totalTimeElapsed <= 0)
                    return 0;
                return std::round(treatmentTime / totalTimeElapsed * 10000.0) / 100.0;
            }

            void Key_Point_Extraction::show_statistics(const double meanFrameTreatmentTime, const uint frameCount) const {
                if (frameCount > 0) { 
                    double meanPointExtractionTime = _meanPointExtractionTime / frameCount;
                    std::cout << "Mean point extraction time is " << meanPointExtractionTime << " seconds (" << get_percent_of_elapsed_time(meanPointExtractionTime, meanFrameTreatmentTime) << "%)" << std::endl;
                }
            }

        }
    }
}
