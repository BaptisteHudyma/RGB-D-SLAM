#include "keypoint_detection.hpp"

#include "../../parameters.hpp"
#include "../../outputs/logger.hpp"

// circle
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>


namespace rgbd_slam {
    namespace features {
        namespace keypoints {

            /*
             * Keypoint extraction
             */

            Key_Point_Extraction::Key_Point_Extraction(const uint maxFeaturesToDetect) :
                // Create feature extractor and matcher
                _featureDetector(cv::ORB::create(maxFeaturesToDetect)),
                _advancedFeatureDetector(cv::ORB::create(maxFeaturesToDetect)),
                _meanPointExtractionDuration(0.0)
            {
                assert(not _featureDetector.empty());
                assert(not _advancedFeatureDetector.empty());
            }

            const std::vector<cv::Point2f> Key_Point_Extraction::detect_keypoints(const cv::Mat& grayImage, const cv::Mat& mask, const uint minimumPointsForValidity) const
            {
                // search keypoints, using an advanced detector if not enough features are found
                std::vector<cv::KeyPoint> frameKeypoints;
                assert(grayImage.size() == mask.size());
                _featureDetector->detect(grayImage, frameKeypoints, mask); 

                if (frameKeypoints.size() <=  minimumPointsForValidity)
                {
                    // Not enough keypoints detected: restart with a more precise detector
                    frameKeypoints.clear();
                    _advancedFeatureDetector->detect(grayImage, frameKeypoints, mask); 
                }

                // compute a subcorner accuracy estimation for all waypoints 
                std::vector<cv::Point2f> framePoints;
                if (frameKeypoints.size() >  minimumPointsForValidity)
                {
                    // Refine keypoints positions
                    cv::KeyPoint::convert(frameKeypoints, framePoints);

                    const static cv::Size winSize  = cv::Size(3, 3);
                    const static cv::Size zeroZone = cv::Size(-1, -1);
                    const static cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
                    cv::cornerSubPix(grayImage, framePoints, winSize, zeroZone, termCriteria);
                }

                return framePoints;
            }

            const cv::Mat Key_Point_Extraction::compute_key_point_mask(const cv::Size imageSize, const std::vector<cv::Point2f>& keypointContainer) const
            {
                const static uint radiusOfAreaAroundPoint = Parameters::get_keypoint_mask_radius();  // in pixels
                const static cv::Scalar fillColor(0);
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
                KeypointsWithIdStruct newKeypointsObject;

                //detect keypoints
                const int64 keypointDetectionStartTime = cv::getTickCount();

                /*
                 * OPTICAL FLOW
                 */

                // load parameters
                const static uint pyramidWindowSize = Parameters::get_optical_flow_pyramid_windown_size();
                const static uint pyramidDepth = Parameters::get_optical_flow_pyramid_depth();
                const static uint maxError = Parameters::get_optical_flow_max_error();
                const static uint maxDistance = Parameters::get_optical_flow_max_distance();
                const static uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();
                const static uint maximumPointsForLocalMap = Parameters::get_maximum_point_count_per_frame();
                const static double maximumMatchDistance = Parameters::get_maximum_match_distance();

                const static cv::Size pyramidSize(pyramidWindowSize, pyramidWindowSize);   // must be >= than the size used in calcOpticalFlow

                // build pyramid
                std::vector<cv::Mat> newImagePyramide;
                cv::buildOpticalFlowPyramid(grayImage, newImagePyramide, pyramidSize, pyramidDepth);
                // TODO: when the optical flow will not show so much drift, maybe we could remove the tracked keypoint redetection
                if (_lastFramePyramide.size() > 0)
                {
                    if (lastKeypointsWithIds.size() > 0) {
                        get_keypoints_from_optical_flow(_lastFramePyramide, newImagePyramide, lastKeypointsWithIds, pyramidDepth, pyramidWindowSize, maxError, maxDistance, newKeypointsObject);

                        // TODO: add descriptors to handle short term rematching of lost optical flow features
                    }
                    //else: no optical flow
                }
                //else: No optical flow for the first frame
                _lastFramePyramide = newImagePyramide;

                const size_t opticalFlowTrackedPointCount = newKeypointsObject.size();

                /*
                 * KEY POINT DETECTION
                 *      Use keypoint detection when low on keypoints, or when requested
                 */   

                // detect keypoint if: it is requested OR not enough points were detected
                std::vector<cv::Point2f> detectedKeypoints;
                cv::Mat keypointDescriptors;
                // TODO: better metric to search for more keypoints
                const bool shouldDetectKeypoints = opticalFlowTrackedPointCount < minimumPointsForOptimization * 3 and opticalFlowTrackedPointCount < maximumPointsForLocalMap;
                if (forceKeypointDetection or shouldDetectKeypoints)
                {
                    // create a mask at current keypoint location
                    const cv::Mat& keypointMask = compute_key_point_mask(grayImage.size(), newKeypointsObject.get_keypoints());

                    // get new keypoints
                    detectedKeypoints = detect_keypoints(grayImage, keypointMask, minimumPointsForOptimization);

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
                        cv::Mat detectedKeypointDescriptors;
                        _featureDetector->compute(grayImage, frameKeypoints, detectedKeypointDescriptors);

                        // convert back to keypoint list
                        detectedKeypoints.clear();
                        cv::KeyPoint::convert(frameKeypoints, detectedKeypoints);

                        if (keypointDescriptors.rows > 0)
                            cv::vconcat(detectedKeypointDescriptors, keypointDescriptors, keypointDescriptors);
                        else
                            keypointDescriptors = detectedKeypointDescriptors;
                    }
                }

                // declare static
                static Keypoint_Handler keypointHandler(depthImage.cols, depthImage.rows, maximumMatchDistance);

                // Update last keypoint struct
                keypointHandler.set(detectedKeypoints, keypointDescriptors, newKeypointsObject, depthImage);
                _meanPointExtractionDuration += (cv::getTickCount() - keypointDetectionStartTime) / static_cast<double>(cv::getTickFrequency());
                return keypointHandler;
            }


            void Key_Point_Extraction::get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide, const std::vector<cv::Mat>& imageCurrentPyramide, const KeypointsWithIdStruct& lastKeypointsWithIds, const uint pyramidDepth, const uint windowSize, const double errorThreshold, const double maxDistanceThreshold, KeypointsWithIdStruct& keypointStruct)
            {
                // START of optical flow
                if (imagePreviousPyramide.empty() or imageCurrentPyramide.empty() or errorThreshold < 0 or lastKeypointsWithIds.empty())
                {
                    outputs::log_error("OpticalFlow: invalid parameters");
                    return;
                }

                // Calculate optical flow
                std::vector<uchar> statusContainer;
                std::vector<float> errorContainer;
                std::vector<cv::Point2f> forwardPoints;

                const size_t previousKeyPointCount = lastKeypointsWithIds.size();
                const cv::Size windowSizeObject = cv::Size(windowSize, windowSize);
                const static cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);

                // Get forward points: optical flow from previous to current image to extract new keypoints
                cv::calcOpticalFlowPyrLK(imagePreviousPyramide, imageCurrentPyramide, lastKeypointsWithIds.get_keypoints(), forwardPoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

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
                    outputs::log("No new points detected for backtracking");
                    return;
                }

                // Contains the keypoints from this frame, without outliers
                std::vector<cv::Point2f> backwardKeypoints;

                // Backward tracking: go from this frame inliers to the last frame inliers
                cv::calcOpticalFlowPyrLK(imageCurrentPyramide, imagePreviousPyramide, newKeypoints, backwardKeypoints, statusContainer, errorContainer, windowSizeObject, pyramidDepth, criteria);

                // mark outliers as false and visualize
                const size_t keypointSize = backwardKeypoints.size();
                keypointStruct.reserve(keypointSize);
                for(size_t i = 0; i < keypointSize; ++i)
                {
                    const size_t keypointIndex = keypointIndexContainer[i];
                    if(statusContainer[i] != 1) {
                        continue;
                    }

                    const KeypointsWithIdStruct::keypointWithId& lastKeypoint = lastKeypointsWithIds.get(keypointIndex);
                    // check distance of the backpropagated point to the original point
                    if (cv::norm(lastKeypoint._point - backwardKeypoints[i]) > maxDistanceThreshold) {
                        continue;
                    }

                    // we tracked the point: keep the map id of the keypoint in the previous frame (low cost feature association)
                    keypointStruct.add(lastKeypoint._id, forwardPoints[keypointIndex]);
                }
            }



            double get_percent_of_elapsed_time(const double treatmentTime, const double totalTimeElapsed) 
            {
                if (totalTimeElapsed <= 0)
                    return 0;
                return std::round(treatmentTime / totalTimeElapsed * 10000.0) / 100.0;
            }

            void Key_Point_Extraction::show_statistics(const double meanFrameTreatmentDuration, const uint frameCount) const {
                if (frameCount > 0) { 
                    const double meanPointExtractionDuration = _meanPointExtractionDuration / static_cast<double>(frameCount);
                    std::cout << "\tMean point extraction time is " << meanPointExtractionDuration << " seconds (" << get_percent_of_elapsed_time(meanPointExtractionDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
                }
            }

        }
    }
}
