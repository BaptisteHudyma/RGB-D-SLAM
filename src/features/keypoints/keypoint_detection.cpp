#include "keypoint_detection.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"

// circle
#include <array>
#include <cmath>
#include <mutex>
#include <tbb/parallel_for.h>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace rgbd_slam::features::keypoints {

/*
 * Keypoint extraction
 */

Key_Point_Extraction::Key_Point_Extraction()
{
    static const size_t imageHeight = Parameters::get_camera_1_image_size().y();
    static const size_t imageWidth = Parameters::get_camera_1_image_size().x();

    static constexpr size_t numCellsY = parameters::detection::keypointCellDetectionHeightCount;
    static constexpr size_t numCellsX = parameters::detection::keypointCellDetectionWidthCount;

    static const size_t cellSizeY = imageHeight / numCellsY;
    static const size_t cellSizeX = imageWidth / numCellsX;

    // Create feature extractor and matcher
#ifdef USE_ORB_DETECTOR_AND_MATCHING
    const int detectorThreshold =
            std::max(1, static_cast<int>(parameters::detection::maximumPointPerFrame / numberOfDetectionCells));
    for (size_t i = 0; i < numberOfDetectionCells; ++i)
    {
        _featureDetectors[i] = cv::Ptr<cv::FeatureDetector>(cv::ORB::create(detectorThreshold));
        _advancedFeatureDetectors[i] = cv::Ptr<cv::FeatureDetector>(cv::ORB::create(2 * detectorThreshold));
        assert(not _featureDetectors[i].empty());
        assert(not _advancedFeatureDetectors[i].empty());
    }

    _featureDescriptor = _featureDetectors[0];
#else

    // the parameters for this threshold is based on measurments on the FAST detector thresholds
    static auto get_fast_threshold = [](double pointsToDetects) {
        return 41.2378 * pow(0.99945, pointsToDetects);
    };

    // put more points to detect in the threshold, not expensive at all, gives better results
    static const int detectorThreshold =
            static_cast<int>(ceil(get_fast_threshold(10.0 * (double)parameters::detection::maximumPointPerFrame)));
    static const int advanceDetectorThreshold =
            static_cast<int>(ceil(get_fast_threshold(30.0 * (double)parameters::detection::maximumPointPerFrame)));

    for (size_t i = 0; i < numberOfDetectionCells; ++i)
    {
        _featureDetectors[i] = cv::Ptr<cv::FeatureDetector>(cv::FastFeatureDetector::create(detectorThreshold));
        _advancedFeatureDetectors[i] = cv::FastFeatureDetector::create(advanceDetectorThreshold);
        assert(not _featureDetectors[i].empty());
        assert(not _advancedFeatureDetectors[i].empty());
    }

    _featureDescriptor = cv::Ptr<cv::DescriptorExtractor>(cv::xfeatures2d::BriefDescriptorExtractor::create());
    assert(not _featureDescriptor.empty());
#endif

    // create the detection windows
    size_t i = 0;
    for (size_t cellYIndex = 0; cellYIndex < numCellsY; ++cellYIndex)
    {
        const int maxYIndex = static_cast<int>(cellYIndex * cellSizeY);
        for (size_t cellXIndex = 0; cellXIndex < numCellsX; ++cellXIndex, ++i)
        {
            _detectionWindows[i] = cv::Rect(static_cast<int>(cellXIndex * cellSizeX),
                                            maxYIndex,
                                            static_cast<int>(cellSizeX),
                                            static_cast<int>(cellSizeY));
        }
    }
}

std::vector<cv::Point2f> Key_Point_Extraction::detect_keypoints(
        const cv::Mat& grayImage, const std::vector<cv::Point2f>& alreadyDetectedPoints) const noexcept
{
    // search keypoints, using an advanced detector if not enough features are found
    std::vector<cv::KeyPoint> frameKeypoints;
    perform_keypoint_detection(grayImage, alreadyDetectedPoints, frameKeypoints);

    // compute a subcorner accuracy estimation for all waypoints
    std::vector<cv::Point2f> framePoints;
    if (not frameKeypoints.empty())
    {
        // Refine keypoints positions
        framePoints.reserve(frameKeypoints.size());
        cv::KeyPoint::convert(frameKeypoints, framePoints);

        const static cv::Size winSize = cv::Size(3, 3);
        const static cv::Size zeroZone = cv::Size(-1, -1);
        const static cv::TermCriteria termCriteria =
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.01);
        cv::cornerSubPix(grayImage, framePoints, winSize, zeroZone, termCriteria);
    }

    return framePoints;
}

cv::Mat_<uchar> Key_Point_Extraction::compute_key_point_mask(
        const cv::Size imageSize,
        const std::vector<cv::Point2f>& keypointContainer,
        const double pointMaskRadius_px,
        std::array<uint16_t, numberOfDetectionCells>& detectionWindowDetectionCount) const noexcept
{
    const int radiusOfAreaAroundPoint = static_cast<int>(pointMaskRadius_px); // in pixels
    const static cv::Scalar fillColor(0);

    // this intance is a small optimisation with a static and a reset to white each time
    static cv::Mat_<uchar> mask(imageSize);
    mask.setTo(255); // set all to white (no mask)

    detectionWindowDetectionCount.fill(0);
    for (const cv::Point2f& point: keypointContainer)
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

        // get the index of the associated detection window
        for (size_t i = 0; i < _detectionWindows.size(); ++i)
        {
            const auto& detectionWindow = _detectionWindows[i];
            if (detectionWindow.contains(point))
            {
                // add a point detection to this windows
                detectionWindowDetectionCount[i]++;
                break;
            }
        }
        // it is possible that this loop finishes with no update to detectionWindowDetectionCount.
        // It happens if the point was detected outside of the search windows.
        // They can miss some pixels because the size of the camera is not divisible by the span of windows
        // It can causes a small band of non detections in the image, negligeable
    }

    // i'm not a fan of returning a local static variable, but hey it works...
    return mask;
}

Keypoint_Handler Key_Point_Extraction::compute_keypoints(const cv::Mat& grayImage,
                                                         const cv::Mat_<float>& depthImage,
                                                         const KeypointsWithIdStruct& lastKeypointsWithIds,
                                                         const bool forceKeypointDetection) noexcept
{
    KeypointsWithIdStruct newKeypointsObject;

    // detect keypoints
    const int64 keypointDetectionStartTime = cv::getTickCount();

    /*
     * OPTICAL FLOW
     */

    // load parameters
    constexpr int pyramidDepth = static_cast<int>(parameters::detection::opticalFlowPyramidDepth);
    constexpr double maxDistance = parameters::matching::matchSearchRadius_px;
    constexpr double maximumMatchDistance = parameters::matching::maximumMatchDistance;

    static const cv::Size pyramidSize(static_cast<int>(Parameters::get_camera_1_image_size().x() /
                                                       parameters::detection::opticalFlowPyramidWindowSizeWidthCount),
                                      static_cast<int>(Parameters::get_camera_1_image_size().y() /
                                                       parameters::detection::opticalFlowPyramidWindowSizeHeightCount));

    // build pyramid
    std::vector<cv::Mat> newImagePyramide;
    cv::buildOpticalFlowPyramid(grayImage, newImagePyramide, pyramidSize, pyramidDepth);
    // TODO: when the optical flow will not show so much drift, maybe we could remove the tracked keypoint
    // redetection
    if (not _lastFramePyramide.empty() and not lastKeypointsWithIds.empty())
    {
        const auto opticalFlowStartTime = cv::getTickCount();
        get_keypoints_from_optical_flow(_lastFramePyramide,
                                        newImagePyramide,
                                        lastKeypointsWithIds,
                                        pyramidDepth,
                                        pyramidSize,
                                        maxDistance,
                                        newKeypointsObject);
        _meanPointOpticalFlowTrackingDuration +=
                static_cast<double>(cv::getTickCount() - opticalFlowStartTime) / cv::getTickFrequency();
    }
    // else: No optical flow for the first frame or no optical flow for this frame
    _lastFramePyramide = newImagePyramide;

    const size_t opticalFlowTrackedPointCount = newKeypointsObject.size();

    /*
     * KEY POINT DETECTION
     *      Use keypoint detection when low on keypoints, or when requested
     */

    // detect keypoint if: it is requested OR not enough points were detected
    std::vector<cv::Point2f> detectedKeypoints;
    cv::Mat keypointDescriptors;
    if (forceKeypointDetection or opticalFlowTrackedPointCount < parameters::detection::maximumPointPerFrame)
    {
        const auto pointDetectionStartTime = cv::getTickCount();

        // get new keypoints
        detectedKeypoints = detect_keypoints(grayImage, newKeypointsObject.get_keypoints());
        _meanPointDetectionDuration +=
                static_cast<double>(cv::getTickCount() - pointDetectionStartTime) / cv::getTickFrequency();

        if (not detectedKeypoints.empty())
        {
            const auto pointDescriptorsStartTime = cv::getTickCount();
            /**
             *  DESCRIPTORS
             */
            std::vector<cv::KeyPoint> frameKeypoints;
            cv::KeyPoint::convert(detectedKeypoints, frameKeypoints);

            // Compute descriptors
            // Caution: the frameKeypoints list is mutable by this function
            //          The bad points will be removed by the compute descriptor function
            cv::Mat detectedKeypointDescriptors;
            assert(_featureDescriptor != nullptr);
            _featureDescriptor->compute(grayImage, frameKeypoints, detectedKeypointDescriptors);

            // convert back to keypoint list
            detectedKeypoints.clear();
            cv::KeyPoint::convert(frameKeypoints, detectedKeypoints);

            if (keypointDescriptors.rows > 0)
                cv::vconcat(detectedKeypointDescriptors, keypointDescriptors, keypointDescriptors);
            else
                keypointDescriptors = detectedKeypointDescriptors;

            _meanPointDescriptorDuration +=
                    static_cast<double>(cv::getTickCount() - pointDescriptorsStartTime) / cv::getTickFrequency();
        }
    }

    // declare static
    static Keypoint_Handler keypointHandler(depthImage.cols, depthImage.rows, maximumMatchDistance);

    // Update last keypoint struct
    keypointHandler.set(detectedKeypoints, keypointDescriptors, newKeypointsObject, depthImage);
    _meanPointExtractionDuration +=
            static_cast<double>(cv::getTickCount() - keypointDetectionStartTime) / cv::getTickFrequency();
    return keypointHandler;
}

void Key_Point_Extraction::get_keypoints_from_optical_flow(const std::vector<cv::Mat>& imagePreviousPyramide,
                                                           const std::vector<cv::Mat>& imageCurrentPyramide,
                                                           const KeypointsWithIdStruct& lastKeypointsWithIds,
                                                           const uint pyramidDepth,
                                                           const cv::Size& windowSizeObject,
                                                           const double maxDistanceThreshold,
                                                           KeypointsWithIdStruct& keypointStruct) noexcept
{
    // START of optical flow
    if (imagePreviousPyramide.empty() or imageCurrentPyramide.empty() or lastKeypointsWithIds.empty())
    {
        outputs::log_error("OpticalFlow: invalid parameters");
        return;
    }

    // Calculate optical flow
    std::vector<uchar> statusContainer;
    std::vector<float> errorContainer;
    std::vector<cv::Point2f> forwardPoints;

    const size_t previousKeyPointCount = lastKeypointsWithIds.size();
    const static cv::TermCriteria criteria =
            cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    // Get forward points: optical flow from previous to current image to extract new keypoints
    cv::calcOpticalFlowPyrLK(imagePreviousPyramide,
                             imageCurrentPyramide,
                             lastKeypointsWithIds.get_keypoints(),
                             forwardPoints,
                             statusContainer,
                             errorContainer,
                             windowSizeObject,
                             static_cast<int>(pyramidDepth),
                             criteria);

    std::vector<size_t> keypointIndexContainer; // Contains the ids of the good waypoints
    std::vector<cv::Point2f> newKeypoints;      // Set output structure
    keypointIndexContainer.reserve(previousKeyPointCount);
    newKeypoints.reserve(previousKeyPointCount);

    // Remove outliers from current waypoint list by creating a new one
    for (size_t keypointIndex = 0; keypointIndex < previousKeyPointCount; ++keypointIndex)
    {
        if (statusContainer[keypointIndex] != 1)
        { // point was not associated or error is too great
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
    cv::calcOpticalFlowPyrLK(imageCurrentPyramide,
                             imagePreviousPyramide,
                             newKeypoints,
                             backwardKeypoints,
                             statusContainer,
                             errorContainer,
                             windowSizeObject,
                             static_cast<int>(pyramidDepth),
                             criteria);

    // mark outliers as false and visualize
    const size_t keypointSize = backwardKeypoints.size();
    keypointStruct.reserve(keypointSize);
    for (size_t i = 0; i < keypointSize; ++i)
    {
        if (statusContainer[i] != 1)
        {
            continue;
        }

        const size_t keypointIndex = keypointIndexContainer[i];
        const KeypointsWithIdStruct::keypointWithId& lastKeypoint = lastKeypointsWithIds.at(keypointIndex);
        // check distance of the backpropagated point to the original point
        if (cv::norm(lastKeypoint._point - backwardKeypoints[i]) > maxDistanceThreshold)
        {
            continue;
        }

        // we tracked the point: keep the map id of the keypoint in the previous frame (low cost feature
        // association)
        keypointStruct.add(lastKeypoint._id, forwardPoints[keypointIndex]);
    }
}

void Key_Point_Extraction::show_statistics(const double meanFrameTreatmentDuration,
                                           const uint frameCount,
                                           const bool shouldDisplayDetails) const noexcept
{
    static auto get_percent_of_elapsed_time = [](double treatmentTime, double totalTimeElapsed) {
        if (totalTimeElapsed <= 0)
            return 0.0;
        return (treatmentTime / totalTimeElapsed) * 100.0;
    };

    if (frameCount > 0)
    {
        const double meanPointExtractionDuration = _meanPointExtractionDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean point extraction time is {:.4f} seconds ({:.2f}%)",
                                 meanPointExtractionDuration,
                                 get_percent_of_elapsed_time(meanPointExtractionDuration, meanFrameTreatmentDuration)));

        if (shouldDisplayDetails)
        {
            const double meanPointOpticalFlowDuration =
                    _meanPointOpticalFlowTrackingDuration / static_cast<double>(frameCount);
            outputs::log(std::format(
                    "\t\tMean point optical flow time is {:.4f} seconds ({:.2f}%)",
                    meanPointOpticalFlowDuration,
                    get_percent_of_elapsed_time(meanPointOpticalFlowDuration, meanPointExtractionDuration)));

            const double meanPointDetectionDuration = _meanPointDetectionDuration / static_cast<double>(frameCount);
            outputs::log(
                    std::format("\t\tMean point detection time is {:.4f} seconds ({:.2f}%)",
                                meanPointDetectionDuration,
                                get_percent_of_elapsed_time(meanPointDetectionDuration, meanPointExtractionDuration)));

            const double meanPointDescriptorDuration = _meanPointDescriptorDuration / static_cast<double>(frameCount);
            outputs::log(
                    std::format("\t\tMean point descriptor time is {:.4f} seconds ({:.2f}%)",
                                meanPointDescriptorDuration,
                                get_percent_of_elapsed_time(meanPointDescriptorDuration, meanPointExtractionDuration)));
        }
    }
}

void Key_Point_Extraction::perform_keypoint_detection(const cv::Mat& grayImage,
                                                      const std::vector<cv::Point2f>& alreadyDetectedPoints,
                                                      std::vector<cv::KeyPoint>& frameKeypoints) const noexcept
{
    frameKeypoints.clear();
    std::mutex mut;

    // create a mask around already detected points location
    std::array<uint16_t, numberOfDetectionCells> detectionWindowDetectionCount;
    const cv::Mat_<uchar>& keypointMask = compute_key_point_mask(grayImage.size(),
                                                                 alreadyDetectedPoints,
                                                                 parameters::detection::trackedMaskRadius_px,
                                                                 detectionWindowDetectionCount);

    constexpr size_t maxKeypointToDetectByCell =
            parameters::detection::maximumPointPerFrame / (parameters::detection::keypointCellDetectionHeightCount *
                                                           parameters::detection::keypointCellDetectionWidthCount);
    frameKeypoints.reserve(parameters::detection::maximumPointPerFrame);

#ifndef MAKE_DETERMINISTIC
    tbb::parallel_for(size_t(0),
                      static_cast<size_t>(numberOfDetectionCells),
                      [this, &grayImage, &keypointMask, &detectionWindowDetectionCount, &frameKeypoints, &mut](size_t i)
#else
    for (size_t i = 0; i < numberOfDetectionCells; ++i)
#endif
                      {
                          const uint16_t alreadyDetectedCount = detectionWindowDetectionCount[i];
                          // already enough points, no need to redetect
                          if (alreadyDetectedCount < maxKeypointToDetectByCell)
                          {
                              // new max points to detect
                              const uint16_t maxKeyPointToDetectHere =
                                      std::max(0, static_cast<int>(maxKeypointToDetectByCell) - alreadyDetectedCount);

                              const auto& detectionWindow = _detectionWindows[i];
                              assert(!detectionWindow.empty());

                              const cv::Mat& subImg = grayImage(detectionWindow);
                              const cv::Mat_<uchar>& subMask = keypointMask(detectionWindow);

                              std::vector<cv::KeyPoint> keypoints;
                              keypoints.reserve(maxKeypointToDetectByCell);

                              assert(!_featureDetectors[i].empty());
                              _featureDetectors[i]->detect(subImg, keypoints, subMask);

                              // Not enough keypoints detected: restart with a more precise detector
                              if (keypoints.size() < maxKeyPointToDetectHere)
                              {
                                  keypoints.clear();
                                  assert(!_advancedFeatureDetectors[i].empty());
                                  _advancedFeatureDetectors[i]->detect(subImg, keypoints, subMask);
                              }

                              // filter the keypoints by score, if we have too much
                              cv::KeyPointsFilter::retainBest(keypoints, maxKeyPointToDetectHere);
                              for (cv::KeyPoint& keypoint: keypoints)
                              {
                                  keypoint.pt.x += (float)detectionWindow.x;
                                  keypoint.pt.y += (float)detectionWindow.y;
                              }
                              std::scoped_lock<std::mutex> lock(mut);
                              frameKeypoints.insert(frameKeypoints.end(), keypoints.begin(), keypoints.end());
                          }
                      }
#ifndef MAKE_DETERMINISTIC
    );
#endif
    cv::KeyPointsFilter::removeDuplicated(frameKeypoints);
}

} // namespace rgbd_slam::features::keypoints
