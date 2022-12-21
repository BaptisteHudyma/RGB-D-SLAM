#include "rgbd_slam.hpp"

#include <opencv2/core.hpp>

#include "parameters.hpp"
#include "outputs/logger.hpp"
#include "types.hpp"
#include "utils/random.hpp"
#include "utils/matches_containers.hpp"

#include "pose_optimization/pose_optimization.hpp"

namespace rgbd_slam {

    RGBD_SLAM::RGBD_SLAM(const utils::Pose &startPose, const uint imageWidth, const uint imageHeight) :
        _width(imageWidth),
        _height(imageHeight),
        _isTrackingLost(true),

        _totalFrameTreated(0),
        _meanDepthMapTreatmentDuration(0.0),
        _meanPoseOptimizationDuration(0.0),
        _meanPrimitiveTreatmentDuration(0.0),
        _meanLineTreatmentDuration(0.0),
        _meanFindMatchTime(0.0),
        _meanPoseOptimizationFromFeatures(0.0),
        _meanLocalMapUpdateDuration(0.0)
        {
            // set random seed
            const uint seed = utils::Random::_seed;
            outputs::log("Constructed using seed " + std::to_string(seed));
            std::srand(seed);
            cv::theRNG().state = seed;
            
            // Load parameters (once)
            if (not Parameters::is_valid())
            {
                Parameters::load_defaut();
                if (not Parameters::is_valid())
                {
                    outputs::log_error("Invalid default parameters. Check your static parameters configuration");
                    exit(-1);
                }
                outputs::log("Invalid parameters. Switching to default parameters");
            }

            // set threads
            const uint availableCores = Parameters::get_available_core_number();
            cv::setNumThreads(availableCores);
            Eigen::setNbThreads(availableCores);

            // primitive connected graph creator
            _depthOps = new features::primitives::Depth_Map_Transformation(
                    _width, 
                    _height, 
                    Parameters::get_depth_map_patch_size()
                    );
            if (_depthOps == nullptr or not _depthOps->is_ok()) {
                outputs::log_error("Cannot load parameter files, exiting");
                exit(-1);
            }

            //local map
            _localMap = new map_management::Local_Map();

            //plane/cylinder finder
            _primitiveDetector = new features::primitives::Primitive_Detection(
                    _width,
                    _height,
                    Parameters::get_depth_map_patch_size(),
                    Parameters::get_maximum_plane_match_angle(),
                    Parameters::get_maximum_merge_distance()
                    );

            // Point detector and matcher
            _pointDetector = new features::keypoints::Key_Point_Extraction(Parameters::get_maximum_number_of_detectable_features());

            // Line segment detector
            _lineDetector = new features::lines::Line_Detection(0.3, 0.9);

            if (_primitiveDetector == nullptr) {
                outputs::log_error("Instanciation of Primitive_Detector failed");
                exit(-1);
            }
            if (_pointDetector == nullptr) {
                outputs::log_error("Instanciation of Key_Point_Extraction failed");
                exit(-1);
            }
            if (_lineDetector == nullptr) {
                outputs::log_error("Instanciation of Line_Detector failed");
                exit(-1);
            }

            _computeKeypointCount = 0;
            _currentPose = startPose;

            // init motion model
            _motionModel.reset(_currentPose.get_position(), _currentPose.get_orientation_quaternion());
        }

    RGBD_SLAM::~RGBD_SLAM()
    {
        delete _localMap;
        delete _primitiveDetector;
        delete _pointDetector;
        delete _lineDetector;
        delete _depthOps;
    }


    const utils::Pose RGBD_SLAM::track(const cv::Mat& inputRgbImage, const cv::Mat& inputDepthImage, const bool shouldDetectLines) 
    {
        _isTrackingLost = true;
        assert(static_cast<size_t>(inputDepthImage.rows) == _height);
        assert(static_cast<size_t>(inputDepthImage.cols) == _width);
        assert(static_cast<size_t>(inputRgbImage.rows) == _height);
        assert(static_cast<size_t>(inputRgbImage.cols) == _width);

        cv::Mat depthImage = inputDepthImage.clone();

        //project depth image in an organized cloud
        const double depthImageTreatmentStartTime = cv::getTickCount();
        // organized 3D depth image
        matrixf cloudArrayOrganized = matrixf::Zero(_width * _height, 3);
        _depthOps->get_organized_cloud_array(depthImage, cloudArrayOrganized);
        _meanDepthMapTreatmentDuration += (cv::getTickCount() - depthImageTreatmentStartTime) / static_cast<double>(cv::getTickFrequency());
        
        // Compute a gray image for feature extractions
        cv::Mat grayImage;
        cv::cvtColor(inputRgbImage, grayImage, cv::COLOR_BGR2GRAY);

        if(shouldDetectLines) { //detect lines in image

            const double lineDetectionStartTime = cv::getTickCount();
            const features::lines::line_container& detectedLines = _lineDetector->detect_lines(grayImage, depthImage);
            _meanLineTreatmentDuration += (cv::getTickCount() - lineDetectionStartTime) / (double)cv::getTickFrequency();
            
            cv::Mat outImage = inputRgbImage.clone();
            _lineDetector->get_image_with_lines(detectedLines, depthImage, outImage); 

            cv::imshow("line", outImage);
        }

        // this frame points and  assoc
        const double computePoseStartTime = cv::getTickCount();
        const utils::Pose& refinedPose = this->compute_new_pose(grayImage, depthImage, cloudArrayOrganized);
        _meanPoseOptimizationDuration += (cv::getTickCount() - computePoseStartTime) / (double)cv::getTickFrequency();

        //update motion model with refined pose
        _motionModel.update_model(refinedPose);

        // Update current pose
        _currentPose = refinedPose;

        _totalFrameTreated += 1;
        return refinedPose;
    }

    void RGBD_SLAM::get_debug_image(const utils::Pose& camPose, const cv::Mat& originalRGB, cv::Mat& debugImage, const double elapsedTime, const bool shouldDisplayStagedPoints, const bool shouldDisplayPrimitiveMasks) 
    {
        debugImage = originalRGB.clone();

        const uint bandSize = _height / 25.0;   // 1/25 of the total image should be for the top black band

        // Show frame rate and labels
        cv::rectangle(debugImage, cv::Point(0,0), cv::Point(_width, bandSize), cv::Scalar(0,0,0), -1);
        if(elapsedTime > 0) 
        {
            std::stringstream fps;
            fps << round(1 / elapsedTime + 0.5) << " fps";
            cv::putText(debugImage, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));
        }

        _localMap->get_debug_image(camPose, shouldDisplayStagedPoints, shouldDisplayPrimitiveMasks, debugImage);

        // display a red overlay if tracking is lost
        if (_isTrackingLost)
        {
            const cv::Size& debugImageSize = debugImage.size();
            cv::addWeighted(debugImage, 0.8, cv::Mat(debugImageSize, CV_8UC3, cv::Scalar(0, 0, 255)), 0.2, 1, debugImage);
        }
    }


    const utils::Pose RGBD_SLAM::compute_new_pose(const cv::Mat& grayImage, const cv::Mat& depthImage, const matrixf& cloudArrayOrganized) 
    {
        //get a pose with the decaying motion model
        utils::Pose refinedPose = _motionModel.predict_next_pose(_currentPose);

        // every now and then, restart the search of points even if we have enough features
        const bool shouldRecomputeKeypoints = (_computeKeypointCount % Parameters::get_keypoint_refresh_frequency()) == 0;

        // Detect and match key points with local map points
        const features::keypoints::KeypointsWithIdStruct& trackedKeypointContainer = _localMap->get_tracked_keypoints_features(_currentPose);
        const features::keypoints::Keypoint_Handler& keypointObject = _pointDetector->compute_keypoints(grayImage, depthImage, trackedKeypointContainer, shouldRecomputeKeypoints);

        // Run primitive detection 
        const double primitiveDetectionStartTime = cv::getTickCount();
        features::primitives::plane_container detectedPlanes;
        // TODO: handle detected cylinders in local map
        features::primitives::cylinder_container detectedCylinders;
        _primitiveDetector->find_primitives(cloudArrayOrganized, detectedPlanes, detectedCylinders);
        _meanPrimitiveTreatmentDuration += (cv::getTickCount() - primitiveDetectionStartTime) / static_cast<double>(cv::getTickFrequency());

        const double findMatchesStartTime = cv::getTickCount();
        const matches_containers::matchContainer& matchedFeatures = _localMap->find_feature_matches(refinedPose, keypointObject, detectedPlanes);
        _meanFindMatchTime += (cv::getTickCount() - findMatchesStartTime) / static_cast<double>(cv::getTickFrequency());

        matches_containers::match_sets matchSets;
        // only == 0 if this is the first call to this function
        const bool isFirstCall = (_computeKeypointCount == 0);
        if (not isFirstCall)
        {
            // Optimize refined pose
            const double optimizePoseStartTime = cv::getTickCount();
            utils::Pose optimizedPose;
            //if (matchedPoints.size() >= Parameters::get_minimum_point_count_for_optimization())
            const bool isPoseValid = pose_optimization::Pose_Optimization::compute_optimized_pose(refinedPose, matchedFeatures, optimizedPose, matchSets);
            if (isPoseValid)
            {
                refinedPose = optimizedPose;
                _isTrackingLost = false;
            }
            else {
                outputs::log_error("Could not find an optimized pose");
            }
            // else the refined pose will follow the motion model
            _meanPoseOptimizationFromFeatures += (cv::getTickCount() - optimizePoseStartTime) / static_cast<double>(cv::getTickFrequency());
        }
        //else: first call: no optimization

        // if keypoints were re searched, or if tracking is lost, reset the counter
        if (shouldRecomputeKeypoints or _isTrackingLost) {
            // reset the counter to not overflow
            _computeKeypointCount = 0; 
        }
        // if tracking was lost, counter will be 0. Do not increment counter: it should stay at 0 to redetect new points
        if (isFirstCall or not _isTrackingLost)
            _computeKeypointCount += 1;

        const double updateLocalMapStartTime = cv::getTickCount();
        // First call always update the map, tracking lost does not update the map
        if (isFirstCall or not _isTrackingLost)
        {
            // Update local map if a valid transformation was found
            _localMap->update(refinedPose, keypointObject, detectedPlanes, matchSets._pointSets._outliers, matchSets._planeSets._outliers);
        }
        else
        {
            // no valid transformation
            _localMap->update_no_pose();
        }
        _meanLocalMapUpdateDuration += (cv::getTickCount() - updateLocalMapStartTime) / static_cast<double>(cv::getTickFrequency());

        return refinedPose;
    }



    double get_percent_of_elapsed_time(const double treatmentTime, const double totalTimeElapsed) 
    {
        if (totalTimeElapsed <= 0)
            return 0;
        return std::round(treatmentTime / totalTimeElapsed * 10000) / 100;
    }


    void RGBD_SLAM::show_statistics(const double meanFrameTreatmentDuration) const 
    {
        if (_totalFrameTreated > 0)
        {
            const double pointCloudTreatmentDuration = _meanDepthMapTreatmentDuration / _totalFrameTreated;
            std::cout << "Mean image to point cloud treatment duration is " << pointCloudTreatmentDuration << " seconds (" << get_percent_of_elapsed_time(pointCloudTreatmentDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
            const double poseTreatmentDuration = _meanPoseOptimizationDuration / _totalFrameTreated;
            std::cout << "Mean pose estimation duration is " << poseTreatmentDuration << " seconds (" << get_percent_of_elapsed_time(poseTreatmentDuration, meanFrameTreatmentDuration) << "%)" << std::endl;


            std::cout << std::endl;
            std::cout << "Pose optimization profiling details:" << std::endl;
            // display primitive detection statistic
            const double primitiveTreatmentDuration = _meanPrimitiveTreatmentDuration / _totalFrameTreated;
            std::cout << "\tMean primitive treatment duration is " << primitiveTreatmentDuration << " seconds (" << get_percent_of_elapsed_time(primitiveTreatmentDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
            // display line detection statistics
            const double lineDetectionDuration = _meanLineTreatmentDuration / _totalFrameTreated;
            std::cout << "\tMean line detection duration is " << lineDetectionDuration << " seconds (" << get_percent_of_elapsed_time(lineDetectionDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
            // display point detection statistics
            _pointDetector->show_statistics(meanFrameTreatmentDuration, _totalFrameTreated);
            // display find match statistics
            const double findMatchDuration = _meanFindMatchTime / _totalFrameTreated;
            std::cout << "\tMean find match duration is " << findMatchDuration << " seconds (" << get_percent_of_elapsed_time(findMatchDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
            // display pose optimization from features statistics
            const double poseOptimizationDuration = _meanPoseOptimizationFromFeatures / _totalFrameTreated;
            std::cout << "\tMean pose optimization duration is " << poseOptimizationDuration << " seconds (" << get_percent_of_elapsed_time(poseOptimizationDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
            // display local map update statistics
            const double localMapUpdateDuration = _meanLocalMapUpdateDuration / _totalFrameTreated;
            std::cout << "\tMean local map update duration is " << localMapUpdateDuration << " seconds (" << get_percent_of_elapsed_time(localMapUpdateDuration, meanFrameTreatmentDuration) << "%)" << std::endl;
        }
    }


} /* rgbd_slam */

