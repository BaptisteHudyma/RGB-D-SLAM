#include "RGB_SLAM.hpp"

using namespace poseEstimation;

RGB_SLAM::RGB_SLAM(const Parameters &params) :
    params(params),
    featuresHandler(params),
    localMap(params.get_camera_size(), params.get_point_size()),
    pnpSolver(params.get_fx(), params.get_fy(), params.get_cy(), params.get_baseline()),

{
    reset();
}

void RGB_SLAM::reset() {
    this->localMap.reset();
    this->motionModel.reset();

    this->lastPose = Pose();
    this->frameNumber = 0;
    this->lastMatches = std::deque<int>(N_MATCHES_WINDOWS, std::numeric_limits<int>::max());
    this->eState = eState_NOT_INITIALIZED;

}


/*
 *  Update the current pose from features and motion model
 *
 *  in: imgRGB
 *  in: imgDepth
 *
 * returns the new pose
 */
Pose RGB_SLAM::track(const cv::Mat& imgRGB, const cv::Mat& imgDepth) {
    this->frameNumber += 1;

    if(this->state == eState_LOST) {
        return this->lastPose;
    }

    //In RGBD case, imgRGB must be the grayscale image and imgDepth the depth with format CV_32F
    assert(imgRGB.channels() == 1 and imgDepth.type() == CV_32F);

    Image_Features_Struct features;
    featuresHandler.compute_features_rgbd(imgRGB, imgDepth, &features);

    if(this->state == eState_NOT_INITIALIZED) {
        Pose identityPose;
        this->localMap.update_with_new_triangulation(identityPose, features, true);
        this->state = eState_TRACKING;
        this->lastMatches[0] = this->localMap.get_map_size();
        return identityPose;
    }

    bool isTracking = false;
    Pose predictedPose = this->motionModel.predict_next_pose(this->lastPose);
    Pose computedPose = perform_tracking(predictedPose, features, isTracking);

    if(not isTracking) {
        this->state = eState_LOST;
        return this->lastPose;
    }

    this->lastPose = computedPose;
    return computedPose;
}

/*
 *   Refine the pose estimated from motion model and decide if tracking is lost
 *
 * in estimatedPose Pose estimated from motion model
 * in features Detected features in the image
 * out isTracking Pose estimator not lost
 *
 */
Pose RGB_SLAM::perform_tracking(const Pose& estimatedPose, Image_Features_Struct& features, bool& isTracking) {
    
    vector3_array matchedPoints;    //matched points
    std::vector<int> matchOutliers; //unmatched points
    
    this->localMap.find_matches(estimatedPose, features, matchedPoints, matchOutliers);
    const int matchesCnt = matchedPoints.size();

    if(matchesCnt < this->params.get_min_matches_for_tracking()) {
        //not enough matched features
        isTracking = false;
        return this->lastPose;
    }
    
    this->lastMatches.push_back(matchesCnt);
    this->lastMatches.pop_front();

    Pose optimizedPose = this->pnpSolver.compute_pose(estimatedPose, features, matchedPoints, matchOutliers);

    //remove untracked points
    this->localMap.clean_untracked_points(features);
    
    isTracking = true;
    return optimizedPose;
}






