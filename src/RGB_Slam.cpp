#include "RGB_Slam.hpp"

using namespace poseEstimation;

RGB_SLAM::RGB_SLAM(const Parameters &params) :
    params(params),
    featureHandler(params),
    localMap(params, &featureHandler),
    pnpSolver(params.get_fx(), params.get_fy(), params.get_cx(), params.get_cy(), params.get_baseline())

{
    reset();

    triangulationPolicy = &RGB_SLAM::triangulation_policy_decreasing_matches;
    if(params.get_triangulation_policy() == Parameters::etriangulation_policy_always_triangulate)
        triangulationPolicy = &RGB_SLAM::triangulation_policy_always_triangulate;
    else if(params.get_triangulation_policy() == Parameters::etriangulation_policy_map_size)
        triangulationPolicy = &RGB_SLAM::triangulation_policy_map_size;
}

void RGB_SLAM::reset() {
    this->localMap.reset();
    this->motionModel.reset();

    this->lastPose = Pose();
    this->frameNumber = 0;
    this->lastMatches = std::deque<int>(N_MATCHES_WINDOWS, std::numeric_limits<int>::max());
    this->state = eState_NOT_INITIALIZED;

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
    featureHandler.compute_features(imgRGB, imgDepth, features);

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

    std::cout << features.get_features_count() << " : " << matchesCnt << std::endl;
    if(matchesCnt < this->params.get_min_matches_for_tracking()) {
        //not enough matched features
        std::cout << "Not enough features matched for tracking: only " << matchesCnt << " matches" << std::endl;
        isTracking = false;
        return this->lastPose;
    }
    
    this->lastMatches.push_back(matchesCnt);
    this->lastMatches.pop_front();

    Pose optimizedPose = this->pnpSolver.compute_pose(estimatedPose, features, matchedPoints, matchOutliers);

    //remove untracked points
    this->localMap.clean_untracked_points(features);

    if (params.get_staged_threshold() > 0)
    {
        localMap.update_staged_map_points(optimizedPose, features);
    }

    if (need_new_triangulation())
    {
        localMap.update_with_new_triangulation(optimizedPose, features);
    }
    
    isTracking = true;
    return optimizedPose;
}

bool RGB_SLAM::need_new_triangulation()
{
    return ((this->*triangulationPolicy)());
}

bool RGB_SLAM::triangulation_policy_decreasing_matches()
{
    const float ratio = 0.99;
    for (int i = N_MATCHES_WINDOWS - 1; i > 0; --i)
    {
        if (float(lastMatches[i]) > ratio * float(lastMatches[i - 1]))
        {
            return false;
        }
    }
    return true;
}

bool RGB_SLAM::triangulation_policy_always_triangulate()
{
    return true;
}

bool RGB_SLAM::triangulation_policy_map_size()
{
    return (localMap.get_map_size() < 1000);
}








