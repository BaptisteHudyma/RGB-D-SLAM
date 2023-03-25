#ifndef RGBDSLAM_PARAMETERS_HPP
#define RGBDSLAM_PARAMETERS_HPP

#include <string>

namespace rgbd_slam {

/**
 * \brief Store all parameters of this SLAM program.
 * It should be used as a static class everywhere in the program.
 * If the parameters are not loaded from a file, a set of default parameters will be used.
 */
class Parameters
{
  public:
    /**
     * \brief Parse a yaml configuration file and load the parameters. Sets the default parameters
     */
    static bool parse_file(const std::string& fileName);

    /**
     * \brief Set the default camera parameters
     */
    static void load_defaut();

    /**
     * \brief set the global parameters
     */
    static void set_parameters();

    static bool is_valid() { return _isValid; };

    static uint get_available_core_number() { return _coreNumber; };

    // Camera 1 is the left camera in stereo, and the color camera in RGBD
    static uint get_camera_1_size_x() { return _camera1SizeX; };
    static uint get_camera_1_size_y() { return _camera1SizeY; };
    static double get_camera_1_center_x() { return _camera1CenterX; };
    static double get_camera_1_center_y() { return _camera1CenterY; };
    static double get_camera_1_focal_x() { return _camera1FocalX; };
    static double get_camera_1_focal_y() { return _camera1FocalY; };
    // Camera 2 is the right camera in stereo, and the depth camera in RGBD
    static uint get_camera_2_size_x() { return _camera2SizeX; };
    static uint get_camera_2_size_y() { return _camera2SizeY; };
    static double get_camera_2_center_x() { return _camera2CenterX; };
    static double get_camera_2_center_y() { return _camera2CenterY; };
    static double get_camera_2_focal_x() { return _camera2FocalX; };
    static double get_camera_2_focal_y() { return _camera2FocalY; };

    static double get_camera_2_translation_x() { return _camera2TranslationX; };
    static double get_camera_2_translation_y() { return _camera2TranslationY; };
    static double get_camera_2_translation_z() { return _camera2TranslationZ; };

    static double get_camera_2_rotation_x() { return _camera2RotationX; };
    static double get_camera_2_rotation_y() { return _camera2RotationY; };
    static double get_camera_2_rotation_z() { return _camera2RotationZ; };

    // Primitives matching
    static double get_minimum_iou_for_match() { return _minimumIOUToConsiderMatch; };
    static double get_maximum_plane_normals_angle_for_match() { return _maximumAngleForPlaneMatch; };

    // Optimisation parameters
    static double get_ransac_maximum_retroprojection_error_for_point_inliers()
    {
        return _ransacMaximumRetroprojectionErrorForPointInliers;
    };
    static double get_ransac_maximum_retroprojection_error_for_plane_inliers()
    {
        return _ransacMaximumRetroprojectionErrorForPlaneInliers;
    };
    static double get_ransac_minimum_inliers_proportion_for_early_stop()
    {
        return _ransacMinimumInliersProportionForEarlyStop;
    };
    static double get_ransac_probability_of_success() { return _ransacProbabilityOfSuccess; };
    static double get_ransac_inlier_proportion() { return _ransacInlierProportion; };

    static uint get_minimum_point_count_for_optimization() { return _minimumPointForOptimization; };
    static uint get_minimum_plane_count_for_optimization() { return _minimumPlanesForOptimization; };
    static uint get_maximum_point_count_per_frame() { return _maximumPointPerFrame; };
    static uint get_optimization_maximum_iterations() { return _optimizationMaximumIterations; };
    static double get_optimization_error_precision() { return _optimizationErrorPrecision; };
    static double get_optimization_xtol() { return _optimizationToleranceOfSolutionVectorNorm; };
    static double get_optimization_ftol() { return _optimizationToleranceOfVectorFunction; };
    static double get_optimization_gtol() { return _optimizationToleranceOfErrorFunctionGradient; };
    static double get_optimization_factor() { return _optimizationDiagonalStepBoundShift; };
    static double get_maximum_retroprojection_error() { return _maximumRetroprojectionError; };

    static double get_search_matches_distance() { return _matchSearchRadius; };
    static double get_maximum_match_distance() { return _maximumMatchDistance; };
    static uint get_maximum_number_of_detectable_features() { return _maxNumberOfPointsToDetect; };
    static uint get_keypoint_detection_cell_size() { return _keypointCellDetectionSize; };
    static uint get_keypoint_refresh_frequency() { return _keypointRefreshFrequency; };
    static uint get_optical_flow_pyramid_depth() { return _opticalFlowPyramidDepth; };
    static uint get_optical_flow_pyramid_window_size() { return _opticalFlowPyramidWindowSize; };

    static float get_maximum_plane_merge_angle() { return _maximumPlaneAngleForMerge; };
    static uint get_depth_map_patch_size() { return _depthMapPatchSize; };

    static double get_minimum_plane_seed_proportion() { return _minimumPlaneSeedProportion; };
    static double get_minimum_cell_activated_proportion() { return _minimumCellActivatedProportion; };
    static float get_minimum_zero_depth_proportion() { return _minimumZeroDepthProportion; };
    static double get_depth_sigma_error() { return _depthSigmaError; };
    static double get_depth_sigma_multiplier() { return _depthSigmaMultiplier; };
    static double get_depth_sigma_margin() { return _depthSigmaMargin; };

    static float get_cylinder_ransac_max_distance() { return _cylinderRansacSqrtMaxDistance; };
    static float get_cylinder_ransac_minimum_score() { return _cylinderRansacMinimumScore; };
    static float get_cylinder_ransac_inlier_proportion() { return _cylinderRansacInlierProportions; };
    static float get_cylinder_ransac_probability_of_success() { return _cylinderRansacProbabilityOfSuccess; };

    // Map

    // Max unmatched points to consider this map point as lost
    static uint get_maximum_unmatched_before_removal() { return _pointUnmatchedCountToLoose; };
    // Observe a point for N frames to gain max liability
    static uint get_point_staged_age_confidence() { return _pointStagedAgeConfidence; };
    // Minimum point liability for the local map
    static double get_minimum_confidence_for_local_map() { return _pointMinimumConfidenceForMap; };

  private:
    // Is this set of parameters valid
    inline static bool _isValid = false;

    inline static uint _coreNumber; // number of available cores on the computer (1 for no threads)

    // Cameras intrinsics parameters
    inline static uint _camera1SizeX;
    inline static uint _camera1SizeY;
    inline static double _camera1CenterX;
    inline static double _camera1CenterY;
    inline static double _camera1FocalX;
    inline static double _camera1FocalY;

    inline static uint _camera2SizeX;
    inline static uint _camera2SizeY;
    inline static double _camera2CenterX;
    inline static double _camera2CenterY;
    inline static double _camera2FocalX;
    inline static double _camera2FocalY;

    // Camera 2 position and rotation
    inline static double _camera2TranslationX;
    inline static double _camera2TranslationY;
    inline static double _camera2TranslationZ;

    inline static double _camera2RotationX;
    inline static double _camera2RotationY;
    inline static double _camera2RotationZ;

    // primitive matching
    inline static double
            _minimumIOUToConsiderMatch; // Inter over Union of the two primitive masks, to consider a primitive match
    inline static double _maximumAngleForPlaneMatch; // Maximum angle between two primitives to consider a match

    // Position optimization
    inline static uint _minimumPointForOptimization;  // Minimum points to launch optimization
    inline static uint _minimumPlanesForOptimization; // Minimum planes to launch optimization
    inline static uint _maximumPointPerFrame; // maximum points per frame, over which we do not want to detect more
                                              // points (optimization)

    inline static double _ransacMaximumRetroprojectionErrorForPointInliers; // Maximum retroprojection error in pixels
                                                                            // to consider a point match as inlier
    inline static double _ransacMaximumRetroprojectionErrorForPlaneInliers; // Maximum retroprojection error in pixels
                                                                            // to consider a plane match as inlier
    inline static double
            _ransacMinimumInliersProportionForEarlyStop; // Proportion of inliers to consider that a transformation is
                                                         // good enough to stop optimization
    inline static double _ransacProbabilityOfSuccess; // Probability that the RANSAC process finds a good transformation
    inline static double _ransacInlierProportion;     // Proportion of inliers in original set

    inline static double _optimizationToleranceOfSolutionVectorNorm; // tolerance for the norm of the solution vector
    inline static double _optimizationToleranceOfVectorFunction;     // tolerance for the norm of the vector function

    inline static double _optimizationToleranceOfErrorFunctionGradient; // tolerance for the norm of the gradient of the
                                                                        // error function
    inline static double _optimizationDiagonalStepBoundShift;           // step bound for the diagonal shift
    inline static double _optimizationErrorPrecision;                   // error precision

    inline static uint _optimizationMaximumIterations; // Max iteration of the Levenberg Marquart optimisation
    inline static double _maximumRetroprojectionError; // In pixel: maximum distance after which we can consider a
                                                       // retroprojection as invalid

    // Point Detection & matching
    inline static double _matchSearchRadius; // Radius of the space around a point to search match points in pixels
    inline static double
            _maximumMatchDistance; // Maximum distance between a point and his mach before refusing the match
    inline static uint _maxNumberOfPointsToDetect; // point detector sensitivity
    inline static uint _keypointCellDetectionSize; // in pixel, the size of the keypoint detection window
    inline static uint _keypointRefreshFrequency;
    inline static uint _opticalFlowPyramidDepth;      // depth of the pyramid for optical flow seach (0 based)
    inline static uint _opticalFlowPyramidWindowSize; // search size at each pyramid level (pixel)
    inline static uint _opticalFlowMaxDistance; // max retro optical flow distance, after which the point is rejected
    inline static uint _keypointMaskRadius;     // radius around optical flow points in which we cont detect new points

    // Primitive extraction parameters
    inline static float _maximumPlaneAngleForMerge; // Maximum angle between two planes patches to consider merging
    inline static uint _depthMapPatchSize;          // Size of the minimum search area

    inline static double _minimumPlaneSeedProportion;     // Minimum plane patches proportion (of th total of planar
                                                          // patches) in a set to consider merging
    inline static double _minimumCellActivatedProportion; // Minimum activated plane patches proportion (of th total of
                                                          // planar patches) in a set to consider merging
    inline static float
            _minimumZeroDepthProportion; // proportion of invalid depth pixels in a planar patch to reject it
    inline static double _depthSigmaError;
    inline static double _depthSigmaMultiplier;
    inline static double _depthSigmaMargin; // [3, 8]

    inline static float _cylinderRansacSqrtMaxDistance;
    inline static float _cylinderRansacMinimumScore;
    inline static float _cylinderRansacInlierProportions;
    inline static float _cylinderRansacProbabilityOfSuccess;

    // local map management
    inline static uint _pointUnmatchedCountToLoose;     // Maximum unmatched times before removal
    inline static uint _pointStagedAgeConfidence;       // Minimum age of a point in staged map to consider it good
    inline static double _pointMinimumConfidenceForMap; // Minimum confidence of a staged point to add it to local map

    /**
     * \brief Update the _isValid attribute
     */
    static void check_parameters_validity();
};

}; // namespace rgbd_slam

#endif
