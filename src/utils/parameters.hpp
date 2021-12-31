#ifndef RGBD_SLAM_PARAMETERS_HPP
#define RGDB_SLAM_PARAMETERS_HPP

#include <string>

namespace rgbd_slam {

    class Parameters
    {
        public:
            static bool parse_file(const std::string& fileName );
            static void load_defaut();

            static bool is_valid() { return _isValid; };

            static double get_starting_position_x() { return _startingPositionX; };
            static double get_starting_position_y() { return _startingPositionY; };
            static double get_starting_position_z() { return _startingPositionZ; };

            static double get_starting_rotation_x() { return _startingRotationX; };
            static double get_starting_rotation_y() { return _startingRotationY; };
            static double get_starting_rotation_z() { return _startingRotationZ; };

            // Camera 1 is the left camera in stereo, and the color camera in RGBD
            static double get_camera_1_center_x() { return _camera1CenterX; };
            static double get_camera_1_center_y() { return _camera1CenterY; };
            static double get_camera_1_focal_x() { return _camera1FocalX; };
            static double get_camera_1_focal_y() { return _camera1FocalY; };
            // Camera 2 is the right camera in stereo, and the depth camera in RGBD
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

            // Optimisation parameters
            static uint get_minimum_point_count_for_optimization() { return _minimumPointForOptimization; };
            static uint get_maximum_point_count_per_frame() { return _maximumPointPerFrame; };
            static uint get_optimization_maximum_iterations() { return _optimizationMaximumIterations; };
            static double get_optimization_error_precision() { return _optimizationErrorPrecision; };
            static double get_optimization_xtol() { return _optimizationToleranceOfSolutionVectorNorm; };
            static double get_optimization_ftol() { return _optimizationToleranceOfVectorFunction; };
            static double get_optimization_gtol() { return _optimizationToleranceOfErrorFunctionGradient; };
            static double get_optimization_factor() { return _optimizationDiagonalStepBoundShift; };
            static double get_maximum_optimization_retroprojection_error() { return _maximumRetroprojectionError; };
            static size_t get_maximum_optimization_reiteration() { return _maximumRetroprojectionReiteration; };

            static double get_point_weight_threshold() { return _pointWeightThreshold; };
            static double get_point_weight_coefficient() { return _pointWeightCoefficient; };
            static double get_point_loss_alpha() { return _pointLossAlpha; };
            static double get_point_loss_scale() { return _pointLossScale; };
            static double get_point_error_multiplier() { return _pointErrorMultiplier; };

            static double get_search_matches_distance() { return _matchSearchRadius; };
            static double get_search_matches_cell_size() { return _matchSearchCellSize; };
            static double get_maximum_match_distance() { return _maximumMatchDistance; };
            static uint get_minimum_hessian() { return _detectorMinHessian; };
            static uint get_keypoint_refresh_frequency() { return _keypointRefreshFrequency; };
            static uint get_optical_flow_pyramid_depth() { return _opticalFlowPyramidDepth; };
            static uint get_optical_flow_pyramid_windown_size() { return _opticalFlowPyramidWindowSize; };
            static uint get_optical_flow_max_error() { return _opticalFlowMaxError; };
            static uint get_optical_flow_max_distance() { return _opticalFlowMaxDistance; };
            static uint get_keypoint_mask_diameter() { return _keypointMaskDiameter; };

            static float get_maximum_plane_match_angle() { return _primitiveMaximumCosAngle; };
            static float get_maximum_merge_distance() { return _primitiveMaximumMergeDistance; };
            static uint get_depth_map_patch_size() { return _depthMapPatchSize; };

            static uint get_minimum_plane_seed_count() { return _minimumPlaneSeedCount; };
            static uint get_minimum_cell_activated() { return _minimumCellActivated; };
            static double get_depth_sigma_error() { return _depthSigmaError; };
            static double get_depth_sigma_margin() { return _depthSigmaMargin; };
            static uint get_depth_discontinuity_limit() { return _depthDiscontinuityLimit; };
            static double get_depth_alpha() { return _depthAlpha; };

            static float get_cylinder_ransac_max_distance() { return _cylinderRansacSqrtMaxDistance; };
            static float get_cylinder_ransac_minimm_score() { return _cylinderRansacMinimumScore; };

            // Map

            // Max unmatched points to consider this map point as lost
            static uint get_maximum_unmatched_before_removal() { return _pointUnmatchedCountToLoose; };
            //Observe a point for N frames to gain max liability
            static uint get_point_age_confidence() { return _pointAgeConfidence; };
            static uint get_point_staged_age_confidence() { return _pointStagedAgeConfidence; };
            // Minimum point liability for the local map
            static double get_minimum_confidence_for_local_map() { return _pointMinimumConfidenceForMap; };
            static double get_maximum_map_retroprojection_error() { return _mapMaximumRetroprojectionError; };

        private:
            // Is this set of parameters valid
            inline static bool _isValid; 
            
            // Starting position (m & radians)
            inline static double _startingPositionX;
            inline static double _startingPositionY;
            inline static double _startingPositionZ;

            inline static double _startingRotationX;
            inline static double _startingRotationY;
            inline static double _startingRotationZ;

            // Cameras intrinsics parameters
            inline static double _camera1CenterX;
            inline static double _camera1CenterY;
            inline static double _camera1FocalX;
            inline static double _camera1FocalY;

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

            // Position optimization
            inline static uint _minimumPointForOptimization;    // Minimum points to launch optimization
            inline static uint _maximumPointPerFrame;           // maximum points per frame, over which we do not want to detect more points (optimization)

            inline static double _optimizationToleranceOfSolutionVectorNorm;    // tolerance for the norm of the solution vector
            inline static double _optimizationToleranceOfVectorFunction;        // tolerance for the norm of the vector function

            inline static double _optimizationToleranceOfErrorFunctionGradient; // tolerance for the norm of the gradient of the error function
            inline static double _optimizationDiagonalStepBoundShift;           // step bound for the diagonal shift
            inline static double _optimizationErrorPrecision;                   // error precision

            inline static uint _optimizationMaximumIterations;              // Max iteration of the Levenberg Marquart optimisation
            inline static double _maximumRetroprojectionError;              // maximum projection error over which we will remove outliers et restart optimization in pixels
            inline static size_t _maximumRetroprojectionReiteration;        // Maximum times that we will relaunch the optimization process before stopping

            inline static double _pointWeightThreshold;
            inline static double _pointWeightCoefficient;
            inline static double _pointLossAlpha;   // loss steepness (_infinity, infinity)
            inline static double _pointLossScale;   // loss scale (> 0), Unit: Pixels
            inline static double _pointErrorMultiplier; // multiplier of the final loss value (useful when  using primitives along with points)

            // Point Detection & matching
            inline static double _matchSearchRadius;    // Radius of the space around a point to search match points ins
            inline static int _matchSearchCellSize;     // Size of a search space divider 
            inline static double _maximumMatchDistance; // Maximum distance between a point and his mach before refusing the match
            inline static uint _detectorMinHessian;
            inline static uint _keypointRefreshFrequency;
            inline static uint _opticalFlowPyramidDepth;
            inline static uint _opticalFlowPyramidWindowSize;
            inline static uint _opticalFlowMaxError;
            inline static uint _opticalFlowMaxDistance;
            inline static uint _keypointMaskDiameter;

            // Primitive extraction parameters
            inline static float _primitiveMaximumCosAngle;         // Maximum angle between two planes to consider merging
            inline static float _primitiveMaximumMergeDistance;    // Maximum plane patch merge distance
            inline static uint _depthMapPatchSize;         // Size of the minimum search area

            inline static uint _minimumPlaneSeedCount;     // Minimum plane patches in a set to consider merging 
            inline static uint _minimumCellActivated;
            inline static double _depthSigmaError;
            inline static double _depthSigmaMargin;        // [3, 8]
            inline static uint _depthDiscontinuityLimit; // Max number of discontinuities in a cell to reject it
            inline static double _depthAlpha;              // [0.02, 0.04]

            inline static float _cylinderRansacSqrtMaxDistance;
            inline static float _cylinderRansacMinimumScore;

            // local map management
            inline static uint _pointUnmatchedCountToLoose;    // Maximum unmatched times before removal
            inline static uint _pointAgeConfidence;            // Minimum age of a point to consider it good 
            inline static uint _pointStagedAgeConfidence;        // Minimum age of a point in staged map to consider it good 
            inline static double _pointMinimumConfidenceForMap;        // Minimum confidence of a staged point to add it to local map
            inline static double _mapMaximumRetroprojectionError;       // Maximum error between a map point retro projection and the new point position before removing it from the local map (in millimeters)
    };

};

#endif
