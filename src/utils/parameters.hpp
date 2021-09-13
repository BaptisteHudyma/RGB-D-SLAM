#ifndef RGBD_SLAM_PARAMETERS_HPP
#define RGDB_SLAM_PARAMETERS_HPP

#include <string>

namespace rgbd_slam {

    class Parameters
    {
        public:
            static bool parse_file(const std::string& fileName );

            static double get_camera_center_x() { return _cameraCenterX; };
            static double get_camera_center_y() { return _cameraCenterY; };
            static double get_camera_focal_x() { return _cameraFocalX; };
            static double get_camera_focal_y() { return _cameraFocalY; };

            static unsigned int get_minimum_point_count_for_optimization() { return _minimumPointForOptimization; };
            static unsigned int get_maximum_optimization_iterations() { return _maximumOptimizationCall; };
            static double get_point_weight_threshold() { return _pointWeightThreshold; };
            static double get_point_weight_coefficient() { return _pointWeightCoefficient; };
            static double get_point_Hubert_threshold() { return _pointHubertThreshold; };
            static double get_point_error_multiplier() { return _pointErrorMultiplier; };

            static double get_maximum_match_distance() { return _maximumMatchDistance; };
            static unsigned int get_minimum_hessian() { return _detectorMinHessian; };

            static float get_maximum_plane_match_angle() { return _primitiveMaximumCosAngle; };
            static float get_maximum_merge_distance() { return _primitiveMaximumMergeDistance; };
            static unsigned int get_depth_map_patch_size() { return _depthMapPatchSize; };

            static unsigned int get_minimum_plane_seed_count() { return _minimumPlaneSeedCount; };
            static unsigned int get_minimum_cell_activated() { return _minimumCellActivated; };
            static double get_depth_sigma_error() { return _depthSigmaError; };
            static double get_depth_sigma_margin() { return _depthSigmaMargin; };
            static unsigned int get_depth_discontinuity_limit() { return _depthDiscontinuityLimit; };
            static double get_depth_alpha() { return _depthAlpha; };

            static float get_cylinder_ransac_max_distance() { return _cylinderRansacSqrtMaxDistance; };
            static float get_cylinder_ransac_minimm_score() { return _cylinderRansacMinimumScore; };

            // Max unmatched points to consider this map point as lost
            static unsigned int get_maximum_unmatched_before_removal() { return _pointUnmatchedCountToLoose; };
            //Observe a point for N frames to gain max liability
            static unsigned int get_point_age_confidence() { return _pointAgeConfidence; };
            static unsigned int get_point_staged_age_confidence() { return _pointStagedAgeConfidence; };
            // Minimum point liability for the local map
            static double get_minimum_confidence_for_local_map() { return _pointMinimumConfidenceForMap; };

        private:
            // Camera parameters
            inline static double _cameraCenterX;
            inline static double _cameraCenterY;
            inline static double _cameraFocalX;
            inline static double _cameraFocalY;

            // Position optimization
            inline static unsigned int _minimumPointForOptimization;
            inline static unsigned int _maximumOptimizationCall; 
            inline static double _pointWeightThreshold;
            inline static double _pointWeightCoefficient;
            inline static double _pointHubertThreshold;
            inline static double _pointErrorMultiplier;

            // Point Detection & matching
            inline static double _maximumMatchDistance; // Maximum distance between a point and his mach before refusing the match
            inline static unsigned int _detectorMinHessian;

            // Primitive extraction parameters
            inline static float _primitiveMaximumCosAngle;         // Maximum angle between two planes to consider merging
            inline static float _primitiveMaximumMergeDistance;    // Maximum plane patch merge distance
            inline static unsigned int _depthMapPatchSize;         // Size of the minimum search area

            inline static unsigned int _minimumPlaneSeedCount;     // Minimum plane patches in a set to consider merging 
            inline static unsigned int _minimumCellActivated;
            inline static double _depthSigmaError;
            inline static double _depthSigmaMargin;        // [3, 8]
            inline static unsigned int _depthDiscontinuityLimit; // Max number of discontinuities in a cell to reject it
            inline static double _depthAlpha;              // [0.02, 0.04]

            inline static float _cylinderRansacSqrtMaxDistance;
            inline static float _cylinderRansacMinimumScore;

            // local map management
            inline static unsigned int _pointUnmatchedCountToLoose;    // Maximum unmatched times before removal
            inline static unsigned int _pointAgeConfidence;            // Minimum age of a point to consider it good 
            inline static unsigned int _pointStagedAgeConfidence;        // Minimum age of a point in staged map to consider it good 
            inline static double _pointMinimumConfidenceForMap;        // Minimum confidence of a staged point to add it to local map
    };

};

#endif
