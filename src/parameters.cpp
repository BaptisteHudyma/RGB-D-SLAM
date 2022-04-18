#include "parameters.hpp"

#include <opencv2/core/core.hpp>
#include <math.h>
#include <cfloat>

#include "logger.hpp"

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        // set the global parameters
        set_parameters();

        cv::FileStorage configFile(fileName, cv::FileStorage::READ);
        if (not configFile.isOpened())
        {
            utils::log_error("Cannot load parameter files, starting with default configuration");
            load_defaut();
            check_parameters_validity();
            return false;
        }
        // Load start pose
        _startingPositionX = configFile["starting_position_x"];
        _startingPositionY = configFile["starting_position_y"];
        _startingPositionZ = configFile["starting_position_z"];
        _startingRotationX = configFile["starting_rotation_x"];
        _startingRotationY = configFile["starting_rotation_y"];
        _startingRotationZ = configFile["starting_rotation_z"];

        // Load Camera 1 parameters
        _camera1FocalX = configFile["camera_1_focal_x"];
        _camera1FocalY = configFile["camera_1_focal_y"];
        _camera1CenterX = configFile["camera_1_center_x"];
        _camera1CenterY = configFile["camera_1_center_y"];

        // Load camera 2 parameters
        _camera2FocalX = configFile["camera_2_focal_x"];
        _camera2FocalY = configFile["camera_2_focal_y"];
        _camera2CenterX = configFile["camera_2_center_x"];
        _camera2CenterY = configFile["camera_2_center_y"];

        // Load camera offsets of camera 2, relative to camera 1
        _camera2TranslationX = configFile["camera_2_translation_offset_x"];
        _camera2TranslationY = configFile["camera_2_translation_offset_y"];
        _camera2TranslationZ = configFile["camera_2_translation_offset_z"];
        _camera2RotationX = configFile["camera_2_rotation_offset_x"];
        _camera2RotationY = configFile["camera_2_rotation_offset_y"];
        _camera2RotationZ = configFile["camera_2_rotation_offset_z"];

        check_parameters_validity();

        configFile.release();
        return _isValid;
    }

    void Parameters::load_defaut() 
    {
        // set the global parameters
        set_parameters();

        // Initial position & rotation, if necessary
        _startingPositionX = 0;
        _startingPositionY = 0;
        _startingPositionZ = 0;

        _startingRotationX = 0;
        _startingRotationY = 0;
        _startingRotationZ = 0;

        // Camera intrinsic parameters
        _camera1FocalX = 548.86723733696215;
        _camera1FocalY = 549.58402532237187;
        _camera1CenterX = 316.49655835885483;
        _camera1CenterY = 229.23873484682150;

        _camera2FocalX = 575.92685448804468;
        _camera2FocalY = 576.40791601093247;
        _camera2CenterX = 315.15026356388171;
        _camera2CenterY = 230.58580662101753;

        // Camera 2 position & rotation
        _camera2TranslationX = 0;
        _camera2TranslationY = 0;
        _camera2TranslationZ = 0;

        _camera2RotationX = 0; 
        _camera2RotationY = 0; 
        _camera2RotationZ = 0; 
    }

    void Parameters::set_parameters()
    {
        // Point detection/Matching
        _matchSearchRadius = 30;
        _matchSearchCellSize = 50;
        _maximumMatchDistance = 0.7;    // The closer to 0, the more discriminating
        _detectorMinHessian = 40;       // The higher the least detected points
        _keypointRefreshFrequency = 5;  // Update the keypoint list every N calls
        _opticalFlowPyramidDepth = 5;   // depth of the optical pyramid
        _opticalFlowPyramidWindowSize = 25;
        _opticalFlowMaxError = 35;      // error in pixel after which a point is rejected
        _opticalFlowMaxDistance = 100;  // distance in pixel after which a point is rejected
        _keypointMaskDiameter = 10;     // do not detect points inside an area of this size (pixels) around existing keypoints

        // Pose Optimization
        _ransacMaximumRetroprojectionErrorForInliers = 20;   // Retroprojection error between two screen points, in pixels
        _ransacMinimumInliersProportionForEarlyStop = 0.90; // proportion of inliers in total set, to stop RANSAC early
        _ransacProbabilityOfSuccess = 0.8;   // probability of having at least one correct transformation
        _ransacInlierProportion = 0.6;       // number of inliers in data / number of points in data 

        _minimumPointForOptimization = 6;   // Should be >= 6
        _optimizationMaximumIterations = 1024;
        _optimizationErrorPrecision = 0;
        _optimizationToleranceOfSolutionVectorNorm = 1e-4;  // Smallest delta of doubles
        _optimizationToleranceOfVectorFunction = 1e-3;
        _optimizationToleranceOfErrorFunctionGradient = 0;
        _optimizationDiagonalStepBoundShift = 100;
        _maximumRetroprojectionError = 3;

        _pointWeightThreshold = 1.345;
        _pointWeightCoefficient = 1.4826;
        _pointLossAlpha = 2;    // -infinity, infinity
        _pointLossScale = 100; // Unit: Pixel
        _pointErrorMultiplier = 0.5;  // > 0

        // Local map
        _pointUnmatchedCountToLoose = 10;
        _pointAgeConfidence = 15;
        _pointStagedAgeConfidence = 10;
        _pointMinimumConfidenceForMap = 0.9;
        _mapMaximumRetroprojectionError = 150;
        _maximumPointPerFrame = 200;

        // Primitive extraction
        _minimumIOUToConsiderMatch = 0.2;
        _minimumNormalsDotDifference = 0.9;
        _primitiveMaximumCosAngle = cos(M_PI/10.0);
        _primitiveMaximumMergeDistance = 100;
        _depthMapPatchSize = 20;

        _minimumPlaneSeedCount = 6;
        _minimumCellActivated = 5;
        _depthSigmaError = 1.425e-6;
        _depthSigmaMargin = 12;
        _depthDiscontinuityLimit = 10;
        _depthAlpha = 0.06;

        // Cylinder ransac fitting
        _cylinderRansacSqrtMaxDistance = 0.04;
        _cylinderRansacMinimumScore = 75;
        _cylinderRansacProbabilityOfSuccess = 0.8f;
        _cylinderRansacInlierProportions = 0.33f;
    }

    void Parameters::check_parameters_validity()
    {
        _isValid = true;
        if (_matchSearchRadius <= 0)
        {
            utils::log_error("Match search radius must be > 0");
            _isValid = false;
        }
        if (_matchSearchCellSize <= 0)
        {
            utils::log_error("Match search cell size must be > 0");
            _isValid = false;
        }
        if (_maximumMatchDistance <= 0)
        {
            utils::log_error("Minimum match distance must be > 0");
            _isValid = false;
        }
        if (_detectorMinHessian <= 0)
        {
            utils::log_error("Keypoint detector hessian must be > 0");
            _isValid = false;
        }
        if (_keypointRefreshFrequency <= 0)
        {
            utils::log_error("Keypoint refresh frequency must be > 0");
            _isValid = false;
        }
        if (_opticalFlowPyramidDepth <= 0)
        {
            utils::log_error("Pyramid depth must be > 0");
            _isValid = false;
        }
        if (_opticalFlowPyramidWindowSize <= 0)
        {
            utils::log_error("Pyramid window size must be > 0");
            _isValid = false;
        }
        if (_opticalFlowMaxError <= 0)
        {
            utils::log_error("Optical flow maximum error  must be > 0");
            _isValid = false;
        }
        if (_opticalFlowMaxDistance <= 0)
        {
            utils::log_error("Optical flow maximum distance  must be > 0");
            _isValid = false;
        }
        if (_keypointMaskDiameter <= 0)
        {
            utils::log_error("keypoint mask diameters must be > 0");
            _isValid = false;
        }
        
        if (_ransacMaximumRetroprojectionErrorForInliers <= 0)
        {
            utils::log_error("The RANSAC maximum retroprojection distance must be positive");
            _isValid = false;
        }
        if (_ransacMinimumInliersProportionForEarlyStop < 0 or _ransacMinimumInliersProportionForEarlyStop > 1)
        {
            utils::log_error("The RANSAC proportion of inliers must be between 0 and 1");
            _isValid = false;
        }
        if (_ransacProbabilityOfSuccess < 0 or _ransacProbabilityOfSuccess > 1)
        {
            utils::log_error("The RANSAC probability of success should be between 0 and 1");
            _isValid = false;
        }
        if (_ransacInlierProportion < 0 or _ransacInlierProportion > 1)
        {
            utils::log_error("The RANSAC expected proportion of inliers must be between 0 and 1");
            _isValid = false;
        }

        if (_minimumPointForOptimization < 3)
        {
            utils::log_error("A pose cannot be computed with less than 3 points");
            _isValid = false;
        }
        if (_optimizationMaximumIterations <= 0)
        {
            utils::log_error("Optimization maximum iterations must be > 0");
            _isValid = false;
        }
        if (_optimizationErrorPrecision < 0)
        {
            utils::log_error("Optimization error precision must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfSolutionVectorNorm < 0)
        {
            utils::log_error("The optimization tolerance for the norm of the solution vector must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfVectorFunction < 0)
        {
            utils::log_error("The optimization tolerance for the vector function must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfErrorFunctionGradient < 0)
        {
            utils::log_error("The optimization tolerance for the error function gradient must be >= 0");
            _isValid = false;
        }
        if (_optimizationDiagonalStepBoundShift <= 0)
        {
            utils::log_error("The optimization diagonal stepbound shift must be > 0");
            _isValid = false;
        }
        if (_maximumRetroprojectionError <= 0)
        {
            utils::log_error("The maximum retroprojection error  must be > 0");
            _isValid = false;
        }


        if (_pointWeightThreshold <= 0)
        {
            utils::log_error("Point weight threshold must be > 0");
            _isValid = false;
        }
        if (_pointWeightCoefficient <= 0)
        {
            utils::log_error("The point weight coefficient must be > 0");
            _isValid = false;
        }
        if (std::isnan(_pointLossAlpha))
        {
            utils::log_error("The point loss alpha parameter must be a real number");
            _isValid = false;
        }
        if (_pointLossScale <= 0)
        {
            utils::log_error("The point loss scale parameter must be > 0");
            _isValid = false;
        }
        if (_pointErrorMultiplier <= 0)
        {
            utils::log_error("Point error multiplier must be > 0");
            _isValid = false;
        }


        if (_pointUnmatchedCountToLoose <= 0)
        {
            utils::log_error("Unmatched points to loose tracking must be > 0");
            _isValid = false;
        }
        if (_pointAgeConfidence <= 0)
        {
            utils::log_error("Point age confidence must be > 0");
            _isValid = false;
        }
        if (_pointStagedAgeConfidence <= 0)
        {
            utils::log_error("Staged point confidence must be > 0");
            _isValid = false;
        }
        if (_pointMinimumConfidenceForMap <= 0)
        {
            utils::log_error("Minimum confidence to add staged point to map  must be > 0");
            _isValid = false;
        }
        if (_mapMaximumRetroprojectionError <= 0)
        {
            utils::log_error("Maximum retroprojection must be > 0");
            _isValid = false;
        }


        if (_minimumIOUToConsiderMatch <= 0)
        {
            utils::log_error("Minimum InterOverUnion must be > 0");
            _isValid = false;
        }
        if (_minimumCellActivated <= 0)
        {
            utils::log_error("Minimum cell activated must be > 0");
            _isValid = false;
        }
        if (_depthSigmaError <= 0)
        {
            utils::log_error("Depth sigma error must be > 0");
            _isValid = false;
        }
        if (_depthSigmaMargin <= 0)
        {
            utils::log_error("Depth sigma margin must be > 0");
            _isValid = false;
        }
        if (_depthDiscontinuityLimit <= 0)
        {
            utils::log_error("Depth discontinuous limit must be > 0");
            _isValid = false;
        }
        if (_depthAlpha <= 0)
        {
            utils::log_error("Depth Alpha must be > 0");
            _isValid = false;
        }


        if (_cylinderRansacSqrtMaxDistance <= 0)
        {
            utils::log_error("Cylinder RANSAC max distance must be > 0");
            _isValid = false;
        }
        if (_cylinderRansacMinimumScore <= 0)
        {
            utils::log_error("Cylinder RANSAC minimum score must be > 0");
            _isValid = false;
        }
        if (_cylinderRansacInlierProportions <= 0 or _cylinderRansacInlierProportions >= 1)
        {
            utils::log_error("Cylinder RANSAC inlier proportion must be in ]0, 1[");
            _isValid = false;
        }
        if (_cylinderRansacProbabilityOfSuccess <= 0 or _cylinderRansacProbabilityOfSuccess >= 1)
        {
            utils::log_error("Cylinder RANSAC probability of success must be in ]0, 1[");
            _isValid = false;
        }


        if (_camera1FocalX <= 0)
        {
            utils::log_error("Camera 1 focal X distance must be > 0");
            _isValid = false;
        }
        if (_camera1FocalY <= 0)
        {
            utils::log_error("Camera 1 focal Y distance must be > 0");
            _isValid = false;
        }
        if (_camera1CenterX <= 0)
        {
            utils::log_error("Camera 1 center X distance must be > 0");
            _isValid = false;
        }
        if (_camera1CenterY <= 0)
        {
            utils::log_error("Camera 1 center Y distance must be > 0");
            _isValid = false;
        }


        if (_camera2FocalX <= 0)
        {
            utils::log_error("Camera 2 focal X distance must be > 0");
            _isValid = false;
        }
        if (_camera2FocalY <= 0)
        {
            utils::log_error("Camera 2 focal Y distance must be > 0");
            _isValid = false;
        }
        if (_camera2CenterX <= 0)
        {
            utils::log_error("Camera 2 center X distance must be > 0");
            _isValid = false;
        }
        if (_camera2CenterY <= 0)
        {
            utils::log_error("Camera 2 center Y distance must be > 0");
            _isValid = false;
        }


    }

};  /* rgbd_slam */
