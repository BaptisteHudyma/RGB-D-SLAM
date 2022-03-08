#include "parameters.hpp"

#include <opencv2/core/core.hpp>
#include <math.h>
#include <cfloat>
#include <iostream>

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        _isValid = false;

        // set the global parameters
        set_parameters();

        cv::FileStorage configFile(fileName, cv::FileStorage::READ);
        if (not configFile.isOpened())
        {
            std::cerr << "Cannot load parameter files, starting with default configuration" << std::endl;
            load_defaut();
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

        // TODO Check parameters
        _isValid = true;

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
        _keypointRefreshFrequency = 5; // Update the keypoint list every N calls
        _opticalFlowPyramidDepth = 5;   // depth of the optical pyramid
        _opticalFlowPyramidWindowSize = 25;
        _opticalFlowMaxError = 35;      // error in pixel after which a point is rejected
        _opticalFlowMaxDistance = 100;  // distance in pixel after which a point is rejected
        _keypointMaskDiameter = 10;     // do not detect points inside an area of this size (pixels) around existing keypoints

        // Pose Optimization
        _minimumPointForOptimization = 5;
        _optimizationMaximumIterations = 1024;
        _optimizationErrorPrecision = 0;
        _optimizationToleranceOfSolutionVectorNorm = 1e-4;//sqrt(DBL_EPSILON); // Smallest delta of doubles
        _optimizationToleranceOfVectorFunction = 1e-3;
        _optimizationToleranceOfErrorFunctionGradient = 0;
        _optimizationDiagonalStepBoundShift = 100;
        _maximumOptimizationRANSACiterations = 200;
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

        _isValid = true;
    }

};  /* rgbd_slam */
