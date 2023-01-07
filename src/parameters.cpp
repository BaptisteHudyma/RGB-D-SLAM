#include "parameters.hpp"

#include <opencv2/core/core.hpp>
#include <math.h>
#include <cfloat>

#include "outputs/logger.hpp"

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        // set the global parameters
        set_parameters();

        cv::FileStorage configFile(fileName, cv::FileStorage::READ);
        if (not configFile.isOpened())
        {
            outputs::log_error("Cannot load parameter files, starting with default configuration");
            load_defaut();
            check_parameters_validity();
            return false;
        }

        // Load camera 1 parameters
        _camera1SizeX = int(configFile["camera_1_size_x"]);
        _camera1SizeY = int(configFile["camera_1_size_y"]);
        _camera1FocalX = configFile["camera_1_focal_x"];
        _camera1FocalY = configFile["camera_1_focal_y"];
        _camera1CenterX = configFile["camera_1_center_x"];
        _camera1CenterY = configFile["camera_1_center_y"];

        // Load camera 2 parameters
        _camera2SizeX = int(configFile["camera_2_size_x"]);
        _camera2SizeY = int(configFile["camera_2_size_y"]);
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

        // Camera intrinsic parameters
        _camera1SizeX = 640;    //pixels
        _camera1SizeY = 480;    //pixels
        _camera1FocalX = 548.86723733696215;
        _camera1FocalY = 549.58402532237187;
        _camera1CenterX = 316.49655835885483;
        _camera1CenterY = 229.23873484682150;

        _camera2SizeX = 640;    //pixels
        _camera2SizeY = 480;    //pixels
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
        // available core on this computer, 1 disables threads
        _coreNumber = 8;

        // Point detection/Matching
        _matchSearchRadius = 30;        // max distance to a point after which we do not consider a math (pixels)
        _matchSearchCellSize = 50;      // size of the squares dividing the image to search for point matches (pixels)
        _maximumMatchDistance = 0.7;    // The closer to 0, the more discriminating
        _maxNumberOfPointsToDetect = 200;   // Maximum number of point features to detect
        _keypointRefreshFrequency = 5;  // Update the keypoint list every N calls
        _opticalFlowPyramidDepth = 5;   // depth of the optical pyramid
        _opticalFlowPyramidWindowSize = 25;
        _opticalFlowMaxError = 35;      // error in pixel after which a point is rejected
        _opticalFlowMaxDistance = 100;  // distance in pixel after which a point is rejected
        _keypointMaskRadius = 50;       // do not detect points inside an area of this size (pixels) around existing keypoints

        // Pose Optimization
        _ransacMaximumRetroprojectionErrorForPointInliers = 10;  // Max retroprojection error between two screen points, in pixels, before rejecting the match
        _ransacMaximumRetroprojectionErrorForPlaneInliers = 15;  // Max retroprojection error between two screen points, in meters, before rejecting the match
        _ransacMinimumInliersProportionForEarlyStop = 0.90; // proportion of inliers in total set, to stop RANSAC early
        _ransacProbabilityOfSuccess = 0.8;   // probability of having at least one correct transformation
        _ransacInlierProportion = 0.6;       // number of inliers in data / number of points in data 

        _minimumPointForOptimization = 5;   // Should be >= 5, the minimum point count for a 3D pose estimation
        _minimumPlanesForOptimization = 3;  // Should be >= 3, the minimum infinite plane count for a 3D pose estimation
        _optimizationMaximumIterations = 64;
        _optimizationErrorPrecision = 0;
        _optimizationToleranceOfSolutionVectorNorm = 1e-4;  // Smallest delta of doubles
        _optimizationToleranceOfVectorFunction = 1e-3;
        _optimizationToleranceOfErrorFunctionGradient = 0;
        _optimizationDiagonalStepBoundShift = 100;
        _maximumRetroprojectionError = 3;

        // Local map
        _pointUnmatchedCountToLoose = 60;   // consecutive unmatched frames before removing from local map
        _pointStagedAgeConfidence = 3;
        _pointMinimumConfidenceForMap = 0.9;
        _maximumPointPerFrame = 200;

        // Primitive extraction
        _minimumIOUToConsiderMatch = 0.7;   // Inter Over Union of planes
        _maximumAngleForPlaneMatch = 60.0;  // Plane segments could be merged below this angle
        _maximumPlaneAngleForMerge = 18.0;  // plane segments can be merged if their normals angle is below this angle
        _depthMapPatchSize = 20;            // Divide the depth image in patches of this size (pixels) to detect primitives

        _minimumPlaneSeedProportion = 0.8 / 100.0;      // grow planes only if we have more than this proportion of planes patch in this seed
        _minimumCellActivatedProportion = 0.65 / 100.0; // grow planes only if their is this proportion of mergeable planes in the remaining patches
        _minimumZeroDepthProportion = 0.7;              // if this proportion of the points have invalid depth in a planar patch, reject it

        // Parameters taken from "2012 - 3D with Kinect""
        //parameters of equation z_diff = sigmaA + sigmaM * z + sigmaE * z^2, that represent the minimum depth change for a given depth (quantization)
        _depthSigmaError = 2.73;         // It is the sigmaE
        _depthSigmaMultiplier = 0.74;    // It is the sigmaM
        _depthSigmaMargin = -0.53;              // It is the sigmaA

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
            outputs::log_error("Match search radius must be > 0");
            _isValid = false;
        }
        if (_matchSearchCellSize <= 0)
        {
            outputs::log_error("Match search cell size must be > 0");
            _isValid = false;
        }
        if (_maximumMatchDistance <= 0)
        {
            outputs::log_error("Minimum match distance must be > 0");
            _isValid = false;
        }
        if (_maxNumberOfPointsToDetect <= 0)
        {
            outputs::log_error("Keypoint detector hessian must be > 0");
            _isValid = false;
        }
        if (_keypointRefreshFrequency <= 0)
        {
            outputs::log_error("Keypoint refresh frequency must be > 0");
            _isValid = false;
        }
        if (_opticalFlowPyramidDepth <= 0)
        {
            outputs::log_error("Pyramid depth must be > 0");
            _isValid = false;
        }
        if (_opticalFlowPyramidWindowSize <= 0)
        {
            outputs::log_error("Pyramid window size must be > 0");
            _isValid = false;
        }
        if (_opticalFlowMaxError <= 0)
        {
            outputs::log_error("Optical flow maximum error  must be > 0");
            _isValid = false;
        }
        if (_opticalFlowMaxDistance <= 0)
        {
            outputs::log_error("Optical flow maximum distance  must be > 0");
            _isValid = false;
        }
        if (_keypointMaskRadius <= 0)
        {
            outputs::log_error("keypoint mask diameters must be > 0");
            _isValid = false;
        }

        if (_ransacMaximumRetroprojectionErrorForPointInliers <= 0)
        {
            outputs::log_error("The RANSAC maximum retroprojection distance for points must be positive");
            _isValid = false;
        }
        if (_ransacMaximumRetroprojectionErrorForPlaneInliers <= 0)
        {
            outputs::log_error("The RANSAC maximum retroprojection distance for planes must be positive");
            _isValid = false;
        }
        if (_ransacMinimumInliersProportionForEarlyStop < 0 or _ransacMinimumInliersProportionForEarlyStop > 1)
        {
            outputs::log_error("The RANSAC proportion of inliers must be between 0 and 1");
            _isValid = false;
        }
        if (_ransacProbabilityOfSuccess < 0 or _ransacProbabilityOfSuccess > 1)
        {
            outputs::log_error("The RANSAC probability of success should be between 0 and 1");
            _isValid = false;
        }
        if (_ransacInlierProportion < 0 or _ransacInlierProportion > 1)
        {
            outputs::log_error("The RANSAC expected proportion of inliers must be between 0 and 1");
            _isValid = false;
        }

        if (_minimumPointForOptimization < 3)
        {
            outputs::log_error("A pose cannot be computed with less than 3 points");
            _isValid = false;
        }
        if (_optimizationMaximumIterations <= 0)
        {
            outputs::log_error("Optimization maximum iterations must be > 0");
            _isValid = false;
        }
        if (_optimizationErrorPrecision < 0)
        {
            outputs::log_error("Optimization error precision must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfSolutionVectorNorm < 0)
        {
            outputs::log_error("The optimization tolerance for the norm of the solution vector must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfVectorFunction < 0)
        {
            outputs::log_error("The optimization tolerance for the vector function must be >= 0");
            _isValid = false;
        }
        if (_optimizationToleranceOfErrorFunctionGradient < 0)
        {
            outputs::log_error("The optimization tolerance for the error function gradient must be >= 0");
            _isValid = false;
        }
        if (_optimizationDiagonalStepBoundShift <= 0)
        {
            outputs::log_error("The optimization diagonal stepbound shift must be > 0");
            _isValid = false;
        }
        if (_maximumRetroprojectionError <= 0)
        {
            outputs::log_error("The maximum retroprojection error  must be > 0");
            _isValid = false;
        }


        if (_pointUnmatchedCountToLoose <= 0)
        {
            outputs::log_error("Unmatched points to loose tracking must be > 0");
            _isValid = false;
        }
        if (_pointStagedAgeConfidence <= 0)
        {
            outputs::log_error("Staged point confidence must be > 0");
            _isValid = false;
        }
        if (_pointMinimumConfidenceForMap <= 0)
        {
            outputs::log_error("Minimum confidence to add staged point to map  must be > 0");
            _isValid = false;
        }


        if (_minimumIOUToConsiderMatch <= 0)
        {
            outputs::log_error("Minimum InterOverUnion must be > 0");
            _isValid = false;
        }
        if (_maximumAngleForPlaneMatch < 0 or _maximumAngleForPlaneMatch > 180)
        {
            outputs::log_error("Maximum plane match angle must be between 0 and 180");
            _isValid = false;
        }
        if (_maximumPlaneAngleForMerge < 0 or _maximumPlaneAngleForMerge > 180)
        {
            outputs::log_error("Maximum plane patch merge angle must be between 0 and 180");
            _isValid = false;
        }
        if (_minimumCellActivatedProportion < 0 or _minimumCellActivatedProportion > 100)
        {
            outputs::log_error("Minimum cell activated proportion must be in [0, 100]");
            _isValid = false;
        }
        if (_minimumPlaneSeedProportion < 0 or _minimumPlaneSeedProportion > 100)
        {
            outputs::log_error("Minimum plane seed proportion must be in [0, 100]");
            _isValid = false;
        }
        if (_minimumZeroDepthProportion < 0 or _minimumZeroDepthProportion > 1)
        {
            outputs::log_error("Minimum Zero depth proportion must be in [0, 1]");
            _isValid = false;
        }
        if (_depthSigmaError <= 0)
        {
            outputs::log_error("Depth sigma error must be > 0");
            _isValid = false;
        }


        if (_cylinderRansacSqrtMaxDistance <= 0)
        {
            outputs::log_error("Cylinder RANSAC max distance must be > 0");
            _isValid = false;
        }
        if (_cylinderRansacMinimumScore <= 0)
        {
            outputs::log_error("Cylinder RANSAC minimum score must be > 0");
            _isValid = false;
        }
        if (_cylinderRansacInlierProportions <= 0 or _cylinderRansacInlierProportions >= 1)
        {
            outputs::log_error("Cylinder RANSAC inlier proportion must be in ]0, 1[");
            _isValid = false;
        }
        if (_cylinderRansacProbabilityOfSuccess <= 0 or _cylinderRansacProbabilityOfSuccess >= 1)
        {
            outputs::log_error("Cylinder RANSAC probability of success must be in ]0, 1[");
            _isValid = false;
        }

        if (_coreNumber < 1)
        {
            outputs::log_error("Number of available computer cores should be >= 1");
            _isValid = false;
        }

        if (_camera1SizeX <= 0)
        {
            outputs::log_error("Camera 1 size X must be > 0");
            _isValid = false;
        }
        if (_camera1SizeY <= 0)
        {
            outputs::log_error("Camera 1 size Y must be > 0");
            _isValid = false;
        }
        if (_camera1FocalX <= 0)
        {
            outputs::log_error("Camera 1 focal X distance must be > 0");
            _isValid = false;
        }
        if (_camera1FocalY <= 0)
        {
            outputs::log_error("Camera 1 focal Y distance must be > 0");
            _isValid = false;
        }
        if (_camera1CenterX < 0)
        {
            outputs::log_error("Camera 1 center X distance must be >= 0");
            _isValid = false;
        }
        if (_camera1CenterY < 0)
        {
            outputs::log_error("Camera 1 center Y distance must be >= 0");
            _isValid = false;
        }

        if (_camera2SizeX <= 0)
        {
            outputs::log_error("Camera 2 size X must be > 0");
            _isValid = false;
        }
        if (_camera2SizeY <= 0)
        {
            outputs::log_error("Camera 2 size Y must be > 0");
            _isValid = false;
        }
        if (_camera2FocalX <= 0)
        {
            outputs::log_error("Camera 2 focal X distance must be > 0");
            _isValid = false;
        }
        if (_camera2FocalY <= 0)
        {
            outputs::log_error("Camera 2 focal Y distance must be > 0");
            _isValid = false;
        }
        if (_camera2CenterX < 0)
        {
            outputs::log_error("Camera 2 center X distance must be >= 0");
            _isValid = false;
        }
        if (_camera2CenterY < 0)
        {
            outputs::log_error("Camera 2 center Y distance must be >= 0");
            _isValid = false;
        }


    }

};  /* rgbd_slam */
