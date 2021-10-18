#include "parameters.hpp"

#include <math.h>

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        // Camera parameters
        _cameraCenterX = 316.49;
        _cameraCenterY = 229.23;
        _cameraFocalX = 548.86;
        _cameraFocalY = 549.59;

        // Point detection/Matching
        _matchSearchRadius = 30;
        _matchSearchCellSize = 50;
        _maximumMatchDistance = 0.7;   // The closer to 0, the more discriminating
        _detectorMinHessian = 45;

        // Pose Optimization
        _minimumPointForOptimization = 5;
        _maximumGlobalOptimizationCall = 1024;
        _pointWeightThreshold = 1.345;
        _pointWeightCoefficient = 1.4826;
        _pointLossAlpha = -1000;  // -infinity, infinity
        _pointErrorMultiplier = 0.5;  // > 0

        // Local map
        _pointUnmatchedCountToLoose = 10;
        _pointAgeConfidence = 15;
        _pointStagedAgeConfidence = 10;
        _pointMinimumConfidenceForMap = 0.9;

        // Primitive extraction
        _primitiveMaximumCosAngle = cos(M_PI/10.0);
        _primitiveMaximumMergeDistance = 100;
        _depthMapPatchSize = 20;

        _minimumPlaneSeedCount = 6;
        _minimumCellActivated = 5;
        _depthSigmaError = 1.425e-6;
        _depthSigmaMargin = 12;
        _depthDiscontinuityLimit = 4;
        _depthAlpha = 0.06;
        
        // Cylinder ransac fitting
        _cylinderRansacSqrtMaxDistance = 0.04;
        _cylinderRansacMinimumScore = 75;

        return true;
    }

};  /* rgbd_slam */
