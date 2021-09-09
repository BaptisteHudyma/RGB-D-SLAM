#include "parameters.hpp"

#include <math.h>

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        // Camera parameters
        _cameraCenterX = 3.1649e+02;
        _cameraCenterY = 2.2923e+02;
        _cameraFocalX = 5.4886e+02;
        _cameraFocalY = 5.4959e+02;

        // Pose Optimization
        _minimumPointForOptimization = 5;
        _maximumOptimizationCall = 1024;

        // Point detection/Matching
        _maximumMatchDistance = 0.9;
        _detectorMinHessian = 25;

        // Local map
        _pointUnmatchedCountToLoose = 10;
        _pointAgeConfidence = 15;
        _pointStagedAgeConfidence = 5;
        _pointMinimumConfidenceForMap = 0.9;
        _pointWeightThreshold = 1.345;
        _pointWeightCoefficient = 1.4826;
        _pointHubertThreshold = 1.5e-4;
        _pointErrorMultiplier = 0.2;

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
