#include "parameters.hpp"

#include <math.h>

namespace rgbd_slam {

    bool Parameters::parse_file(const std::string& fileName )
    {
        _cameraCenterX = 3.1649e+02;
        _cameraCenterY = 2.2923e+02;
        _cameraFocalX = 5.4886e+02;
        _cameraFocalY = 5.4959e+02;

        _minimumPointForOptimization = 5;
        _maximumOptimizationCall = 4048;

        _maximumMatchDistance = 0.7;
        _detectorMinHessian = 25;

        _primitiveMaximumCosAngle = cos(M_PI/10.0);
        _primitiveMaximumMergeDistance = 100;
        _depthMapPatchSize = 20;

        _minimumPlaneSeedCount = 6;
        _minimumCellActivated = 5;
        _depthSigmaError = 1.425e-6;
        _depthSigmaMargin = 12;
        _depthDiscontinuityLimit = 4;
        _depthAlpha = 0.06;

        _cylinderRansacSqrtMaxDistance = 0.04;
        _cylinderRansacMinimumScore = 75;

        _pointUnmatchedCountToLoose = 5;
        _pointAgeLiability = 0;

        return true;
    }

};  /* rgbd_slam */
