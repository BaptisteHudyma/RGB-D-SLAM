#include "parameters.hpp"
#include "outputs/logger.hpp"
#include <cfloat>
#include <math.h>
#include <opencv2/core/core.hpp>

namespace rgbd_slam {

bool Parameters::parse_file(const std::string& fileName) noexcept
{
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

void Parameters::load_defaut() noexcept
{
    // Camera intrinsic parameters
    _camera1SizeX = 640; // pixels
    _camera1SizeY = 480; // pixels
    _camera1FocalX = 550;
    _camera1FocalY = 550;
    _camera1CenterX = 320;
    _camera1CenterY = 440;

    _camera2SizeX = 640; // pixels
    _camera2SizeY = 480; // pixels
    _camera2FocalX = 550;
    _camera2FocalY = 550;
    _camera2CenterX = 320;
    _camera2CenterY = 440;

    // Camera 2 position & rotation
    _camera2TranslationX = 0;
    _camera2TranslationY = 0;
    _camera2TranslationZ = 0;

    _camera2RotationX = 0;
    _camera2RotationY = 0;
    _camera2RotationZ = 0;
}

void Parameters::check_parameters_validity() noexcept
{
    _isValid = true;
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

    // static asserts
    static_assert(parameters::coreNumber >= 1, "Number of available computer cores should be >= 1");
    static_assert(parameters::depthSigmaError > 0, "Depth sigma error must be > 0");

    static_assert(parameters::optimization::ransac::maximumRetroprojectionErrorForPointInliers_px > 0,
                  "The RANSAC maximum retroprojection distance for points must be positive");
    static_assert(parameters::optimization::ransac::maximumRetroprojectionErrorForPlaneInliers_mm > 0,
                  "The RANSAC maximum retroprojection distance for planes must be positive");
    static_assert(parameters::optimization::ransac::minimumInliersProportionForEarlyStop >= 0 and
                          parameters::optimization::ransac::minimumInliersProportionForEarlyStop <= 1,
                  "The RANSAC proportion of inliers must be between 0 and 1");
    static_assert(parameters::optimization::ransac::probabilityOfSuccess >= 0 and
                          parameters::optimization::ransac::probabilityOfSuccess <= 1,
                  "The RANSAC probability of success should be between 0 and 1");
    static_assert(parameters::optimization::ransac::inlierProportion >= 0 and
                          parameters::optimization::ransac::inlierProportion <= 1,
                  "The RANSAC expected proportion of inliers must be between 0 and 1");

    static_assert(parameters::optimization::minimumPointForOptimization >= 3,
                  "A pose cannot be computed with less than 3 points");
    static_assert(parameters::optimization::maximumIterations > 0, "Optimization maximum iterations must be > 0");
    static_assert(parameters::optimization::errorPrecision >= 0, "Optimization error precision must be >= 0");
    static_assert(parameters::optimization::toleranceOfSolutionVectorNorm >= 0,
                  "The optimization tolerance for the norm of the solution vector must be >= 0");
    static_assert(parameters::optimization::toleranceOfVectorFunction >= 0,
                  "The optimization tolerance for the vector function must be >= 0");
    static_assert(parameters::optimization::toleranceOfErrorFunctionGradient >= 0,
                  "The optimization tolerance for the error function gradient must be >= 0");
    static_assert(parameters::optimization::diagonalStepBoundShift > 0,
                  "The optimization diagonal stepbound shift must be > 0");

    static_assert(parameters::optimization::maximumRetroprojectionError > 0,
                  "The maximum retroprojection error  must be > 0");

    static_assert(parameters::detection::maxNumberOfPointsToDetect > 0, "Keypoint detector hessian must be > 0");
    static_assert(parameters::detection::keypointCellDetectionSize_px > 0, "Keypoint detection cell size must be > 0");
    static_assert(parameters::detection::keypointRefreshFrequency > 0, "Keypoint refresh frequency must be > 0");
    static_assert(parameters::detection::opticalFlowPyramidDepth > 0, "Pyramid depth must be > 0");
    static_assert(parameters::detection::opticalFlowPyramidWindowSize_px > 0, "Pyramid window size must be > 0");

    static_assert(parameters::detection::minimumPlaneSeedProportion >= 0 and
                          parameters::detection::minimumPlaneSeedProportion <= 100,
                  "Minimum plane seed proportion must be in [0, 100]");
    static_assert(parameters::detection::minimumZeroDepthProportion >= 0 and
                          parameters::detection::minimumZeroDepthProportion <= 1,
                  "Minimum Zero depth proportion must be in [0, 1]");

    static_assert(parameters::detection::maximumPlaneAngleForMerge_d >= 0 and
                          parameters::detection::maximumPlaneAngleForMerge_d <= 180,
                  "Maximum plane patch merge angle must be between 0 and 180");
    static_assert(parameters::detection::maximumPlaneDistanceForMerge_mm >= 0,
                  "Maximum plane patch merge distance must be between positive");
    static_assert(parameters::detection::minimumCellActivatedProportion >= 0 and
                          parameters::detection::minimumCellActivatedProportion <= 100,
                  "Minimum cell activated proportion must be in [0, 100]");

    static_assert(parameters::detection::cylinderRansacSqrtMaxDistance > 0, "Cylinder RANSAC max distance must be > 0");
    static_assert(parameters::detection::cylinderRansacMinimumScore > 0, "Cylinder RANSAC minimum score must be > 0");
    static_assert(parameters::detection::cylinderRansacInlierProportions > 0 and
                          parameters::detection::cylinderRansacInlierProportions < 1,
                  "Cylinder RANSAC inlier proportion must be in ]0, 1[");
    static_assert(parameters::detection::cylinderRansacProbabilityOfSuccess > 0 or
                          parameters::detection::cylinderRansacProbabilityOfSuccess < 1,
                  "Cylinder RANSAC probability of success must be in ]0, 1[");

    static_assert(parameters::matching::minimumPlaneOverlapToConsiderMatch > 0, "Minimum InterOverArea must be > 0");
    static_assert(parameters::matching::maximumAngleForPlaneMatch_d >= 0 and
                          parameters::matching::maximumAngleForPlaneMatch_d <= 90,
                  "Maximum plane match angle must be between 0 and 90 degrees");
    static_assert(parameters::matching::maximumDistanceForPlaneMatch_mm >= 0,
                  "Maximum plane match distance must be greater than zero");

    static_assert(parameters::matching::matchSearchRadius_px > 0, "Match search radius must be > 0");
    static_assert(parameters::matching::maximumMatchDistance > 0, "Minimum match distance must be > 0");

    static_assert(parameters::mapping::pointUnmatchedCountToLoose > 0,
                  "Unmatched points to loose tracking must be > 0");
    static_assert(parameters::mapping::pointStagedAgeConfidence > 0, "Staged point confidence must be > 0");
    static_assert(parameters::mapping::pointMinimumConfidenceForMap > 0,
                  "Minimum confidence to add staged point to map  must be > 0");
}

}; // namespace rgbd_slam
