#include "parameters.hpp"
#include "angle_utils.hpp"
#include "camera_transformation.hpp"
#include "outputs/logger.hpp"
#include "types.hpp"
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
    _camera1ImageSize.x() = int(configFile["camera_1_size_x"]);
    _camera1ImageSize.y() = int(configFile["camera_1_size_y"]);
    _camera1Focal.x() = configFile["camera_1_focal_x"];
    _camera1Focal.y() = configFile["camera_1_focal_y"];
    _camera1Center.x() = configFile["camera_1_center_x"];
    _camera1Center.y() = configFile["camera_1_center_y"];

    // Load camera 2 parameters
    _camera2ImageSize.x() = int(configFile["camera_2_size_x"]);
    _camera2ImageSize.y() = int(configFile["camera_2_size_y"]);
    _camera2Focal.x() = configFile["camera_2_focal_x"];
    _camera2Focal.y() = configFile["camera_2_focal_y"];
    _camera2Center.x() = configFile["camera_2_center_x"];
    _camera2Center.y() = configFile["camera_2_center_y"];

    // Load camera offsets of camera 2, relative to camera 1
    const float camera2TranslationX = configFile["camera_2_translation_offset_x"];
    const float camera2TranslationY = configFile["camera_2_translation_offset_y"];
    const float camera2TranslationZ = configFile["camera_2_translation_offset_z"];
    const vector3 cam2tocam1Translation(camera2TranslationX, camera2TranslationY, camera2TranslationZ);

    const float camera2RotationX = configFile["camera_2_rotation_offset_x"];
    const float camera2RotationY = configFile["camera_2_rotation_offset_y"];
    const float camera2RotationZ = configFile["camera_2_rotation_offset_z"];
    const quaternion& cam2tocam1Rotation =
            utils::get_quaternion_from_euler_angles(EulerAngles(camera2RotationX, camera2RotationY, camera2RotationZ));

    _camera2toCamera1transformation = utils::get_transformation_matrix(cam2tocam1Rotation, cam2tocam1Translation);

    // -------

    check_parameters_validity();

    configFile.release();
    return _isValid;
}

void Parameters::load_defaut() noexcept
{
    // Camera intrinsic parameters
    _camera1ImageSize = vector2_uint(640, 480); // pixels
    _camera1Focal = vector2(550, 550);
    _camera1Center.x() = static_cast<float>(_camera1ImageSize.x()) / 2;
    _camera1Center.y() = static_cast<float>(_camera1ImageSize.y()) / 2;

    _camera2ImageSize = vector2_uint(640, 480); // pixels
    _camera2Focal = vector2(550, 550);
    _camera2Center.x() = static_cast<float>(_camera2ImageSize.x()) / 2;
    _camera2Center.y() = static_cast<float>(_camera2ImageSize.y()) / 2;

    // Camera 2 position & rotation
    _camera2toCamera1transformation = utils::get_transformation_matrix(quaternion::Identity(), vector3::Zero());
}

void Parameters::check_parameters_validity() noexcept
{
    _isValid = true;
    if (_camera1ImageSize.x() <= 0)
    {
        outputs::log_error("Camera 1 size X must be > 0");
        _isValid = false;
    }
    if (_camera1ImageSize.y() <= 0)
    {
        outputs::log_error("Camera 1 size Y must be > 0");
        _isValid = false;
    }
    if (_camera1Center.x() < 0)
    {
        outputs::log_error("Camera 1 center X distance must be >= 0");
        _isValid = false;
    }
    if (_camera1Center.y() < 0)
    {
        outputs::log_error("Camera 1 center Y distance must be >= 0");
        _isValid = false;
    }

    if (_camera2ImageSize.x() <= 0)
    {
        outputs::log_error("Camera 2 size X must be > 0");
        _isValid = false;
    }
    if (_camera2ImageSize.y() <= 0)
    {
        outputs::log_error("Camera 2 size Y must be > 0");
        _isValid = false;
    }
    if (_camera2Center.x() < 0)
    {
        outputs::log_error("Camera 2 center X distance must be >= 0");
        _isValid = false;
    }
    if (_camera2Center.y() < 0)
    {
        outputs::log_error("Camera 2 center Y distance must be >= 0");
        _isValid = false;
    }

    // static asserts
    static_assert(parameters::coreNumber >= 1, "Number of available computer cores should be >= 1");
    static_assert(parameters::depthSigmaError > 0, "Depth sigma error must be > 0");

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

    static_assert(parameters::detection::keypointCellDetectionHeightCount > 0,
                  "Keypoint detection height cell count must be > 0");
    static_assert(parameters::detection::keypointCellDetectionWidthCount > 0,
                  "Keypoint detection width cell size must be > 0");
    static_assert(parameters::detection::keypointRefreshFrequency > 0, "Keypoint refresh frequency must be > 0");
    static_assert(parameters::detection::opticalFlowPyramidDepth > 0, "Pyramid depth must be > 0");
    static_assert(parameters::detection::opticalFlowPyramidWindowSizeHeightCount > 0,
                  "Pyramid window count vertical size must be > 0");
    static_assert(parameters::detection::opticalFlowPyramidWindowSizeWidthCount > 0,
                  "Pyramid window count horizontal size must be > 0");

    static_assert(parameters::detection::inverseDepthBaseline > 0, "inverseDepthBaseline should be > 0");
    static_assert(parameters::detection::inverseDepthAngleBaseline > 0, "inverseDepthAngleBaseline should be > 0");

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
