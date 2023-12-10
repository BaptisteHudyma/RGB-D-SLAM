#ifndef RGBDSLAM_PARAMETERS_HPP
#define RGBDSLAM_PARAMETERS_HPP

#include "types.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <string>

namespace rgbd_slam {

namespace parameters {
constexpr uint coreNumber = 8; // number of available cores on the computer (1 for no threads)

// Parameters taken from "2012 - 3D with Kinect""
// parameters of equation z_diff = sigmaA + sigmaM * z + sigmaE * z^2, that represent the minimum depth
// change for a given depth (quantization)
constexpr double depthSigmaError = 2.73;      // It is the sigmaE;
constexpr double depthSigmaMultiplier = 0.74; // It is the sigmaM;
constexpr double depthSigmaMargin = -0.53;    // It is the sigmaA

// Optimisation (ransac)
namespace optimization {
namespace ransac {
constexpr float maximumRetroprojectionErrorForPoint2DInliers_mm =
        20.0; // Max retroprojection error between the line of two inverse depth points to reject match (mm)
constexpr float maximumRetroprojectionErrorForPointInliers_px =
        10.0; // Max retroprojection error between two screen points before rejecting the match (pixels);
constexpr float maximumRetroprojectionErrorForPlaneInliers_mm =
        20.0; // Max retroprojection error between two screen planes, in millimeters, before rejecting the match
constexpr double minimumInliersProportionForEarlyStop =
        0.90f; // proportion of inliers in total set, to stop RANSAC early

// RANSAC iteration parameters
constexpr float probabilityOfSuccess = 0.8f; // probability of having at least one correct transformation
constexpr float inlierProportion = 0.6f;     // number of inliers in data / number of points in data
constexpr float featureTrustCount = 10.0;    // number of expected features expected to pass the test
} // namespace ransac

constexpr uint minimumPointForOptimization = 5; // Should be >= 5, the minimum point count for a 3D pose estimation
constexpr uint minimumPlanesForOptimization =
        3;                             // Should be >= 3, the minimum infinite plane count for a 3D pose estimation
constexpr uint maximumIterations = 64; // Max iteration of the Levenberg Marquart optimisation
constexpr float errorPrecision = 0.0f; // tolerance for the norm of the solution vector
constexpr float toleranceOfSolutionVectorNorm = 1e-4f;   // Smallest delta of doubles
constexpr float toleranceOfVectorFunction = 1e-3f;       // tolerance for the norm of the vector function
constexpr float toleranceOfErrorFunctionGradient = 0.0f; // tolerance for the norm of the gradient of the error function
constexpr float diagonalStepBoundShift = 100.0f;         // step bound for the diagonal shift
constexpr float maximumRetroprojectionError_px =
        3.0f; // In pixel: maximum distance after which we can consider a retroprojection as invalid
} // namespace optimization

namespace detection {
// point detection
constexpr uint keypointCellDetectionHeightCount = 3; // the number of the keypoint detection window in height
constexpr uint keypointCellDetectionWidthCount = 3;  // the number of the keypoint detection window in width
#ifdef USE_ORB_DETECTOR_AND_MATCHING
constexpr uint pointDetectorOrbThreshold = 200; // number of points to detect (per frame)
#else
constexpr uint pointDetectorThreshold = 10; // point detector sensitivity (per cell detection window)
#endif
constexpr uint maximumPointPerFrame =
        100; // maximum points per frame, over which we do not want to detect more points (optimization)
constexpr uint keypointRefreshFrequency = 5; // Update the keypoint list every N calls (opti)

// point tracking
constexpr uint opticalFlowPyramidDepth =
        4; // depth of the optical pyramid (0 based. Higher than 5 levels is mostly useless)
constexpr uint opticalFlowPyramidWindowSizeHeightCount = 9; // search size window count (vertical) at each pyramid level
constexpr uint opticalFlowPyramidWindowSizeWidthCount =
        12; // search size window count (horizontal) at each pyramid level

// inverse depth
constexpr double inverseDepthBaseline = 1.0 / 1000.0; // baseline of the inverse depth, in 1/millimeters
constexpr double inverseDepthAngleBaseline = 0.5;     // baseline of the inverse depth measurment angles, in degrees

// plane detection
constexpr double minimumPlaneSeedProportion =
        0.8 / 100.0; // grow planes only if we have more than this proportion of planes patch in this seed
constexpr double minimumCellActivatedProportion =
        0.65 / 100.0; // grow planes only if their is this proportion of mergeable planes in the remaining patches
constexpr float minimumZeroDepthProportion =
        0.7f; // if this proportion of the points have invalid depth in a planar patch, reject it

constexpr float maximumPlaneAngleForMerge_d =
        18.0; // plane patches can be merged if their normals angle is below this angle (degrees)
constexpr float maximumPlaneDistanceForMerge_mm =
        50.0; // plane patched can be merged if their distances is below this distance (millimeters)
constexpr uint depthMapPatchSize_px =
        20; // Divide the depth image in patches of this size (pixels) to detect primitives

// Cylinder ransac fitting
constexpr float cylinderRansacSqrtMaxDistance = 0.04f;
constexpr float cylinderRansacMinimumScore = 75;
constexpr float cylinderRansacInlierProportions = 0.33f;
constexpr float cylinderRansacProbabilityOfSuccess = 0.8f;
} // namespace detection

namespace matching {
constexpr float minimumPlaneOverlapToConsiderMatch =
        0.4f; // // Inter over area of the two primitive masks, to consider a primitive match
constexpr double maximumAngleForPlaneMatch_d =
        20.0; // Maximum angle between two primitives to consider a match (degrees)
constexpr double maximumDistanceForPlaneMatch_mm =
        100; // Maximum distance between two plane d component to consider a match (millimeters)

constexpr double matchSearchRadius_px = 30;  // Radius of the space around a point to search match points in pixels
constexpr double maximumMatchDistance = 0.7; // Maximum distance between a point and his mach before refusing the
                                             // match (closer to zero = more discriminating)
} // namespace matching

namespace mapping {
// local map management
constexpr uint pointUnmatchedCountToLoose =
        10; // consecutive unmatched frames before removing from local map (high is good, but consumes more perfs);
constexpr uint planeUnmatchedCountToLoose =
        10; // consecutive unmatched frames before removing from local map (high is good, but consumes more perfs);
constexpr uint pointStagedAgeConfidence = 3;         // Minimum age of a point in staged map to consider it good
constexpr double pointMinimumConfidenceForMap = 0.9; // Minimum confidence of a staged point to add it to local map
constexpr float maximumRetroprojectionErrorForTriangulatePow_px =
        6.0; // In pixel: maximum squared distance after which we can consider a retroprojection as invalid
} // namespace mapping

} // namespace parameters

/**
 * \brief Store all parameters of this SLAM program.
 * It should be used as a static class everywhere in the program.
 * If the parameters are not loaded from a file, a set of default parameters will be used.
 */
class Parameters
{
    using vector2_uint = Eigen::Vector<uint, 2>;

  public:
    /**
     * \brief Parse a yaml configuration file and load the parameters. Sets the default parameters
     */
    [[nodiscard]] static bool parse_file(const std::string& fileName) noexcept;

    /**
     * \brief Set the default camera parameters
     */
    static void load_defaut() noexcept;

    [[nodiscard]] static bool is_valid() noexcept { return _isValid; };

    // Camera 1 is the left camera in stereo, and the color camera in RGBD
    [[nodiscard]] static vector2_uint get_camera_1_image_size() noexcept { return _camera1ImageSize; };
    [[nodiscard]] static vector2 get_camera_1_center() noexcept { return _camera1Center; };
    [[nodiscard]] static vector2 get_camera_1_focal() noexcept { return _camera1Focal; };

    /**
     * \brief Get the camera pinhole model
     */
    [[nodiscard]] static matrix33 get_camera_1_intrinsics()
    {
        const static matrix33 cameraMatrix(
                {{_camera1Focal.x(), 0, _camera1Center.x()}, {0, _camera1Focal.y(), _camera1Center.y()}, {0, 0, 1}});
        return cameraMatrix;
    };

    // Camera 2 is the right camera in stereo, and the depth camera in RGBD
    [[nodiscard]] static vector2_uint get_camera_2_image_size() noexcept { return _camera2ImageSize; };
    [[nodiscard]] static vector2 get_camera_2_center() noexcept { return _camera2Center; };
    [[nodiscard]] static vector2 get_camera_2_focal() noexcept { return _camera2Focal; };

    /**
     * \brief Get the camera pinhole model
     */
    [[nodiscard]] static matrix33 get_camera_2_intrinsics()
    {
        const static matrix33 cameraMatrix(
                {{_camera2Focal.x(), 0, _camera2Center.x()}, {0, _camera2Focal.y(), _camera2Center.y()}, {0, 0, 1}});
        return cameraMatrix;
    };

    [[nodiscard]] static matrix44 get_camera_2_to_camera_1_transformation() noexcept
    {
        return _camera2toCamera1transformation;
    }

  private:
    // Is this set of parameters valid
    inline static bool _isValid = false;

    // Cameras intrinsics parameters
    inline static vector2_uint _camera1ImageSize;
    inline static vector2 _camera1Center;
    inline static vector2 _camera1Focal;

    inline static vector2_uint _camera2ImageSize;
    inline static vector2 _camera2Center;
    inline static vector2 _camera2Focal;

    // Camera 2 position and rotation to go to camera 1
    inline static matrix44 _camera2toCamera1transformation;

    /**
     * \brief Update the _isValid attribute
     */
    static void check_parameters_validity() noexcept;
};

}; // namespace rgbd_slam

#endif
