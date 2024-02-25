#ifndef RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP
#define RGBDSLAM_POSEOPTIMIZATION_POSEOPTIMIZATION_HPP

#include "matches_containers.hpp"
#include "utils/pose.hpp"

namespace rgbd_slam::pose_optimization {

/**
 * \brief Find the transformation between a matches feature sets, using a custom Levenberg Marquardt method
 */
class Pose_Optimization
{
  public:
    /**
     * \brief Compute a new observer global pose, to replace the current estimated pose
     *
     * \param[in] currentPose Last observer optimized pose
     * \param[in] matchedFeatures Object containing the match between observed screen features and reliable map features
     * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned
     * true
     * \param[out] featureSets The inliers/outliers matched features for the finalPose. Valid if the function returned
     * true
     *
     * \return True if a valid pose was computed
     */
    [[nodiscard]] static bool compute_optimized_pose(const utils::PoseBase& currentPose,
                                                     const matches_containers::match_container& matchedFeatures,
                                                     utils::Pose& optimizedPose,
                                                     matches_containers::match_sets& featureSets) noexcept;

    static void show_statistics(const double meanFrameTreatmentDuration,
                                const uint frameCount,
                                const bool shouldDisplayDetails = false) noexcept;

  private:
    /**
     * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
     *
     * \param[in] currentPose Last observer optimized pose
     * \param[in] matchedFeatures Object containing the match between observed screen features and reliable map features
     * \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the function returned
     * true
     * \param[out] optimizationJacobian The jacobian of the distance to pose optimization, if the function returns true
     *
     * \return True if a valid pose was computed
     */
    [[nodiscard]] static bool compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                            const matches_containers::match_container& matchedFeatures,
                                                            utils::PoseBase& optimizedPose,
                                                            matrixd& optimizationJacobian) noexcept;
    [[nodiscard]] static bool compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                            const matches_containers::match_container& matchedFeatures,
                                                            utils::PoseBase& optimizedPose) noexcept;

    [[nodiscard]] static bool compute_optimized_pose_coefficients(
            const utils::PoseBase& currentPose,
            const matches_containers::match_container& matchedFeatures,
            vector6& optimizedCoefficients,
            matrixd& optimizationJacobian) noexcept;

    /**
     * \brief Optimize a global pose (orientation/translation) of the observer, given a match set
     *
     * \param[in] currentPose Last observer optimized pose
     * \param[in] matchedFeatures Object containing the match between observed screen features and reliable map
     * features \param[out] optimizedPose The estimated world translation & rotation of the camera pose, if the
     * function returned true. A covariance will be associated to the pose
     *
     * \return True if a valid pose was computed
     */
    [[nodiscard]] static bool compute_optimized_global_pose_with_covariance(
            const utils::PoseBase& currentPose,
            const matches_containers::match_container& matchedFeatures,
            utils::Pose& optimizedPose) noexcept;

    /**
     * \brief Compute an optimized pose, using a RANSAC methodology
     *
     * \param[in] currentPose The current pose of the observer
     * \param[in] matchedFeatures Object container the match between observed screen features and local map features
     * \param[out] finalPose The optimized pose, valid if the function returned true
     * \param[out] featureSets The matched features detected as inlier and outliers. Valid if the function returned true
     *
     * \return True if a valid pose and inliers were found
     */
    [[nodiscard]] static bool compute_pose_with_ransac(const utils::PoseBase& currentPose,
                                                       const matches_containers::match_container& matchedFeatures,
                                                       utils::Pose& finalPose,
                                                       matches_containers::match_sets& featureSets) noexcept;

    // perf monitoring
    inline static double _meanPoseRANSACDuration = 0.0;

    inline static double _meanGetRandomSubsetDuration = 0.0;
    inline static double _meanRANSACPoseOptimizationDuration = 0.0;
    inline static double _meanRANSACGetInliersDuration = 0.0;
};

} // namespace rgbd_slam::pose_optimization

#endif
