#include "pose_optimization.hpp"

#include "angle_utils.hpp"
#include "covariances.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "levenberg_marquardt_functors.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "ransac.hpp"
#include "types.hpp"

#include "utils/camera_transformation.hpp"

#include <Eigen/StdVector>
#include <cmath>
#include <exception>
#include <limits>
#include <opencv2/core/utility.hpp>
#include <stdexcept>
#include <string>
#include <format>

#include <tbb/parallel_for.h>
#include <utility>

namespace rgbd_slam::pose_optimization {

constexpr size_t numberOfFeatures = 3;

constexpr size_t featureIndexPlane = 0;
constexpr size_t featureIndexPoint = 1;
constexpr size_t featureIndex2dPoint = 2;

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] featuresToEvaluate The set of features to evaluate the transformation on
 * \param[in] transformationPose The transformation that needs to be evaluated
 * \param[out] matchSets The set of inliers/outliers of this transformation
 * \return The transformation score, and the inlier feature score
 */
[[nodiscard]] double get_features_inliers_outliers(const matches_containers::match_container& featuresToEvaluate,
                                                   const utils::PoseBase& transformationPose,
                                                   matches_containers::match_sets& matchSets) noexcept
{
    matchSets.clear();

    // get a world to camera transform to evaluate the retroprojection score
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            transformationPose.get_orientation_quaternion(), transformationPose.get_position());

    double featureScore = 0.0;
    for (const auto& match: featuresToEvaluate)
    {
        const vectorxd& std = match->get_distance_covariance(worldToCamera).diagonal().cwiseSqrt();
        // Retroproject world point to screen, and compute screen distance
        try
        {
            // get the feature distance to it's match
            const vectorxd& distances = match->get_distance(worldToCamera).cwiseAbs();
            // all inliers
            if ((distances.array() < std.array()).all())
            {
                matchSets._inliers.insert(matchSets._inliers.end(), match);
                featureScore += match->get_score();
            }
            // outlier
            else
            {
                matchSets._outliers.insert(matchSets._outliers.end(), match);
            }
        }
        catch (const std::exception& ex)
        {
            outputs::log_error("get_features_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
            matchSets._outliers.insert(matchSets._outliers.end(), match);
        }
    }
    return featureScore;
}

/**
 * \brief compute the score of a feature set
 */
double get_feature_set_optimization_score(const matches_containers::match_container& matchedFeatures)
{
    double score = 0.0;
    for (const auto& match: matchedFeatures)
    {
        score += match->get_score();
    }
    return score;
}

/**
 * \brief Return a subset of features
 */
[[nodiscard]] matches_containers::match_container get_random_subset(
        const matches_containers::match_container& matchedFeatures)
{
    matches_containers::match_container matchSubset;

    // we can have a lot of points, so use a more efficient but with potential duplicates subset
    // matchSubset._inliers = ransac::get_random_subset_with_score_with_duplicates(matchedFeatures, 1.0);
    matchSubset = ransac::get_random_subset_with_score(matchedFeatures, 1.0);
    if (get_feature_set_optimization_score(matchSubset) < 1.0)
    {
        throw std::logic_error("get_random_subset: the output subset should have the requested score");
    }

    return matchSubset;
}

bool Pose_Optimization::compute_pose_with_ransac(const utils::PoseBase& currentPose,
                                                 const matches_containers::match_container& matchedFeatures,
                                                 utils::Pose& finalPose,
                                                 matches_containers::match_sets& featureSets) noexcept
{
    const double computePoseRansacStartTime = static_cast<double>(cv::getTickCount());

    featureSets.clear();

    // check that we have enough features for minimal pose optimization
    const double initialFeatureScore = get_feature_set_optimization_score(matchedFeatures);
    if (initialFeatureScore < 1.0)
    {
        // if there is not enough potential inliers to optimize a pose
        outputs::log_warning(
                std::format("Not enough features to optimize a pose (score {} is < 1.0)", initialFeatureScore));
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    // Compute maximum iteration with the original RANSAC formula
    static const uint maximumIterations = static_cast<uint>(
            std::ceil(std::log(1.0f - parameters::optimization::ransac::probabilityOfSuccess) /
                      std::log(1.0f - std::pow(parameters::optimization::ransac::inlierProportion,
                                               parameters::optimization::ransac::featureTrustCount))));
    if (maximumIterations <= 0)
    {
        outputs::log_error("maximumIterations should be > 0, no pose optimization will be made");
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    // TODO: check the constant: 60% inlier proportion is weak
    // matchedFeatures.size() * parameters::optimization::ransac::inlierProportion);
    const size_t inliersToStop = static_cast<size_t>(static_cast<double>(matchedFeatures.size()) * 0.80);

    double maxScore = 1.0;
    utils::PoseBase bestPose = currentPose;
    matches_containers::match_sets finalFeatureSets;

    std::atomic_bool canQuit = false;
    std::mutex mut;

#ifndef MAKE_DETERMINISTIC
    // parallel loop to speed up the process
    // USING THIS PARALLEL LOOP BREAKS THE RANDOM SEEDING
    tbb::parallel_for(
            uint(0),
            maximumIterations,
            [&](uint iteration) {
                if (canQuit.load())
                    return;
#else
    for (uint iteration = 0; iteration < maximumIterations; ++iteration)
    {
        if (canQuit.load())
            break;
#endif
                const double getRandomSubsetStartTime = static_cast<double>(cv::getTickCount());
                // get a random subset for this iteration
                const matches_containers::match_container& selectedMatches = get_random_subset(matchedFeatures);
                _meanGetRandomSubsetDuration +=
                        (static_cast<double>(cv::getTickCount()) - getRandomSubsetStartTime) / cv::getTickFrequency();

                // compute a new candidate pose to evaluate
                const double computeOptimisedPoseTime = static_cast<double>(cv::getTickCount());
                utils::PoseBase candidatePose;
                if (not Pose_Optimization::compute_optimized_global_pose(currentPose, selectedMatches, candidatePose))
#ifndef MAKE_DETERMINISTIC
                    return;
#else
            continue;
#endif
                _meanRANSACPoseOptimizationDuration +=
                        (static_cast<double>(cv::getTickCount()) - computeOptimisedPoseTime) / cv::getTickFrequency();

                // get inliers and outliers for this transformation
                const double getRANSACInliersTime = static_cast<double>(cv::getTickCount());
                matches_containers::match_sets potentialInliersOutliers;
                const double featureInlierScore =
                        get_features_inliers_outliers(matchedFeatures, candidatePose, potentialInliersOutliers);
                _meanRANSACGetInliersDuration +=
                        (static_cast<double>(cv::getTickCount()) - getRANSACInliersTime) / cv::getTickFrequency();

                // We have a better score than the previous best one, and enough inliers to consider it valid
                if (featureInlierScore > maxScore)
                {
                    std::scoped_lock<std::mutex> lock(mut);
                    maxScore = featureInlierScore;
                    bestPose = candidatePose;
                    // save features inliers and outliers
                    finalFeatureSets.swap(potentialInliersOutliers);
                }

                // we have enough features, quit the loop
                // The first guess forces the program to try at least some iterations
                static constexpr size_t minIterations = 3;
                if (iteration >= minIterations and finalFeatureSets._inliers.size() > inliersToStop)
                {
                    canQuit.store(true);
                }
            }
#ifndef MAKE_DETERMINISTIC
    );
#endif

    // We do not have enough inliers to consider this optimization as valid
    const double inlierScore = get_feature_set_optimization_score(finalFeatureSets._inliers);
    if (inlierScore < 1.0)
    {
        outputs::log_error(
                std::format("Could not find a transformation with enough inliers using RANSAC in {} iterations. Final "
                            "inlier score is {}",
                            maximumIterations,
                            inlierScore));

        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    // optimize on all inliers, starting pose is the best pose we found with RANSAC
    const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose_with_covariance(
            bestPose, finalFeatureSets._inliers, finalPose);
    if (isPoseValid)
    {
        // store the result
        featureSets.swap(finalFeatureSets);
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return true;
    }

    // should never happen
    outputs::log_warning("Could not compute a global pose, even when we found a valid inlier set");
    _meanPoseRANSACDuration +=
            (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
    return false;
}

bool Pose_Optimization::compute_optimized_pose(const utils::PoseBase& currentPose,
                                               const matches_containers::match_container& matchedFeatures,
                                               utils::Pose& optimizedPose,
                                               matches_containers::match_sets& featureSets) noexcept
{
    return compute_pose_with_ransac(currentPose, matchedFeatures, optimizedPose, featureSets);
}

bool Pose_Optimization::compute_optimized_pose_coefficients(const utils::PoseBase& currentPose,
                                                            const matches_containers::match_container& matchedFeatures,
                                                            vector6& optimizedCoefficients,
                                                            matrixd& optimizationJacobian) noexcept
{
    // set the input of the optimization function
    vectorxd input = get_optimization_coefficient_from_pose(currentPose);
    if (input.hasNaN() or not input.allFinite())
    {
        outputs::log_error("Coefficient are invalid before optimization");
        return false;
    }

    // get the number of distance coefficients
    size_t optiParts = 0;
    for (const auto& feat: matchedFeatures)
    {
        optiParts += feat->get_feature_part_count();
    }

    // Optimization function (ok to use pointers: optimization of copy)
    Global_Pose_Functor pose_optimisation_functor(Global_Pose_Estimator(optiParts, &matchedFeatures));
    // Optimization algorithm
    Eigen::LevenbergMarquardt poseOptimizator(pose_optimisation_functor);

    // Start optimization (always use it just after the constructor, to ensure pointer validity)
    const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);
    if (endStatus <= 0)
    {
        // Error while optimizing
        outputs::log(std::format("Failed to converge with {} features | Status {}",
                                 matchedFeatures.size(),
                                 get_human_readable_end_message(endStatus)));
        return false;
    }

    if (input.hasNaN() or not input.allFinite())
    {
        outputs::log_error("Coefficient are invalid after optimization");
        return false;
    }

    // get the jacobian of the transformation
    matrixd jacobian(optiParts, 6);
    pose_optimisation_functor.df(input, jacobian);

    // TODO: reactivate when the metric will make sense
    /*
        // Gauss newton approximation: suppose that all residuals are close to 0 (often false)
        matrix66 inputCovariance;
        inputCovariance = (jacobian.transpose() * jacobian).completeOrthogonalDecomposition().pseudoInverse();
        inputCovariance = inputCovariance.selfadjointView<Eigen::Lower>(); // make it symetrical
        inputCovariance.diagonal() += vector6::Constant(1e-4);             // add a small constant for rounding errors

        std::string failureReason;
        if (not utils::is_covariance_valid(inputCovariance, failureReason))
        {
            outputs::log("Initial parameter covariance is invalid: " + failureReason);
            return false;
        }


        // no need to fill the upper part, the adjoint solver does not need it (check only the translation)
        Eigen::SelfAdjointEigenSolver<matrix33> eigenPositionSolver(inputCovariance.block<3, 3>(0, 0));
        // eigen values are sorted by ascending order
        const auto& eigenValuesPosition = eigenPositionSolver.eigenvalues();

        // check that the largest eigen value is inferior to a threshold
        // TODO: find a better threshold principle (this just checks the biggest eigen value)
        if (eigenValuesPosition.hasNaN() or eigenValuesPosition.tail<1>()(0) > 500.0)
        {
            return false;
        }
    */

    optimizedCoefficients = input;
    optimizationJacobian = jacobian;
    return true;
}

bool Pose_Optimization::compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                      const matches_containers::match_container& matchedFeatures,
                                                      utils::PoseBase& optimizedPose,
                                                      matrixd& optimizationJacobian) noexcept
{
    vector6 coefficients;
    if (not compute_optimized_pose_coefficients(currentPose, matchedFeatures, coefficients, optimizationJacobian))
    {
        return false;
    }

    // set the pose
    const auto& outputPose = get_pose_from_optimization_coefficients(coefficients);
    optimizedPose.set_parameters(outputPose.get_position(), outputPose.get_orientation_quaternion());
    return true;
}

bool Pose_Optimization::compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                      const matches_containers::match_container& matchedFeatures,
                                                      utils::PoseBase& optimizedPose) noexcept
{
    matrixd jacobian;
    return compute_optimized_global_pose(currentPose, matchedFeatures, optimizedPose, jacobian);
}

bool Pose_Optimization::compute_optimized_global_pose_with_covariance(
        const utils::PoseBase& currentPose,
        const matches_containers::match_container& matchedFeatures,
        utils::Pose& optimizedPose) noexcept
{
    matrixd jacobian;
    vector6 coefficients;
    if (not compute_optimized_pose_coefficients(currentPose, matchedFeatures, coefficients, jacobian))
    {
        return false;
    }

    if (coefficients.hasNaN())
    {
        outputs::log_error("Coefficient are nan after optimization");
        return false;
    }
    if (not coefficients.allFinite())
    {
        outputs::log_error("Coefficient are infinite after optimization");
        return false;
    }

    Eigen::Matrix<double, 7, 6> coeffToPoseJacobian;
    const utils::PoseBase& outputPose = get_pose_from_optimization_coefficients(coefficients, coeffToPoseJacobian);

    // get real covariance
    matrix66 inputCovariance;
    if (not Global_Pose_Estimator::get_input_covariance(matchedFeatures, outputPose, jacobian, inputCovariance))
    {
        outputs::log_error("get_input_covariance failed");
        return false;
    }

    // convert it to the covariance of the pose
    matrix77 paramCovariance = coeffToPoseJacobian * inputCovariance * coeffToPoseJacobian.transpose();
    paramCovariance.diagonal() += vector7::Constant(1e-4);
    std::string failureReason;
    if (not utils::is_covariance_valid(paramCovariance, failureReason))
    {
        outputs::log_error(std::format("The transformed covariance is invalid, check the jacobian: {}", failureReason));
        return false;
    }

    // transform back to euler angle and position covariance
    const auto& quaternionToEulerJacobian =
            utils::get_position_quaternion_to_position_euler_jacobian(outputPose.get_orientation_quaternion());
    const matrix66& finalPoseCovariance =
            quaternionToEulerJacobian * paramCovariance * quaternionToEulerJacobian.transpose();
    if (not utils::is_covariance_valid(finalPoseCovariance, failureReason))
    {
        outputs::log_error(
                std::format("The optimized pose covariance is invalid, check the jacobian: {}", failureReason));
        return false;
    }

    optimizedPose.set_position_variance(finalPoseCovariance);
    optimizedPose.set_parameters(outputPose.get_position(), outputPose.get_orientation_quaternion());
    return true;
}

void Pose_Optimization::show_statistics(const double meanFrameTreatmentDuration,
                                        const uint frameCount,
                                        const bool shouldDisplayDetails) noexcept
{
    static auto get_percent_of_elapsed_time = [](double treatmentTime, double totalTimeElapsed) {
        if (totalTimeElapsed <= 0)
            return 0.0;
        return (treatmentTime / totalTimeElapsed) * 100.0;
    };

    if (frameCount > 0)
    {
        const double meanPoseRANSACDuration = _meanPoseRANSACDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean pose optimisation with RANSAC time is {:.4f} seconds ({:.2f}%)",
                                 meanPoseRANSACDuration,
                                 get_percent_of_elapsed_time(meanPoseRANSACDuration, meanFrameTreatmentDuration)));
        if (shouldDisplayDetails)
        {
            const double meanGetRandomSubsetDuration = _meanGetRandomSubsetDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\t\tMean pose RANSAC get subset time is {:.4f} seconds ({:.2f}%)",
                                     meanGetRandomSubsetDuration,
                                     get_percent_of_elapsed_time(meanGetRandomSubsetDuration, meanPoseRANSACDuration)));

            const double meanRANSACPoseOptiDuration =
                    _meanRANSACPoseOptimizationDuration / static_cast<double>(frameCount);
            outputs::log(std::format("\t\tMean pose RANSAC optimize pose time is {:.4f} seconds ({:.2f}%)",
                                     meanRANSACPoseOptiDuration,
                                     get_percent_of_elapsed_time(meanRANSACPoseOptiDuration, meanPoseRANSACDuration)));

            const double meanRANSACgetInliersDuration = _meanRANSACGetInliersDuration / static_cast<double>(frameCount);
            outputs::log(
                    std::format("\t\tMean pose RANSAC get inliers time is {:.4f} seconds ({:.2f}%)",
                                meanRANSACgetInliersDuration,
                                get_percent_of_elapsed_time(meanRANSACgetInliersDuration, meanPoseRANSACDuration)));
        }
    }
}

} // namespace rgbd_slam::pose_optimization
