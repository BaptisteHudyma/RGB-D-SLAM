#include "pose_optimization.hpp"

#include "covariances.hpp"
#include "distance_utils.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "levenberg_marquardt_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"
#include "types.hpp"

#include "utils/camera_transformation.hpp"

#include <Eigen/StdVector>
#include <cmath>
#include <exception>
#include <opencv2/core/utility.hpp>
#include <stdexcept>
#include <string>
#include <format>

#include <tbb/parallel_for.h>

namespace rgbd_slam::pose_optimization {

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] featuresToEvaluate The set of features to evaluate the transformation on
 * \param[in] transformationPose The transformation that needs to be evaluated
 * \param[out] matchSets The set of inliers/outliers of this transformation
 * \return The feature score
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
        bool isInlier = false;
        // Retroproject world feature to screen, and compute screen distance
        try
        {
            // this may throw on rare occasions
            isInlier = match->is_inlier(worldToCamera);
        }
        catch (const std::exception& ex)
        {
            outputs::log_error("get_features_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
        }

        // inlier
        if (isInlier)
        {
            matchSets._inliers.insert(matchSets._inliers.end(), match);
            featureScore += match->get_score();
        }
        // not an inlier, add to ouliers
        else
        {
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

    // we can have a lot of features, so use a more efficient but with potential duplicates subset
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
                                                 utils::PoseBase& finalPose,
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
    static constexpr uint maximumIterations =
            (std::ceil(std::log(1.0f - parameters::optimization::ransac::probabilityOfSuccess) /
                       std::log(1.0f - std::pow(parameters::optimization::ransac::inlierProportion,
                                                parameters::optimization::ransac::featureTrustCount))));
    if (maximumIterations <= 0)
    {
        outputs::log_error("maximumIterations should be > 0, no pose optimization will be made");
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    const size_t inliersToStop = (size_t)std::ceil(
            matchedFeatures.size() * parameters::optimization::ransac::minimumInliersProportionForEarlyStop);

    double maxScore = 1.0; // 1.0 is the minimum score we can have for a set of matches to optimize a pose
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

                // TODO: refuse a random subset if it is illformed, or uses the same map/detected feature id
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

                // optimization failed, not enough inliers
                if (featureInlierScore < 1.0)
                {
#ifndef MAKE_DETERMINISTIC
                    return;
#else
            continue;
#endif
                }

                // Better score, or same score but more inliers
                const bool canOverload = (featureInlierScore > maxScore) or
                                         (utils::double_equal(featureInlierScore, maxScore, 0.1) and
                                          finalFeatureSets._inliers.size() < potentialInliersOutliers._inliers.size());
                if (canOverload)
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
    const bool isPoseValid =
            Pose_Optimization::compute_optimized_global_pose(bestPose, finalFeatureSets._inliers, finalPose);
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
    // TODO: maybe we could use a reduced result, instead of the reoptimization result
    return false;
}

bool Pose_Optimization::compute_optimized_pose(const utils::PoseBase& currentPose,
                                               const matches_containers::match_container& matchedFeatures,
                                               utils::Pose& optimizedPose,
                                               matches_containers::match_sets& featureSets) noexcept
{
    // check every feature for validity
    bool success = true;
    for (const auto& feature: matchedFeatures)
    {
        if (not feature->is_valid())
        {
            success = false;
            outputs::log_error("feature is invalid: " + to_string(feature->get_feature_type()));
        }
    }
    if (not success)
    {
        return false;
    }

    // compute an optimized pose with a random sample consensus of the feature matches
    if (compute_pose_with_ransac(currentPose, matchedFeatures, optimizedPose, featureSets))
    {
        // Compute pose variance
        if (matrix66 estimatedPoseCovariance;
            compute_pose_variance(optimizedPose, featureSets._inliers, estimatedPoseCovariance))
        {
            optimizedPose.set_position_variance(estimatedPoseCovariance);
            return true;
        }
        else
        {
            outputs::log_warning("Could not compute pose variance after succesful optimization");
        }
    }
    return false;
}

bool Pose_Optimization::compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                      const matches_containers::match_container& matchedFeatures,
                                                      utils::PoseBase& optimizedPose) noexcept
{
    // set the input of the optimization function
    vectorxd input = get_optimization_coefficient_from_pose(currentPose);
    if (input.hasNaN() or not input.allFinite())
    {
        outputs::log_error("position as invalid values after transformation in optimization space");
        return false;
    }

    // get the number of distance coefficients, and optimization parts
    double optimizationScore = 0.0;
    size_t optiParts = 0;
    for (const auto& feat: matchedFeatures)
    {
        optiParts += feat->get_feature_part_count();
        optimizationScore += feat->get_score();
    }
    if (input.cols() > 0 and optiParts <= static_cast<size_t>(input.cols()))
    {
        outputs::log_error("Not enought feature parts to optimize for a pose");
        return false;
    }
    if (optimizationScore < 1.0)
    {
        outputs::log_error("Not enought features to optimize for a pose: " + std::to_string(optimizationScore));
        return false;
    }

    // Optimization function
    Global_Pose_Functor pose_optimisation_functor {Global_Pose_Estimator {optiParts, matchedFeatures}};
    // Optimization algorithm
    Eigen::LevenbergMarquardt poseOptimizator(pose_optimisation_functor);

    // Start optimization (always use it just after the constructor, to ensure feature object reference validity)
    const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);
    if (endStatus <= 0)
    {
        // Error while optimizing
        outputs::log(std::format("Failed to converge with {} features | Status {}",
                                 matchedFeatures.size(),
                                 get_human_readable_end_message(endStatus)));
        return false;
    }

    const auto& outputPose = get_pose_from_optimization_coefficients(input);
    if (outputPose.get_vector().hasNaN())
    {
        outputs::log_error("optimized pose contains invalid values");

        return false;
    }
    // Update refined pose with optimized pose
    optimizedPose.set_parameters(outputPose.get_position(), outputPose.get_orientation_quaternion());
    return true;
}

bool Pose_Optimization::compute_pose_variance(const utils::PoseBase& optimizedPose,
                                              const matches_containers::match_container& matchedFeatures,
                                              matrix66& poseCovariance,
                                              const uint iterations) noexcept
{
    const double computePoseVarianceStartTime = static_cast<double>(cv::getTickCount());

    if (iterations == 0)
    {
        outputs::log_error("Cannot compute pose variance with 0 iterations");
        return false;
    }
    poseCovariance.setZero();
    vector6 medium = vector6::Zero();

    std::vector<vector6> poses;
    poses.reserve(iterations);

#ifndef MAKE_DETERMINISTIC
    std::mutex mut;
    tbb::parallel_for(uint(0),
                      iterations,
                      [&](uint i)
#else
    for (uint i = 0; i < iterations; ++i)
#endif
                      {
                          std::ignore = i;
                          utils::PoseBase newPose;
                          if (compute_random_variation_of_pose(optimizedPose, matchedFeatures, newPose))
                          {
                              const vector6& pose6dof = newPose.get_vector();
#ifndef MAKE_DETERMINISTIC
                              std::scoped_lock<std::mutex> lock(mut);
#endif
                              medium += pose6dof;
                              poses.emplace_back(pose6dof);
                          }
                          else
                          {
                              outputs::log_warning(std::format("fail iteration {}: rejected pose optimization", i));
                          }
                      }
#ifndef MAKE_DETERMINISTIC
    );
#endif

    if (poses.size() < iterations / 2)
    {
        outputs::log_error("Could not compute covariance: too many faileds iterations");
        _meanComputePoseVarianceDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseVarianceStartTime) / cv::getTickFrequency();
        return false;
    }
    medium /= static_cast<double>(poses.size());

    for (const vector6& pose: poses)
    {
        const vector6 def = pose - medium;
        poseCovariance += def * def.transpose();
    }
    poseCovariance /= static_cast<double>(poses.size() - 1);
    poseCovariance.diagonal() += vector6::Constant(
            0.001); // add small variance on diagonal in case of perfect covariance (rare but existing case)

    std::string errorMsg;
    if (not utils::is_covariance_valid(poseCovariance, errorMsg))
    {
        outputs::log_error("Could not compute covariance: final covariance is ill formed: " + errorMsg);
        _meanComputePoseVarianceDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseVarianceStartTime) / cv::getTickFrequency();
        return false;
    }
    _meanComputePoseVarianceDuration +=
            (static_cast<double>(cv::getTickCount()) - computePoseVarianceStartTime) / cv::getTickFrequency();
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

        const double meanPoseVarianceDuration = _meanComputePoseVarianceDuration / static_cast<double>(frameCount);
        outputs::log(std::format("\tMean pose variance computation time is {:.4f} seconds ({:.2f}%)",
                                 meanPoseVarianceDuration,
                                 get_percent_of_elapsed_time(meanPoseVarianceDuration, meanFrameTreatmentDuration)));
    }
}

bool Pose_Optimization::compute_random_variation_of_pose(const utils::PoseBase& currentPose,
                                                         const matches_containers::match_container& matchedFeatures,
                                                         utils::PoseBase& optimizedPose) noexcept
{
    matches_containers::match_container variatedSet;
    for (const auto& feature: matchedFeatures)
    {
        // add to the new match set
        variatedSet.emplace_back(feature->compute_random_variation());
#if 0 // this check gets expensive, activate it only if needed
        const auto& variated = variatedSet.back();
        if (not variated->is_valid())
        {
            outputs::log_error("a variated coordinate became invalid: " + to_string(feature->get_feature_type()));
        }
#endif
    }

    return compute_optimized_global_pose(currentPose, variatedSet, optimizedPose);
}

} // namespace rgbd_slam::pose_optimization
