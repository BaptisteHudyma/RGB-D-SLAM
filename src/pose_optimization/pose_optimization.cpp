#include "pose_optimization.hpp"

#include "covariances.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "levenberg_marquardt_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"
#include "types.hpp"

#include "utils/camera_transformation.hpp"

#include <Eigen/StdVector>
#include <algorithm>
#include <cmath>
#include <exception>
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
[[nodiscard]] std::pair<double, double> get_features_inliers_outliers(
        const matches_containers::match_container& featuresToEvaluate,
        const utils::PoseBase& transformationPose,
        matches_containers::match_sets& matchSets) noexcept
{
    matchSets.clear();

    // get a world to camera transform to evaluate the retroprojection score
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            transformationPose.get_orientation_quaternion(), transformationPose.get_position());

    double retroprojectionScore = 0.0;
    double featureScore = 0.0;
    for (const auto& match: featuresToEvaluate)
    {
        // TODO: handle each component separatly ?
        const double maxRetroprojectionError = match->get_max_retroprojection_error();

        // Retroproject world point to screen, and compute screen distance
        try
        {
            // get the feature distance to it's match
            const double distance = match->get_distance(worldToCamera).norm();
            // inlier
            if (distance < maxRetroprojectionError)
            {
                // TODO: there is a bias toward feature that are numerous but low score (eg: points)
                // Planes should be considered to be much more important than points (higher level of feature)
                matchSets._inliers.insert(matchSets._inliers.end(), match);
                retroprojectionScore += distance;
                featureScore += match->get_score();
            }
            // outlier
            else
            {
                matchSets._outliers.insert(matchSets._outliers.end(), match);
                // outliers are still taken into account (MSAC)
                retroprojectionScore += maxRetroprojectionError;
            }
        }
        catch (const std::exception& ex)
        {
            outputs::log_error("get_features_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
            matchSets._outliers.insert(matchSets._outliers.end(), match);
            retroprojectionScore += maxRetroprojectionError;
        }
    }
    return std::make_pair(retroprojectionScore, featureScore);
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
 * \brief compute the score of a feature set
 */
double get_feature_set_retroprojection_score(const matches_containers::match_container& matchedFeatures)
{
    double score = 0.0;
    for (const auto& match: matchedFeatures)
    {
        score += match->get_max_retroprojection_error();
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

    // set the start score to the maximum score
    const double maxFittingScore = get_feature_set_retroprojection_score(matchedFeatures);

    // TODO: check the constant: 60% inlier proportion is weak
    // matchedFeatures.size() * parameters::optimization::ransac::inlierProportion);
    const size_t inliersToStop = matchedFeatures.size() * 0.80;

    double maxScore = 1.0;
    utils::PoseBase bestPose = currentPose;
    matches_containers::match_sets finalFeatureSets;

    std::atomic_bool canQuit = false;
    std::mutex mut;

    uint iteration = 0;
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
    for (; iteration < maximumIterations; ++iteration)
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
                const auto& transformationScores =
                        get_features_inliers_outliers(matchedFeatures, candidatePose, potentialInliersOutliers);
                const double transformationScore = transformationScores.first;
                const double featureInlierScore = transformationScores.second;
                _meanRANSACGetInliersDuration +=
                        (static_cast<double>(cv::getTickCount()) - getRANSACInliersTime) / cv::getTickFrequency();

                // safety
                if (transformationScore > maxFittingScore)
                {
                    outputs::log_error("The computed score is higher than the max fitting score");
#ifndef MAKE_DETERMINISTIC
                    return;
#else
            continue;
#endif
                }

                // We have a better score than the previous best one, and enough inliers to consider it valid
                if (featureInlierScore > 1.0 and featureInlierScore > maxScore)
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
        outputs::log_error(std::format(
                "Could not find a transformation with enough inliers using RANSAC in {} over {} iterations. Final "
                "inlier score is {}",
                iteration,
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
    return false;
}

bool Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose,
                                               const matches_containers::match_container& matchedFeatures,
                                               utils::Pose& optimizedPose,
                                               matches_containers::match_sets& featureSets) noexcept
{
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
    const vector3& position = currentPose.get_position(); // Work in millimeters
    const quaternion& rotation = currentPose.get_orientation_quaternion();

    // Vector to optimize: (0, 1, 2) is position,
    // Vector (3, 4, 5) is a rotation parametrization, representing a delta in rotation in the tangential hyperplane
    // -From Using Quaternions for Parametrizing 3-D Rotation in Unconstrained Nonlinear Optimization)
    vectorxd input(6);
    // 3D pose
    input[0] = position.x();
    input[1] = position.y();
    input[2] = position.z();
    // X Y Z of a quaternion representation. (0, 0, 0) corresponds to the quaternion itself
    const vector3& rotationCoefficients = get_scaled_axis_coefficients_from_quaternion(rotation);
    input[3] = rotationCoefficients.x();
    input[4] = rotationCoefficients.y();
    input[5] = rotationCoefficients.z();

    size_t optiParts = 0;
    for (const auto& feat: matchedFeatures)
    {
        optiParts += feat->get_feature_part_count();
    }

    // Optimization function (ok to use pointers: optimization of copy)
    Global_Pose_Functor pose_optimisation_functor(Global_Pose_Estimator(optiParts, &matchedFeatures));
    // Optimization algorithm
    Eigen::LevenbergMarquardt poseOptimizator(pose_optimisation_functor);

    // maxfev   : maximum number of function evaluation
    poseOptimizator.parameters.maxfev = parameters::optimization::maximumIterations;
    // epsfcn   : error precision
    poseOptimizator.parameters.epsfcn = parameters::optimization::errorPrecision;
    // xtol     : tolerance for the norm of the solution vector
    poseOptimizator.parameters.xtol = parameters::optimization::toleranceOfSolutionVectorNorm;
    // ftol     : tolerance for the norm of the vector function
    poseOptimizator.parameters.ftol = parameters::optimization::toleranceOfVectorFunction;
    // gtol     : tolerance for the norm of the gradient of the error function
    poseOptimizator.parameters.gtol = parameters::optimization::toleranceOfErrorFunctionGradient;
    // factor   : step bound for the diagonal shift
    poseOptimizator.parameters.factor = parameters::optimization::diagonalStepBoundShift;

    // Start optimization (always use it just after the constructor, to ensure pointer validity)
    const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);

    // Get result
    const quaternion& endRotation = get_quaternion_from_scale_axis_coefficients(vector3(input[3], input[4], input[5]));
    const vector3 endPosition(input[0], input[1], input[2]);

    if (endStatus <= 0)
    {
        // Error while optimizing
        outputs::log(std::format("Failed to converge with {} features | Status {}",
                                 matchedFeatures.size(),
                                 get_human_readable_end_message(endStatus)));
        return false;
    }

    // Update refined pose with optimized pose
    optimizedPose.set_parameters(endPosition, endRotation);
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

    if (not utils::is_covariance_valid(poseCovariance))
    {
        outputs::log_error("Could not compute covariance: final covariance is ill formed");
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
    }

    return compute_optimized_global_pose(currentPose, variatedSet, optimizedPose);
}

} // namespace rgbd_slam::pose_optimization
