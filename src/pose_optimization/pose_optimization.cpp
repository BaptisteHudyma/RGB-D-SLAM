#include "pose_optimization.hpp"
#include "../../third_party/p3p.hpp"
#include "../outputs/logger.hpp"
#include "../parameters.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/covariances.hpp"
#include "../utils/random.hpp"
#include "coordinates.hpp"
#include "levenberg_marquard_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"
#include "types.hpp"
#include <Eigen/StdVector>
#include <cmath>
#include <string>

namespace rgbd_slam::pose_optimization {

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] pointsToEvaluate The set of points to evaluate the transformation on
 * \param[in] pointMaxRetroprojectionError_px The maximum retroprojection error between two point, below which we
 * classifying the match as inlier
 * \param[in] transformationPose The transformation that needs to be evaluated
 * \param[out] pointMatcheSets The set of inliers/outliers of this transformation
 * \return The transformation score (sum of retroprojection distances)
 */
[[nodiscard]] double get_point_inliers_outliers(const matches_containers::match_point_container& pointsToEvaluate,
                                                const double pointMaxRetroprojectionError_px,
                                                const utils::PoseBase& transformationPose,
                                                matches_containers::point_match_sets& pointMatcheSets) noexcept
{
    pointMatcheSets.clear();

    // get a world to camera transform to evaluate the retroprojection score
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            transformationPose.get_orientation_quaternion(), transformationPose.get_position());

    double retroprojectionScore = 0.0;
    for (const matches_containers::PointMatch& match: pointsToEvaluate)
    {
        // Retroproject world point to screen, and compute screen distance
        const double distance = match._worldFeature.get_distance_px(match._screenFeature, worldToCamera);
        assert(distance >= 0 and not std::isnan(distance));
        // inlier
        if (distance < pointMaxRetroprojectionError_px)
        {
            pointMatcheSets._inliers.insert(pointMatcheSets._inliers.end(), match);
        }
        // outlier
        else
        {
            pointMatcheSets._outliers.insert(pointMatcheSets._outliers.end(), match);
        }
        retroprojectionScore += std::min(pointMaxRetroprojectionError_px, distance);
    }
    return retroprojectionScore;
}

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] planesToEvaluate The set of planes to evaluate the transformation on
 * \param[in] planeMaxRetroprojectionError The maximum retroprojection error between two planes, below which we
 * classifying the match as inlier
 * \param[in] transformationPose The transformation that needs to be evaluated
 * \param[out] planeMatchSets The set of inliers/outliers of this transformation
 * \return The transformation score
 */
[[nodiscard]] double get_plane_inliers_outliers(const matches_containers::match_plane_container& planesToEvaluate,
                                                const double planeMaxRetroprojectionError_mm,
                                                const utils::PoseBase& transformationPose,
                                                matches_containers::plane_match_sets& planeMatchSets) noexcept
{
    planeMatchSets.clear();

    // get a world to camera transform to evaluate the retroprojection score
    const PlaneWorldToCameraMatrix& worldToCamera =
            utils::compute_plane_world_to_camera_matrix(utils::compute_world_to_camera_transform(
                    transformationPose.get_orientation_quaternion(), transformationPose.get_position()));

    double retroprojectionScore = 0.0;
    for (const matches_containers::PlaneMatch& match: planesToEvaluate)
    {
        // Retroproject world point to screen, and compute screen distance
        const double distance =
                match._worldFeature.get_reduced_signed_distance(match._screenFeature, worldToCamera).norm();
        assert(distance >= 0 and not std::isnan(distance));
        // inlier
        if (distance < planeMaxRetroprojectionError_mm)
        {
            planeMatchSets._inliers.insert(planeMatchSets._inliers.end(), match);
        }
        // outlier
        else
        {
            planeMatchSets._outliers.insert(planeMatchSets._outliers.end(), match);
        }
        retroprojectionScore += std::min(planeMaxRetroprojectionError_mm, distance);
    }
    return retroprojectionScore;
}

[[nodiscard]] double get_features_inliers_outliers(const matches_containers::matchContainer& featuresToEvaluate,
                                                   const double pointMaxRetroprojectionError_px,
                                                   const double planeMaxRetroprojectionError_mm,
                                                   const utils::PoseBase& transformationPose,
                                                   matches_containers::match_sets& featureSet) noexcept
{
    return get_point_inliers_outliers(featuresToEvaluate._points,
                                      pointMaxRetroprojectionError_px,
                                      transformationPose,
                                      featureSet._pointSets) +
           get_plane_inliers_outliers(featuresToEvaluate._planes,
                                      planeMaxRetroprojectionError_mm,
                                      transformationPose,
                                      featureSet._planeSets);
}

/**
 * \brief Return a subset of a given inlier set
 */
[[nodiscard]] matches_containers::match_sets get_random_subset(
        const uint numberOfPointsToSample,
        const uint numberOfPlanesToSample,
        const matches_containers::matchContainer& matchedFeatures) noexcept
{
    matches_containers::match_sets matchSubset;
    // we can have a lot of points, so use a more efficient but with potential duplicates subset
    // matchSubset._pointSets._inliers = ransac::get_random_subset_with_duplicates(matchedFeatures._points,
    // numberOfPointsToSample);
    matchSubset._pointSets._inliers = ransac::get_random_subset(matchedFeatures._points, numberOfPointsToSample);
    matchSubset._planeSets._inliers = ransac::get_random_subset(matchedFeatures._planes, numberOfPlanesToSample);
    assert(matchSubset._pointSets._inliers.size() == numberOfPointsToSample);
    assert(matchSubset._planeSets._inliers.size() == numberOfPlanesToSample);

    return matchSubset;
}

bool Pose_Optimization::compute_pose_with_ransac(const utils::PoseBase& currentPose,
                                                 const matches_containers::matchContainer& matchedFeatures,
                                                 utils::PoseBase& finalPose,
                                                 matches_containers::match_sets& featureSets) noexcept
{
    featureSets.clear();

    const double matchedPointSize = static_cast<double>(matchedFeatures._points.size());
    const double matchedPlaneSize = static_cast<double>(matchedFeatures._planes.size());

    constexpr uint minimumPointsForOptimization =
            parameters::optimization::minimumPointForOptimization; // Number of random points to select
    constexpr uint minimumPlanesForOptimization =
            parameters::optimization::minimumPlanesForOptimization; // Number of random planes to select
    static_assert(minimumPointsForOptimization > 0);
    static_assert(minimumPlanesForOptimization > 0);

    // individual feature score
    constexpr double pointFeatureScore = 1.0 / minimumPointsForOptimization;
    constexpr double planeFeatureScore = 1.0 / minimumPlanesForOptimization;

    // check that we have enough features for minimal pose optimization
    const double initialFeatureScore = pointFeatureScore * matchedPointSize + planeFeatureScore * matchedPlaneSize;
    if (initialFeatureScore < 1.0)
    {
        // if there is not enough potential inliers to optimize a pose
        outputs::log_warning(std::format("Not enough features to optimize a pose ({} points, {} planes)",
                                         static_cast<int>(matchedPointSize),
                                         static_cast<int>(matchedPlaneSize)));
        return false;
    }

    constexpr double pointMaxRetroprojectionError_px =
            parameters::optimization::ransac::maximumRetroprojectionErrorForPointInliers_px; // maximum inlier threshold
    constexpr double planeMaxRetroprojectionError_mm =
            parameters::optimization::ransac::maximumRetroprojectionErrorForPlaneInliers_mm; // maximum inlier threshold
    static_assert(pointMaxRetroprojectionError_px > 0);
    static_assert(planeMaxRetroprojectionError_mm > 0);
    const uint acceptablePointInliersForEarlyStop = static_cast<uint>(
            matchedPointSize *
            parameters::optimization::ransac::minimumInliersProportionForEarlyStop); // RANSAC will stop early if
                                                                                     // this inlier count is
                                                                                     // reached
    const uint acceptablePlaneInliersForEarlyStop = static_cast<uint>(
            matchedPlaneSize *
            parameters::optimization::ransac::minimumInliersProportionForEarlyStop); // RANSAC will stop early if
                                                                                     // this inlier count is
                                                                                     // reached

    // check that we have enough inlier features for a pose optimization with RANSAC
    // This score is to stop the RANSAC process early (limit to 1 if we are low on features)
    const double enoughInliersScore = std::max(1.0,
                                               pointFeatureScore * acceptablePointInliersForEarlyStop +
                                                       planeFeatureScore * acceptablePlaneInliersForEarlyStop);

    // get the min and max values of planes and points to select
    const uint maxNumberOfPoints = std::min(minimumPointsForOptimization, (uint)matchedPointSize);
    const uint maxNumberOfPlanes = std::min(minimumPlanesForOptimization, (uint)matchedPlaneSize);
    const uint minNumberOfPlanes =
            static_cast<uint>(std::ceil((1.0 - maxNumberOfPoints * pointFeatureScore) / planeFeatureScore));
    const uint minNumberOfPoints =
            static_cast<uint>(std::ceil((1.0 - maxNumberOfPlanes * planeFeatureScore) / pointFeatureScore));

    // Compute maximum iteration with the original RANSAC formula
    const uint maximumIterations = static_cast<uint>(std::ceil(
            log(1.0 - parameters::optimization::ransac::probabilityOfSuccess) /
            log(1.0 - pow(parameters::optimization::ransac::inlierProportion, minimumPointsForOptimization))));
    assert(maximumIterations > 0);

    // set the start score to the maximum score
    double minScore =
            matchedPointSize * pointMaxRetroprojectionError_px + matchedPlaneSize * planeMaxRetroprojectionError_mm;
    utils::PoseBase bestPose = currentPose;
    for (uint iteration = 0; iteration < maximumIterations; ++iteration)
    {
        // get random number of planes, between minNumberOfPlanes and maxNumberOfPlanes
        const uint numberOfPlanesToSample = minNumberOfPlanes + (maxNumberOfPlanes - minNumberOfPlanes) *
                                                                        (utils::Random::get_random_double() > 0.5);
        // depending on this number of planes, get a number of points to sample for this RANSAC iteration
        const uint numberOfPointsToSample =
                static_cast<uint>(std::ceil((1 - numberOfPlanesToSample * planeFeatureScore) / pointFeatureScore));

        const double subsetScore =
                numberOfPointsToSample * pointFeatureScore + numberOfPlanesToSample * planeFeatureScore;
        if (subsetScore < 1.0)
        {
            outputs::log_warning(
                    std::format("Selected {} points and {} planes, not enough for optimization (score: {})",
                                numberOfPointsToSample,
                                numberOfPlanesToSample,
                                subsetScore));
            continue;
        }
        if (numberOfPlanesToSample < minNumberOfPlanes or numberOfPlanesToSample > maxNumberOfPlanes or
            numberOfPlanesToSample > matchedPlaneSize)
        {
            outputs::log_warning(std::format(
                    "Selected {} planes but we have {} available", numberOfPlanesToSample, matchedPointSize));
            continue;
        }
        if (numberOfPointsToSample < minNumberOfPoints or numberOfPointsToSample > maxNumberOfPoints or
            numberOfPointsToSample > matchedPointSize)
        {
            outputs::log_warning(std::format(
                    "Selected {} points but we have {} available", numberOfPointsToSample, matchedPlaneSize));
            continue;
        }

        const matches_containers::match_sets& selectedMatches =
                get_random_subset(numberOfPointsToSample, numberOfPlanesToSample, matchedFeatures);

        // compute a new candidate pose to evaluate
        utils::PoseBase candidatePose;
        if (not Pose_Optimization::compute_optimized_global_pose(currentPose, selectedMatches, candidatePose))
            continue;

        // get inliers and outliers for this transformation
        matches_containers::match_sets potentialInliersOutliers;
        const double transformationScore = get_features_inliers_outliers(matchedFeatures,
                                                                         pointMaxRetroprojectionError_px,
                                                                         planeMaxRetroprojectionError_mm,
                                                                         candidatePose,
                                                                         potentialInliersOutliers);
        // We have a better score than the previous best one
        if (transformationScore < minScore)
        {
            minScore = transformationScore;
            bestPose = candidatePose;
            // save features inliers and outliers
            featureSets.swap(potentialInliersOutliers);

            const double inlierScore = static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore +
                                       static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
            if (inlierScore >= enoughInliersScore)
            {
                // We can stop here, the optimization is good enough
                break;
            }
        }
    }

    // We do not have enough inliers to consider this optimization as valid
    const double inlierScore = static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore +
                               static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
    if (inlierScore < 1.0)
    {
        outputs::log_warning("Could not find a transformation with enough inliers using RANSAC");
        return false;
    }

    // optimize on all inliers, starting pose is the best pose we found with RANSAC
    const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(bestPose, featureSets, finalPose);
    if (isPoseValid)
    {
        return true;
    }
    // should never happen
    outputs::log_warning("Could not compute a global pose, even though we found a valid inlier set");
    return false;
}

bool Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose,
                                               const matches_containers::matchContainer& matchedFeatures,
                                               utils::Pose& optimizedPose,
                                               matches_containers::match_sets& featureSets) noexcept
{
    if (compute_pose_with_ransac(currentPose, matchedFeatures, optimizedPose, featureSets))
    {
        // Compute pose variance
        if (matrix66 estimatedPoseCovariance;
            compute_pose_variance(optimizedPose, featureSets, estimatedPoseCovariance))
        {
            optimizedPose.set_position_variance(estimatedPoseCovariance);
            return true;
        }
        else
        {
            outputs::log_warning("Could not compute pose variance");
        }
    }
    return false;
}

bool Pose_Optimization::compute_optimized_global_pose(const utils::PoseBase& currentPose,
                                                      const matches_containers::match_sets& matchedFeatures,
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

    // Optimization function
    Global_Pose_Functor pose_optimisation_functor(
            Global_Pose_Estimator(matchedFeatures._pointSets._inliers, matchedFeatures._planeSets._inliers));
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

    // Start optimization
    const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);

    // Get result
    const quaternion& endRotation = get_quaternion_from_scale_axis_coefficients(vector3(input[3], input[4], input[5]));
    const vector3 endPosition(input[0], input[1], input[2]);

    if (endStatus <= 0)
    {
        // Error while optimizing
        outputs::log(std::format("Failed to converge with {} points | Status {}",
                                 matchedFeatures._pointSets._inliers.size(),
                                 get_human_readable_end_message(endStatus)));
        return false;
    }

    // Update refined pose with optimized pose
    optimizedPose.set_parameters(endPosition, endRotation);
    return true;
}

bool Pose_Optimization::compute_pose_variance(const utils::PoseBase& optimizedPose,
                                              const matches_containers::match_sets& matchedFeatures,
                                              matrix66& poseCovariance,
                                              const uint iterations) noexcept
{
    assert(iterations > 0);
    poseCovariance.setZero();
    vector6 medium = vector6::Zero();

    std::vector<vector6> poses;
    poses.reserve(iterations);
    for (uint i = 0; i < iterations; ++i)
    {
        utils::PoseBase newPose;
        if (compute_random_variation_of_pose(optimizedPose, matchedFeatures, newPose))
        {
            const vector6& pose6dof = newPose.get_vector();
            medium += pose6dof;
            poses.emplace_back(pose6dof);
        }
        else
        {
            outputs::log_warning(std::format("fail iteration {}: rejected pose optimization", i));
        }
    }
    if (poses.size() < iterations / 2)
    {
        outputs::log_error("Could not compute covariance: too many faileds iterations");
        return false;
    }
    medium /= static_cast<double>(poses.size());

    for (const vector6& pose: poses)
    {
        const vector6 def = pose - medium;
        poseCovariance += def * def.transpose();
    }
    poseCovariance /= static_cast<double>(poses.size() - 1);

    return true;
}

bool Pose_Optimization::compute_random_variation_of_pose(const utils::PoseBase& currentPose,
                                                         const matches_containers::match_sets& matchedFeatures,
                                                         utils::PoseBase& optimizedPose) noexcept
{
    matches_containers::match_sets variatedSet;
    for (const matches_containers::PointMatch& match: matchedFeatures._pointSets._inliers)
    {
        // make random variation
        utils::WorldCoordinate variatedCoordinates = match._worldFeature;
        variatedCoordinates +=
                utils::Random::get_normal_doubles<3>().cwiseProduct(match._worldFeatureCovariance.cwiseSqrt());

        // add to the new match set
        variatedSet._pointSets._inliers.emplace_back(
                match._screenFeature, variatedCoordinates, match._worldFeatureCovariance, match._idInMap);
    }
    for (const matches_containers::PlaneMatch& match: matchedFeatures._planeSets._inliers)
    {
        utils::PlaneWorldCoordinates variatedCoordinates = match._worldFeature;

        const vector4& diagonalSqrt = match._worldFeatureCovariance.diagonal().cwiseSqrt();
        variatedCoordinates.normal() += utils::Random::get_normal_doubles<3>().cwiseProduct(diagonalSqrt.head<3>());
        variatedCoordinates.normal().normalize();

        variatedCoordinates.d() += utils::Random::get_normal_double() * diagonalSqrt(3);

        variatedSet._planeSets._inliers.emplace_back(
                match._screenFeature, variatedCoordinates, match._worldFeatureCovariance, match._idInMap);
    }

    return compute_optimized_global_pose(currentPose, variatedSet, optimizedPose);
}

} // namespace rgbd_slam::pose_optimization
