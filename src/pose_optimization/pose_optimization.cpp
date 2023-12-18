#include "pose_optimization.hpp"

#include "covariances.hpp"
#include "outputs/logger.hpp"
#include "parameters.hpp"
#include "levenberg_marquard_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"
#include "types.hpp"

#include "utils/camera_transformation.hpp"
#include "utils/random.hpp"
#include "utils/coordinates/point_coordinates.hpp"
#include "utils/coordinates/plane_coordinates.hpp"

#include <Eigen/StdVector>
#include <algorithm>
#include <array>
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

constexpr std::array<uint, numberOfFeatures> minNumberOfFeatureForOpti = {
        parameters::optimization::minimumPlanesForOptimization,
        parameters::optimization::minimumPointForOptimization,
        parameters::optimization::minimumPointForOptimization};
constexpr std::array<double, numberOfFeatures> scorePerFeature = {
        1.0 / minNumberOfFeatureForOpti[0], 1.0 / minNumberOfFeatureForOpti[1], 1.0 / minNumberOfFeatureForOpti[2]};

std::array<std::pair<uint, uint>, numberOfFeatures> get_min_max_number_of_features(
        const std::array<uint, numberOfFeatures>& numberOfFeature) noexcept
{
    std::array<std::pair<uint, uint>, numberOfFeatures> minMaxFeatures;
    // set the max
    for (size_t i = 0; i < numberOfFeatures; ++i)
    {
        minMaxFeatures[i].second = std::min(minNumberOfFeatureForOpti[i], numberOfFeature[i]);
    }

    // set min for features
    for (size_t i = 0; i < numberOfFeatures; ++i)
    {
        // feature index loop
        double otherFeatureScore = 0.0;
        for (size_t j = 0; j < numberOfFeatures; j++)
        {
            // other feature index loop
            if (i == j)
                continue;
            otherFeatureScore += minMaxFeatures[j].second * scorePerFeature[j];
        }

        const double missingFeatureScore = 1.0 - std::clamp(otherFeatureScore, 0.0, 1.0);
        minMaxFeatures[i].first = std::ceil(missingFeatureScore / scorePerFeature[i]);
    }
    return minMaxFeatures;
}

/*
 * \brief Select a random subset of features to perform a pose optimisation
 * TODO improve performances
 */
std::array<uint, numberOfFeatures> get_random_selection(
        const std::array<uint, numberOfFeatures>& featuresCounts,
        const std::array<std::pair<uint, uint>, numberOfFeatures>& minMaxNumberOfFeatures) noexcept
{
    std::array<uint, numberOfFeatures> selection = {0, 0, 0};

    double scoreAccumulation = 0.0;
    // set the minimum features
    for (size_t i = 0; i < numberOfFeatures; ++i)
    {
        const auto& minmax = minMaxNumberOfFeatures[i];
        scoreAccumulation += minmax.first * scorePerFeature[i];
        selection[i] = minmax.first;
    }

    // fill the remaining space with random features
    static constexpr size_t MAX_ITERATIONS = 10;
    for (size_t currentIteration = 0; (scoreAccumulation < 1.0) and (currentIteration < MAX_ITERATIONS);
         ++currentIteration)
    {
        for (size_t i = 0; i < numberOfFeatures; ++i)
        {
            const auto select = selection[i];
            const auto& minmax = minMaxNumberOfFeatures[i];
            const double scoreLeft = 1.0 - std::clamp(scoreAccumulation, 0.0, 1.0);
            const uint selectedFeatureCount =
                    (uint)std::ceil(utils::Random::get_normal_double() * scoreLeft / scorePerFeature[i]);
            const uint newVal = std::clamp(select + selectedFeatureCount, minmax.first, minmax.second);
            if (newVal > select)
            {
                scoreAccumulation += (newVal - select) * scorePerFeature[i];
                selection[i] = newVal;
            }
        }
    }

    // fill the remaining features (maybe not needed)
    if (scoreAccumulation < 1.0)
    {
        outputs::log_warning("filling remaining features with non random loop");
        for (size_t i = 0; i < numberOfFeatures; ++i)
        {
            const auto select = selection[i];
            const auto& minmax = minMaxNumberOfFeatures[i];
            const uint maxFeatures = minmax.second;
            if (maxFeatures > select)
            {
                scoreAccumulation += (maxFeatures - select) * scorePerFeature[i];
                selection[i] = maxFeatures;
            }
        }
    }

    // sanity check loop
    for (size_t i = 0; i < numberOfFeatures; ++i)
    {
        const auto& minmax = minMaxNumberOfFeatures[i];
        const auto select = selection[i];
        if (select < minmax.first or select > minmax.second or select > featuresCounts[i])
        {
            outputs::log_warning(
                    std::format("Selected {} feature at index {} but we have {} available [min {}, max {}]",
                                select,
                                i,
                                featuresCounts[i],
                                minmax.first,
                                minmax.second));
        }
    }

    return selection;
}

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] pointsToEvaluate The set of points to evaluate the transformation on
 * \param[in] point2dMaxRetroprojectionError_px The maximum retroprojection error between two point, below which we
 * classifying the match as inlier
 * \param[in] transformationPose The transformation that needs to be evaluated
 * \param[out] pointMatcheSets The set of inliers/outliers of this transformation
 * \return The transformation score (sum of retroprojection distances)
 */
[[nodiscard]] double get_2Dpoint_inliers_outliers(const matches_containers::match_point2D_container& pointsToEvaluate,
                                                  const double point2dMaxRetroprojectionError_px,
                                                  const utils::PoseBase& transformationPose,
                                                  matches_containers::point2D_match_sets& pointMatcheSets) noexcept
{
    pointMatcheSets.clear();

    // get a world to camera transform to evaluate the retroprojection score
    const WorldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(
            transformationPose.get_orientation_quaternion(), transformationPose.get_position());

    double retroprojectionScore = 0.0;
    for (const matches_containers::PointMatch2D& match: pointsToEvaluate)
    {
        // Retroproject world point to screen, and compute screen distance
        try
        {
            const double distance =
                    match._worldFeature.compute_signed_screen_distance(match._screenFeature, worldToCamera).lpNorm<1>();
            // inlier
            if (distance < point2dMaxRetroprojectionError_px)
            {
                pointMatcheSets._inliers.insert(pointMatcheSets._inliers.end(), match);
            }
            // outlier
            else
            {
                pointMatcheSets._outliers.insert(pointMatcheSets._outliers.end(), match);
            }
            retroprojectionScore += std::min(point2dMaxRetroprojectionError_px, distance);
        }
        catch (const std::exception& ex)
        {
            // treat as outlier
            outputs::log_error("get_2Dpoint_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
            pointMatcheSets._outliers.insert(pointMatcheSets._outliers.end(), match);
            retroprojectionScore += point2dMaxRetroprojectionError_px;
        }
    }
    return retroprojectionScore;
}

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
        try
        {
            const double distance = match._worldFeature.get_distance_px(match._screenFeature, worldToCamera);
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
        catch (const std::exception& ex)
        {
            // treat as outlier
            outputs::log_error("get_point_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
            pointMatcheSets._outliers.insert(pointMatcheSets._outliers.end(), match);
            retroprojectionScore += pointMaxRetroprojectionError_px;
        }
    }
    return retroprojectionScore;
}

/**
 * \brief Compute a score for a transformation, and compute an inlier and outlier set
 * \param[in] planesToEvaluate The set of planes to evaluate the transformation on
 * \param[in] planeMaxRetroprojectionError_mm The maximum retroprojection error between two planes, below which we
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
        try
        {
            const double distance =
                    match._worldFeature.get_reduced_signed_distance(match._screenFeature, worldToCamera).norm();
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
        catch (const std::exception& ex)
        {
            outputs::log_error("get_plane_inliers_outliers: caught exeption while computing distance: " +
                               std::string(ex.what()));
            planeMatchSets._outliers.insert(planeMatchSets._outliers.end(), match);
            retroprojectionScore += planeMaxRetroprojectionError_mm;
        }
    }
    return retroprojectionScore;
}

[[nodiscard]] double get_features_inliers_outliers(const matches_containers::matchContainer& featuresToEvaluate,
                                                   const double point2dMaxRetroprojectionError_px,
                                                   const double pointMaxRetroprojectionError_px,
                                                   const double planeMaxRetroprojectionError_mm,
                                                   const utils::PoseBase& transformationPose,
                                                   matches_containers::match_sets& featureSet) noexcept
{
    return get_2Dpoint_inliers_outliers(featuresToEvaluate._points2D,
                                        point2dMaxRetroprojectionError_px,
                                        transformationPose,
                                        featureSet._point2DSets) +
           get_point_inliers_outliers(featuresToEvaluate._points,
                                      pointMaxRetroprojectionError_px,
                                      transformationPose,
                                      featureSet._pointSets) +
           get_plane_inliers_outliers(featuresToEvaluate._planes,
                                      planeMaxRetroprojectionError_mm,
                                      transformationPose,
                                      featureSet._planeSets);
}

/**
 * \brief Return a subset of features
 */
[[nodiscard]] matches_containers::match_sets get_random_subset(
        const std::array<uint, numberOfFeatures>& featuresCounts,
        const std::array<std::pair<uint, uint>, numberOfFeatures>& minMaxNumberOfFeatures,
        const matches_containers::matchContainer& matchedFeatures)
{
    matches_containers::match_sets matchSubset;
    const auto selection = get_random_selection(featuresCounts, minMaxNumberOfFeatures);
    const uint numberOfPoint2dToSample = selection[featureIndex2dPoint];
    const uint numberOfPointsToSample = selection[featureIndexPoint];
    const uint numberOfPlanesToSample = selection[featureIndexPlane];

    const double subsetScore = numberOfPoint2dToSample * scorePerFeature[featureIndex2dPoint] +
                               numberOfPointsToSample * scorePerFeature[featureIndexPoint] +
                               numberOfPlanesToSample * scorePerFeature[featureIndexPlane];

    if (subsetScore < 1.0)
        throw std::logic_error(std::format(
                "Selected 2d points {} and {} points and {} planes, not enough for optimization (score: {})",
                numberOfPoint2dToSample,
                numberOfPointsToSample,
                numberOfPlanesToSample,
                subsetScore));

    // we can have a lot of points, so use a more efficient but with potential duplicates subset
    // matchSubset._pointSets._inliers = ransac::get_random_subset_with_duplicates(matchedFeatures._points,
    // numberOfPointsToSample);
    matchSubset._point2DSets._inliers = ransac::get_random_subset(matchedFeatures._points2D, numberOfPoint2dToSample);
    matchSubset._pointSets._inliers = ransac::get_random_subset(matchedFeatures._points, numberOfPointsToSample);
    matchSubset._planeSets._inliers = ransac::get_random_subset(matchedFeatures._planes, numberOfPlanesToSample);

    if (matchSubset._point2DSets._inliers.size() != numberOfPoint2dToSample or
        matchSubset._pointSets._inliers.size() != numberOfPointsToSample or
        matchSubset._planeSets._inliers.size() != numberOfPlanesToSample)
    {
        throw std::logic_error("get_random_subset: the output subset should have the requested size");
    }

    return matchSubset;
}

bool Pose_Optimization::compute_pose_with_ransac(const utils::PoseBase& currentPose,
                                                 const matches_containers::matchContainer& matchedFeatures,
                                                 utils::PoseBase& finalPose,
                                                 matches_containers::match_sets& featureSets) noexcept
{
    const double computePoseRansacStartTime = static_cast<double>(cv::getTickCount());

    featureSets.clear();

    const double matched2dPointSize = static_cast<double>(matchedFeatures._points2D.size());
    const double matchedPointSize = static_cast<double>(matchedFeatures._points.size());
    const double matchedPlaneSize = static_cast<double>(matchedFeatures._planes.size());

    // individual feature score
    constexpr double point2dFeatureScore = scorePerFeature[featureIndex2dPoint];
    constexpr double pointFeatureScore = scorePerFeature[featureIndexPoint];
    constexpr double planeFeatureScore = scorePerFeature[featureIndexPlane];

    // check that we have enough features for minimal pose optimization
    const double initialFeatureScore = point2dFeatureScore * matched2dPointSize + pointFeatureScore * matchedPointSize +
                                       planeFeatureScore * matchedPlaneSize;
    if (initialFeatureScore < 1.0)
    {
        // if there is not enough potential inliers to optimize a pose
        outputs::log_warning(std::format("Not enough features to optimize a pose ({} 2d points, {} points, {} planes)",
                                         static_cast<int>(matched2dPointSize),
                                         static_cast<int>(matchedPointSize),
                                         static_cast<int>(matchedPlaneSize)));
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    constexpr double point2dMaxRetroprojectionError_px =
            parameters::optimization::ransac::maximumRetroprojectionErrorForPoint2DInliers_px; // maximum inlier
    constexpr double pointMaxRetroprojectionError_px =
            parameters::optimization::ransac::maximumRetroprojectionErrorForPointInliers_px; // maximum inlier threshold
    constexpr double planeMaxRetroprojectionError_mm =
            parameters::optimization::ransac::maximumRetroprojectionErrorForPlaneInliers_mm; // maximum inlier threshold
                                                                                             // threshold
    static_assert(point2dMaxRetroprojectionError_px > 0);
    static_assert(pointMaxRetroprojectionError_px > 0);
    static_assert(planeMaxRetroprojectionError_mm > 0);
    const uint acceptablePointInliersForEarlyStop = static_cast<uint>(
            matchedPointSize *
            parameters::optimization::ransac::minimumInliersProportionForEarlyStop); // RANSAC will stop early if
                                                                                     // this inlier count is
                                                                                     // reached
    const uint acceptable2dPointInliersForEarlyStop = static_cast<uint>(
            matched2dPointSize *
            parameters::optimization::ransac::minimumInliersProportionForEarlyStop); // RANSAC will stop early if
                                                                                     // this inlier count is
                                                                                     // reached
    const uint acceptablePlaneInliersForEarlyStop = static_cast<uint>(
            matchedPlaneSize *
            parameters::optimization::ransac::minimumInliersProportionForEarlyStop); // RANSAC will stop early if
                                                                                     // this inlier count is
                                                                                     // reached

    const std::array<uint, numberOfFeatures> featureCount = {
            (uint)matchedPlaneSize, (uint)matchedPointSize, (uint)matched2dPointSize};
    const std::array<std::pair<uint, uint>, numberOfFeatures> minMaxPerFeature =
            get_min_max_number_of_features(featureCount);

    // check that we have enough inlier features for a pose optimization with RANSAC
    // This score is to stop the RANSAC process early (limit to 1 if we are low on features)
    const double enoughInliersScore = std::max(1.0,
                                               point2dFeatureScore * acceptable2dPointInliersForEarlyStop +
                                                       pointFeatureScore * acceptablePointInliersForEarlyStop +
                                                       planeFeatureScore * acceptablePlaneInliersForEarlyStop);

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
    const double maxFittingScore = matched2dPointSize * point2dMaxRetroprojectionError_px +
                                   matchedPointSize * pointMaxRetroprojectionError_px +
                                   matchedPlaneSize * planeMaxRetroprojectionError_mm;
    if (maxFittingScore < enoughInliersScore)
    {
        // the minimum feature score. Below that, no optimization can be made
        outputs::log_error("max fitting score should be >= 1.0");
        _meanPoseRANSACDuration +=
                (static_cast<double>(cv::getTickCount()) - computePoseRansacStartTime) / cv::getTickFrequency();
        return false;
    }

    double minScore = maxFittingScore;
    utils::PoseBase bestPose = currentPose;
    uint iteration = 0;
    for (; iteration < maximumIterations; ++iteration)
    {
        const double getRandomSubsetStartTime = static_cast<double>(cv::getTickCount());
        // get a random subset for this iteration
        const matches_containers::match_sets& selectedMatches =
                get_random_subset(featureCount, minMaxPerFeature, matchedFeatures);
        _meanGetRandomSubsetDuration +=
                (static_cast<double>(cv::getTickCount()) - getRandomSubsetStartTime) / cv::getTickFrequency();

        // compute a new candidate pose to evaluate
        const double computeOptimisedPoseTime = static_cast<double>(cv::getTickCount());
        utils::PoseBase candidatePose;
        if (not Pose_Optimization::compute_optimized_global_pose(currentPose, selectedMatches, candidatePose))
            continue;
        _meanRANSACPoseOptimizationDuration +=
                (static_cast<double>(cv::getTickCount()) - computeOptimisedPoseTime) / cv::getTickFrequency();

        // get inliers and outliers for this transformation
        const double getRANSACInliersTime = static_cast<double>(cv::getTickCount());
        matches_containers::match_sets potentialInliersOutliers;
        const double transformationScore = get_features_inliers_outliers(matchedFeatures,
                                                                         point2dMaxRetroprojectionError_px,
                                                                         pointMaxRetroprojectionError_px,
                                                                         planeMaxRetroprojectionError_mm,
                                                                         candidatePose,
                                                                         potentialInliersOutliers);
        _meanRANSACGetInliersDuration +=
                (static_cast<double>(cv::getTickCount()) - getRANSACInliersTime) / cv::getTickFrequency();

        // safety
        if (transformationScore > maxFittingScore)
        {
            outputs::log_error("The computed score is higher than the max fitting score");
            continue;
        }

        // We have a better score than the previous best one
        if (transformationScore < minScore)
        {
            minScore = transformationScore;
            bestPose = candidatePose;
            // save features inliers and outliers
            featureSets.swap(potentialInliersOutliers);

            const double inlierScore =
                    static_cast<double>(featureSets._point2DSets._inliers.size()) * pointFeatureScore +
                    static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore +
                    static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
            if (inlierScore >= enoughInliersScore)
            {
                // We can stop here, the optimization is good enough
                break;
            }
        }
    }

    // We do not have enough inliers to consider this optimization as valid
    const double inlierScore = static_cast<double>(featureSets._point2DSets._inliers.size()) * pointFeatureScore +
                               static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore +
                               static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
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
    const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(bestPose, featureSets, finalPose);
    if (isPoseValid)
    {
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
            outputs::log_warning("Could not compute pose variance after succesful optimization");
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

    // Optimization function (ok to use pointers: optimization of copy)
    Global_Pose_Functor pose_optimisation_functor(Global_Pose_Estimator(&matchedFeatures._point2DSets._inliers,
                                                                        &matchedFeatures._pointSets._inliers,
                                                                        &matchedFeatures._planeSets._inliers));
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
        outputs::log(std::format("Failed to converge with {} 2d points, {} points, {} planes | Status {}",
                                 matchedFeatures._point2DSets._inliers.size(),
                                 matchedFeatures._pointSets._inliers.size(),
                                 matchedFeatures._planeSets._inliers.size(),
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
                                                         const matches_containers::match_sets& matchedFeatures,
                                                         utils::PoseBase& optimizedPose) noexcept
{
    matches_containers::match_sets variatedSet;
    for (const matches_containers::PointMatch2D& match: matchedFeatures._point2DSets._inliers)
    {
        utils::WorldCoordinate variatedObservationPoint = match._worldFeature.get_first_observation();
        // TODO: variate the observation point
        // variatedObservationPoint +=
        // utils::Random::get_normal_doubles<3>().cwiseProduct(match._worldFeatureCovariance.diagonal().head<3>());
        const double variatedInverseDepth =
                match._worldFeature
                        .get_inverse_depth(); // do not variate the depth, the uncertainty is too great anyway
        const double variatedTheta =
                std::clamp(match._worldFeature.get_theta() + utils::Random::get_normal_double() *
                                                                     match._worldFeatureCovariance.diagonal()(
                                                                             utils::InverseDepthWorldPoint::thetaIndex),
                           0.0,
                           M_PI);
        const double variatedPhi =
                std::clamp(match._worldFeature.get_phi() + utils::Random::get_normal_double() *
                                                                   match._worldFeatureCovariance.diagonal()(
                                                                           utils::InverseDepthWorldPoint::phiIndex),
                           -M_PI,
                           M_PI);
        utils::InverseDepthWorldPoint variatedCoordinates(
                variatedObservationPoint, variatedInverseDepth, variatedTheta, variatedPhi);

        variatedSet._point2DSets._inliers.emplace_back(
                match._screenFeature, variatedCoordinates, match._worldFeatureCovariance, match._idInMap);
    }
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

    assert(variatedSet._point2DSets._inliers.size() == matchedFeatures._point2DSets._inliers.size());
    assert(variatedSet._pointSets._inliers.size() == matchedFeatures._pointSets._inliers.size());
    assert(variatedSet._planeSets._inliers.size() == matchedFeatures._planeSets._inliers.size());
    return compute_optimized_global_pose(currentPose, variatedSet, optimizedPose);
}

} // namespace rgbd_slam::pose_optimization
