#include "pose_optimization.hpp"

#include "levenberg_marquard_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"

#include "../parameters.hpp"

#include "../utils/camera_transformation.hpp"
#include "../utils/covariances.hpp"
#include "../outputs/logger.hpp"
#include "../utils/random.hpp"

#include "../../third_party/p3p.hpp"

#include <Eigen/StdVector>
#include <cmath>
#include <string>

namespace rgbd_slam {
    namespace pose_optimization {
        
        /**
         * \brief Compute a score for a transformation, and compute an inlier and outlier set
         * \param[in] pointsToEvaluate The set of points to evaluate the transformation on
         * \param[in] pointMaxRetroprojectionError The maximum retroprojection error between two point, below which we classifying the match as inlier 
         * \param[in] transformationPose The transformation that needs to be evaluated
         * \param[out] pointMatcheSets The set of inliers/outliers of this transformation
         * \return The transformation score (sum of retroprojection distances)
         */
        double get_point_inliers_outliers(const matches_containers::match_point_container& pointsToEvaluate, const double pointMaxRetroprojectionError, const utils::PoseBase& transformationPose, matches_containers::point_match_sets& pointMatcheSets)
        {
            pointMatcheSets.clear();

            // get a world to camera transform to evaluate the retroprojection score
            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(transformationPose.get_orientation_quaternion(), transformationPose.get_position());

            double retroprojectionScore = 0.0;
            for (const matches_containers::PointMatch& match : pointsToEvaluate)
            {
                // Retroproject world point to screen, and compute screen distance
                const double distance = match._worldFeature.get_distance(match._screenFeature, worldToCamera);
                assert(distance >= 0 and not std::isnan(distance));
                // inlier
                if (distance < pointMaxRetroprojectionError)
                {
                    pointMatcheSets._inliers.insert(pointMatcheSets._inliers.end(), match);
                }
                // outlier
                else
                {
                    pointMatcheSets._outliers.insert(pointMatcheSets._outliers.end(), match);
                }
                retroprojectionScore += std::min(pointMaxRetroprojectionError, distance);
            }
            return retroprojectionScore;
        }

        /**
         * \brief Compute a score for a transformation, and compute an inlier and outlier set
         * \param[in] planesToEvaluate The set of planes to evaluate the transformation on
         * \param[in] planeMaxRetroprojectionError The maximum retroprojection error between two planes, below which we classifying the match as inlier 
         * \param[in] transformationPose The transformation that needs to be evaluated
         * \param[out] planeMatchSets The set of inliers/outliers of this transformation
         * \return The transformation score
         */
        double get_plane_inliers_outliers(const matches_containers::match_plane_container& planesToEvaluate, const double planeMaxRetroprojectionError, const utils::PoseBase& transformationPose, matches_containers::plane_match_sets& planeMatchSets)
        {
            planeMatchSets.clear();

            // get a world to camera transform to evaluate the retroprojection score
            const planeWorldToCameraMatrix& worldToCamera = utils::compute_plane_world_to_camera_matrix(utils::compute_world_to_camera_transform(transformationPose.get_orientation_quaternion(), transformationPose.get_position()));

            double retroprojectionScore = 0.0;
            for (const matches_containers::PlaneMatch& match : planesToEvaluate)
            {
                // Retroproject world point to screen, and compute screen distance
                const double distance = match._worldFeature.get_reduced_signed_distance(match._screenFeature, worldToCamera).norm() / 10.0;
                assert(distance >= 0 and not std::isnan(distance));
                // inlier
                if (distance < planeMaxRetroprojectionError)
                {
                    planeMatchSets._inliers.insert(planeMatchSets._inliers.end(), match);
                }
                // outlier
                else
                {
                    planeMatchSets._outliers.insert(planeMatchSets._outliers.end(), match);
                }
                retroprojectionScore += std::min(planeMaxRetroprojectionError, distance);
            }
            return retroprojectionScore;
        }

        double get_features_inliers_outliers(const matches_containers::matchContainer& featuresToEvaluate, const double pointMaxRetroprojectionError, const double planeMaxRetroprojectionError, const utils::PoseBase& transformationPose, matches_containers::match_sets& featureSet)
        {
            return 
                get_point_inliers_outliers(featuresToEvaluate._points, pointMaxRetroprojectionError, transformationPose, featureSet._pointSets) +
                get_plane_inliers_outliers(featuresToEvaluate._planes, planeMaxRetroprojectionError, transformationPose, featureSet._planeSets);
        }

        /**
         * \brief Return a subset of a given inlier set
         */
        matches_containers::match_sets get_random_subset(const uint numberOfPointsToSample, const uint numberOfPlanesToSample, const matches_containers::matchContainer& matchedFeatures)
        {
            matches_containers::match_sets matchSubset;
            matchSubset._pointSets._inliers = ransac::get_random_subset(matchedFeatures._points, numberOfPointsToSample);
            matchSubset._planeSets._inliers = ransac::get_random_subset(matchedFeatures._planes, numberOfPlanesToSample);
            assert(matchSubset._pointSets._inliers.size() == numberOfPointsToSample);
            assert(matchSubset._planeSets._inliers.size() == numberOfPlanesToSample);

            return matchSubset;
        }


        bool Pose_Optimization::compute_pose_with_ransac(const utils::PoseBase& currentPose, const matches_containers::matchContainer& matchedFeatures, utils::PoseBase& finalPose, matches_containers::match_sets& featureSets) 
        {
            featureSets.clear();

            const double matchedPointSize = static_cast<double>(matchedFeatures._points.size());
            const double matchedPlaneSize = static_cast<double>(matchedFeatures._planes.size());

            const static uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();    // Number of random points to select
            const static uint minimumPlanesForOptimization = Parameters::get_minimum_plane_count_for_optimization();    // Number of random planes to select
            assert(minimumPointsForOptimization > 0);
            assert(minimumPlanesForOptimization > 0);

            // individual feature score
            const static double pointFeatureScore = 1.0 / minimumPointsForOptimization;
            const static double planeFeatureScore = 1.0 / minimumPlanesForOptimization;

            // check that we have enough features for minimal pose optimization
            const double initialFeatureScore = pointFeatureScore * matchedPointSize + planeFeatureScore * matchedPlaneSize;
            if (initialFeatureScore < 1.0)
            {
                // if there is not enough potential inliers to optimize a pose
                outputs::log_warning("Not enough features to optimize a pose (" + std::to_string(matchedPointSize) + " points, " +  std::to_string(matchedPlaneSize) + " planes)");
                return false;
            }

            const static double pointMaxRetroprojectionError = Parameters::get_ransac_maximum_retroprojection_error_for_point_inliers(); // maximum inlier threshold, in pixels 
            const static double planeMaxRetroprojectionError = Parameters::get_ransac_maximum_retroprojection_error_for_plane_inliers();
            assert(pointMaxRetroprojectionError > 0);
            assert(planeMaxRetroprojectionError > 0);
            const uint acceptablePointInliersForEarlyStop = static_cast<uint>(matchedPointSize * Parameters::get_ransac_minimum_inliers_proportion_for_early_stop()); // RANSAC will stop early if this inlier count is reached
            const uint acceptablePlaneInliersForEarlyStop = static_cast<uint>(matchedPlaneSize * Parameters::get_ransac_minimum_inliers_proportion_for_early_stop()); // RANSAC will stop early if this inlier count is reached

            // check that we have enough inlier features for a pose optimization with RANSAC
            // This score is to stop the RANSAC process early (limit to 1 if we are low on features)
            const double enoughInliersScore = std::max(1.0, pointFeatureScore * acceptablePointInliersForEarlyStop + planeFeatureScore * acceptablePlaneInliersForEarlyStop);

            // get the min and max values of planes and points to select
            const uint maxNumberOfPoints = std::min(minimumPointsForOptimization, (uint)matchedPointSize);
            const uint maxNumberOfPlanes = std::min(minimumPlanesForOptimization, (uint)matchedPlaneSize);
            const uint minNumberOfPlanes = std::ceil((1.0 - maxNumberOfPoints * pointFeatureScore) / planeFeatureScore);
            const uint minNumberOfPoints = std::ceil((1.0 - maxNumberOfPlanes * planeFeatureScore) / pointFeatureScore);

            // Compute maximum iteration with the original RANSAC formula
            const uint maximumIterations = std::ceil(log(1.0 - Parameters::get_ransac_probability_of_success()) / log(1.0 - pow(Parameters::get_ransac_inlier_proportion(), minimumPointsForOptimization)));
            assert(maximumIterations > 0);

            // set the start score to the maximum score
            double minScore = matchedPointSize * pointMaxRetroprojectionError + matchedPlaneSize * planeMaxRetroprojectionError;
            utils::PoseBase bestPose = currentPose;
            for(uint iteration = 0; iteration < maximumIterations; ++iteration)
            {
                // get random number of planes, between minNumberOfPlanes and maxNumberOfPlanes
                const uint numberOfPlanesToSample = minNumberOfPlanes + (maxNumberOfPlanes - minNumberOfPlanes) * (utils::Random::get_random_double() > 0.5);
                // depending on this number of planes, get a number of points to sample for this RANSAC iteration
                const uint numberOfPointsToSample = std::ceil((1 - numberOfPlanesToSample * planeFeatureScore) / pointFeatureScore);

                const double subsetScore = numberOfPointsToSample * pointFeatureScore + numberOfPlanesToSample * planeFeatureScore;
                if (subsetScore < 1.0)
                {
                    outputs::log_warning("Selected " + std::to_string(numberOfPointsToSample) + " points and " + std::to_string(numberOfPlanesToSample) + " planes, not enough for optimization (score: " + std::to_string(subsetScore) + ")");
                    continue;
                }
                if (numberOfPlanesToSample < minNumberOfPlanes or numberOfPlanesToSample > maxNumberOfPlanes or numberOfPlanesToSample > matchedPlaneSize)
                {
                    outputs::log_warning("Selected " + std::to_string(numberOfPlanesToSample) + " planes but we have " + std::to_string(matchedPointSize) + " available");
                    continue;
                }
                if (numberOfPointsToSample < minNumberOfPoints or numberOfPointsToSample > maxNumberOfPoints or numberOfPointsToSample > matchedPointSize)
                {
                    outputs::log_warning("Selected " + std::to_string(numberOfPointsToSample) + " points but we have " + std::to_string(matchedPlaneSize) + " available");
                    continue;
                }

                const matches_containers::match_sets& selectedMatches = get_random_subset(numberOfPointsToSample, numberOfPlanesToSample, matchedFeatures);

                // compute a new candidate pose to evaluate
                utils::PoseBase candidatePose;
                const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(currentPose, selectedMatches, candidatePose);
                //const bool isPoseValid = Pose_Optimization::compute_p3p_pose(currentPose, selectedPointMatches, candidatePose);
                if (not isPoseValid)
                    continue;

                // get inliers and outliers for this transformation
                matches_containers::match_sets potentialInliersOutliers;
                const double transformationScore = get_features_inliers_outliers(matchedFeatures, pointMaxRetroprojectionError, planeMaxRetroprojectionError, candidatePose, potentialInliersOutliers);
                // We have a better score than the previous best one
                if (transformationScore < minScore)
                {
                    minScore = transformationScore;
                    bestPose = candidatePose;
                    // save features inliers and outliers
                    featureSets.swap(potentialInliersOutliers);

                    const double inlierScore = static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore + static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
                    if (inlierScore >= enoughInliersScore)
                    {
                        // We can stop here, the optimization is good enough
                        break;
                    }
                }
            }

            // We do not have enough inliers to consider this optimization as valid
            const double inlierScore = static_cast<double>(featureSets._pointSets._inliers.size()) * pointFeatureScore + static_cast<double>(featureSets._planeSets._inliers.size()) * planeFeatureScore;
            if (inlierScore < 1.0)
            {
                outputs::log_warning("Could not find a transformation with enough inliers using RANSAC");
                return false;
            }

            // optimize on all inliers
            const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(bestPose, featureSets, finalPose);
            if (isPoseValid)
            {
                return true;
            }

            outputs::log_warning("Could not compute a global pose, even though we found a valid inlier set");
            return false;
        }

        bool Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::matchContainer& matchedFeatures, utils::Pose& optimizedPose, matches_containers::match_sets& featureSets) 
        {
            const bool isPoseValid = compute_pose_with_ransac(currentPose, matchedFeatures, optimizedPose, featureSets);
            if (isPoseValid)
            {
                // Compute pose variance
                vector3 estimatedPoseVariance;
                // TODO: compute variance with planes too
                if (not featureSets._pointSets._inliers.empty() and utils::compute_pose_variance(optimizedPose, featureSets._pointSets._inliers, estimatedPoseVariance))
                {
                    optimizedPose.set_position_variance( estimatedPoseVariance + currentPose.get_position_variance());
                }
                else
                {
                    outputs::log_warning("Could not compute pose variance, as we only work with 2D points");
                    optimizedPose.set_position_variance(currentPose.get_position_variance());
                }
            }
            return isPoseValid;
        }


        bool Pose_Optimization::compute_optimized_global_pose(const utils::PoseBase& currentPose, const matches_containers::match_sets& matchedFeatures, utils::PoseBase& optimizedPose) 
        {
            const vector3& position = currentPose.get_position();    // Work in millimeters
            const quaternion& rotation = currentPose.get_orientation_quaternion();

            // Vector to optimize: (0, 1, 2) is position,
            // Vector (3, 4, 5) is a rotation parametrization, representing a delta in rotation in the tangential hyperplane -From Using Quaternions for Parametrizing 3-D Rotation in Unconstrained Nonlinear Optimization)
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
                    Global_Pose_Estimator(
                        input.size(), 
                        matchedFeatures._pointSets._inliers,
                        matchedFeatures._planeSets._inliers
                        )
                    );
            // Optimization algorithm
            Eigen::LevenbergMarquardt<Global_Pose_Functor, double> poseOptimizator( pose_optimisation_functor );

            // maxfev   : maximum number of function evaluation
            poseOptimizator.parameters.maxfev = Parameters::get_optimization_maximum_iterations();
            // epsfcn   : error precision
            poseOptimizator.parameters.epsfcn = Parameters::get_optimization_error_precision();
            // xtol     : tolerance for the norm of the solution vector
            poseOptimizator.parameters.xtol = Parameters::get_optimization_xtol();
            // ftol     : tolerance for the norm of the vector function
            poseOptimizator.parameters.ftol = Parameters::get_optimization_ftol();
            // gtol     : tolerance for the norm of the gradient of the error function
            poseOptimizator.parameters.gtol = Parameters::get_optimization_gtol();
            // factor   : step bound for the diagonal shift
            poseOptimizator.parameters.factor = Parameters::get_optimization_factor();

            // Start optimization
            const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);

            // Get result
            const quaternion& endRotation = get_quaternion_from_scale_axis_coefficients(
                    vector3(
                        input[3],
                        input[4],
                        input[5]
                        )
                    ); 
            const vector3 endPosition(
                    input[0],
                    input[1],
                    input[2]
                    );

            if (endStatus <= 0) 
            {
                // Error while optimizing 
                const std::string message = get_human_readable_end_message(endStatus);
                outputs::log("Failed to converge with " + std::to_string(matchedFeatures._pointSets._inliers.size()) + " points | Status " + message);
                return false;
            }

            // Update refined pose with optimized pose
            optimizedPose.set_parameters(endPosition, endRotation);
            return true;
        }

        bool Pose_Optimization::compute_p3p_pose(const utils::PoseBase& currentPose, const matches_containers::match_point_container& matchedPoints, utils::PoseBase& optimizedPose)
        {
            assert(matchedPoints.size() == 3);
            // Do all operations on meters, while this SLAM uses millimeters
            const double multiplier = 1000.0;

            std::vector<vector3> cameraPoints;
            std::vector<vector3> worldPoints;

            for (const matches_containers::PointMatch& match : matchedPoints)
            {
                const vector3& cameraPoint = match._screenFeature.to_camera_coordinates().base();

                cameraPoints.push_back(cameraPoint.normalized());
                worldPoints.push_back(match._worldFeature / multiplier);
            }

            const std::vector<lambdatwist::CameraPose>& finalCameraPoses = lambdatwist::p3p(cameraPoints, worldPoints);
            assert(finalCameraPoses.size() <= 4);

            const vector3& posePosition = currentPose.get_position();
            const matrix33& poseRotation = currentPose.get_orientation_matrix();

            double closestPoseDistance = std::numeric_limits<double>::max();
            for(const lambdatwist::CameraPose& cameraPose : finalCameraPoses)
            {
                const double rotationDistance = (cameraPose.R - poseRotation).norm();
                const double positionDistance = (cameraPose.t * multiplier - posePosition).norm();

                assert(rotationDistance >= 0 and not std::isnan(rotationDistance));
                assert(positionDistance >= 0 and not std::isnan(positionDistance));
                const double distance = rotationDistance + positionDistance;

                if (distance < closestPoseDistance)
                {
                    closestPoseDistance = distance;
                    optimizedPose.set_parameters(cameraPose.t * multiplier, quaternion(cameraPose.R));
                }
            }
            // At least one valid pose found
            return closestPoseDistance < std::numeric_limits<double>::max();
        }

    }   /* pose_optimization*/
}   /* rgbd_slam */
