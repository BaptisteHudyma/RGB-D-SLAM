#include "pose_optimization.hpp"

#include "levenberg_marquard_functors.hpp"
#include "matches_containers.hpp"
#include "ransac.hpp"

#include "../parameters.hpp"

#include "../utils/camera_transformation.hpp"
#include "../utils/distance_utils.hpp"
#include "../utils/covariances.hpp"
#include "../outputs/logger.hpp"

#include "../../third_party/p3p.hpp"

#include <Eigen/StdVector>
#include <random>

namespace rgbd_slam {
    namespace pose_optimization {
        
        /**
         * \brief Compute a score for a transformation, and compute an inlier and outlier set
         * \param[in] pointsToEvaluate The set of points to evaluate the transformation on
         * \param[in] maximumRetroprojectionThreshold The maximum retroprojection error between two point, below which we classifying the match as inlier 
         * \param[in] transformationPose The transformation that needs to be evaluated
         * \param[out] inliersContainer The set inliers of inliers of this transformation
         * \param[out] outliersContainer The set inliers of outliers of this transformation
         * \return The transformation score (sum of retroprojection distances)
         */
        double get_point_inliers_outliers(const matches_containers::match_point_container& pointsToEvaluate, const double maximumRetroprojectionThreshold, const utils::Pose& transformationPose, matches_containers::match_point_container& inliersContainer, matches_containers::match_point_container& outliersContainer)
        {
            inliersContainer.clear();
            outliersContainer.clear();

            // get a world to camera transform to evaluate the retroprojection score
            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(transformationPose.get_orientation_quaternion(), transformationPose.get_position());

            double retroprojectionScore = 0.0;
            for (const matches_containers::PointMatch& match : pointsToEvaluate)
            {
                // Retroproject world point to screen, and compute screen distance
                const double distance = utils::get_3D_to_2D_distance(match._worldPoint, match._screenPoint, worldToCamera);
                assert(distance >= 0 and not std::isnan(distance));
                // inlier
                if (distance < maximumRetroprojectionThreshold)
                {
                    inliersContainer.insert(inliersContainer.end(), match);
                }
                // outlier
                else
                {
                    outliersContainer.insert(outliersContainer.end(), match);
                }
                retroprojectionScore += std::min(maximumRetroprojectionThreshold, distance);
            }
            return retroprojectionScore;
        }

        /**
         * \brief Compute a score for a transformation, and compute an inlier and outlier set
         * \param[in] pointsToEvaluate The set of planes to evaluate the transformation on
         * \param[in] maximumRetroprojectionThreshold The maximum retroprojection error between two planes, below which we classifying the match as inlier 
         * \param[in] transformationPose The transformation that needs to be evaluated
         * \param[out] inliersContainer The set inliers of inliers of this transformation
         * \param[out] outliersContainer The set inliers of outliers of this transformation
         * \return The transformation score
         */
        double get_plane_inliers_outliers(const matches_containers::match_plane_container& planesToEvaluate, const double maximumRetroprojectionThreshold, const utils::Pose& transformationPose, matches_containers::match_plane_container& inliersContainer, matches_containers::match_plane_container& outliersContainer)
        {
            inliersContainer.clear();
            outliersContainer.clear();

            // get a world to camera transform to evaluate the retroprojection score
            const planeWorldToCameraMatrix& worldToCamera = utils::compute_plane_world_to_camera_matrix(utils::compute_world_to_camera_transform(transformationPose.get_orientation_quaternion(), transformationPose.get_position()));

            double retroprojectionScore = 0.0;
            for (const matches_containers::PlaneMatch& match : planesToEvaluate)
            {
                // Retroproject world point to screen, and compute screen distance
                const double distance = utils::get_3D_to_2D_plane_distance(match._worldPlane, match._cameraPlane, worldToCamera).norm();
                assert(distance >= 0 and not std::isnan(distance));
                // inlier
                if (distance < maximumRetroprojectionThreshold)
                {
                    inliersContainer.insert(inliersContainer.end(), match);
                }
                // outlier
                else
                {
                    outliersContainer.insert(outliersContainer.end(), match);
                }
                retroprojectionScore += std::min(maximumRetroprojectionThreshold, distance);
            }
            return retroprojectionScore;
        }

        bool Pose_Optimization::compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, const matches_containers::match_plane_container& matchedPlanes, utils::Pose& finalPose, matches_containers::match_point_container& outlierMatchedPoints) 
        {
            outlierMatchedPoints.clear();

            const double matchedPointSize = static_cast<double>(matchedPoints.size());
            const double matchedPlaneSize = static_cast<double>(matchedPlanes.size());

            const uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();    // Number of random points to select
            const uint minimumPlanesForOptimization = Parameters::get_minimum_plane_count_for_optimization();    // Number of random planes to select
            assert(minimumPointsForOptimization > 0);
            assert(minimumPlanesForOptimization > 0);

            // individual feature score
            const double pointFeatureScore = 1.0 / minimumPointsForOptimization;
            const double planeFeatureScore = 1.0 / minimumPlanesForOptimization;

            // check that we have enough features for minimal pose optimization
            const double initialFeatureScore = pointFeatureScore * matchedPointSize + planeFeatureScore * matchedPlaneSize;
            if (initialFeatureScore < 1.0)
            {
                // if there is not enough potential inliers to optimize a pose
                outputs::log_warning("Not enough features to optimize a pose (" + std::to_string(matchedPoints.size()) + " points, " +  std::to_string(matchedPlanes.size()) + " planes)");
                return false;
            }

            const double maximumRetroprojectionThreshold = Parameters::get_ransac_maximum_retroprojection_error_for_inliers(); // maximum inlier threshold, in pixels 
            assert(maximumRetroprojectionThreshold > 0);
            const uint acceptablePointInliersForEarlyStop = static_cast<uint>(matchedPointSize * Parameters::get_ransac_minimum_inliers_proportion_for_early_stop()); // RANSAC will stop early if this inlier count is reached
            const uint acceptablePlaneInliersForEarlyStop = static_cast<uint>(matchedPlaneSize * Parameters::get_ransac_minimum_inliers_proportion_for_early_stop()); // RANSAC will stop early if this inlier count is reached

            // check that we have enough inlier features for a pose optimization with RANSAC
            // This score is to stop the RANSAC process early
            const double enoughInliersScore = pointFeatureScore * acceptablePointInliersForEarlyStop + planeFeatureScore * acceptablePlaneInliersForEarlyStop;
            if (enoughInliersScore < 1.0)
            {
                // if there is not enough potential inliers to optimize a pose
                outputs::log_warning("Not enough minimum inlier features to safely optimize a pose with RANSAC (" + std::to_string(acceptablePointInliersForEarlyStop) + " points, " +  std::to_string(acceptablePlaneInliersForEarlyStop) + " planes)");
                return false;
            }

            // get the min and max values of planes and points to select
            const uint maxNumberOfPoints = std::min(minimumPointsForOptimization, (uint)matchedPoints.size());
            const uint maxNumberOfPlanes = std::min(minimumPlanesForOptimization, (uint)matchedPlanes.size());
            const uint minNumberOfPlanes = static_cast<uint>((1.0 - maxNumberOfPoints * pointFeatureScore) / planeFeatureScore);
            //const uint minNumberOfPoints = static_cast<uint>((1.0 - maxNumberOfPlanes * planeFeatureScore) / pointFeatureScore);

            std::random_device randomDevice;
            std::mt19937 randomEngine(randomDevice());
            std::uniform_real_distribution<float> planeInRansacDistribution(0, 1.0);

            // Compute maximum iteration with the original RANSAC formula
            const uint maximumIterations = static_cast<uint>(log(1.0 - Parameters::get_ransac_probability_of_success()) / log(1.0 - pow(Parameters::get_ransac_inlier_proportion(), minimumPointsForOptimization)));
            assert(maximumIterations > 0);

            // set the start score to the maximum score
            double minScore = matchedPointSize * maximumRetroprojectionThreshold + matchedPlaneSize * maximumRetroprojectionThreshold;
            utils::Pose bestPose = currentPose;
            matches_containers::match_point_container inlierMatchedPoints;  // Contains the best pose inliers
            matches_containers::match_plane_container inlierMatchedPlanes;  // Contains the best pose inliers
            for(uint iteration = 0; iteration < maximumIterations; ++iteration)
            {
                const uint numberOfPlanesToSample = minNumberOfPlanes + (maxNumberOfPlanes - minNumberOfPlanes) * (planeInRansacDistribution(randomEngine) > 0.5);
                const uint numberOfPointsToSample = static_cast<uint>((1 - numberOfPlanesToSample * planeFeatureScore) / pointFeatureScore);

                const matches_containers::match_point_container& selectedPointMatches = ransac::get_random_subset(matchedPoints, numberOfPointsToSample);
                const matches_containers::match_plane_container& selectedPlaneMatches = ransac::get_random_subset(matchedPlanes, numberOfPlanesToSample);
                assert(selectedPointMatches.size() == numberOfPointsToSample);
                assert(selectedPlaneMatches.size() == numberOfPlanesToSample);

                // compute a new candidate pose to evaluate
                utils::Pose candidatePose;
                const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(currentPose, selectedPointMatches, selectedPlaneMatches, candidatePose);
                //const bool isPoseValid = Pose_Optimization::compute_p3p_pose(currentPose, selectedPointMatches, candidatePose);
                if (not isPoseValid)
                    continue;

                // get inliers and outliers for this transformation
                matches_containers::match_point_container potentialPointInliersContainer, potentialPointOutliersContainer;
                matches_containers::match_plane_container potentialPlaneInliersContainer, potentialPlaneOutliersContainer;
                const double pointTransformationScore = get_point_inliers_outliers(matchedPoints, maximumRetroprojectionThreshold, candidatePose, potentialPointInliersContainer, potentialPointOutliersContainer);
                const double planeTransformationScore = get_plane_inliers_outliers(matchedPlanes, maximumRetroprojectionThreshold, candidatePose, potentialPlaneInliersContainer, potentialPlaneOutliersContainer);
                const double transformationScore = pointTransformationScore + planeTransformationScore;
                // We have a better score than the previous best one
                if (transformationScore < minScore)
                {
                    minScore = transformationScore;
                    bestPose = candidatePose;
                    // save points inliers and outliers
                    inlierMatchedPoints.swap(potentialPointInliersContainer);
                    outlierMatchedPoints.swap(potentialPointOutliersContainer);
                    // save planes inliers and outliers
                    inlierMatchedPlanes.swap(potentialPlaneInliersContainer);

                    const double inlierScore = inlierMatchedPoints.size() * pointFeatureScore + inlierMatchedPlanes.size() * planeFeatureScore;
                    if (inlierScore >= enoughInliersScore)
                    {
                        // We can stop here, the optimization is good enough
                        break;
                    }
                }
            }

            // We do not have enough inliers to consider this optimization as valid
            const double inlierScore = inlierMatchedPoints.size() * pointFeatureScore + inlierMatchedPlanes.size() * planeFeatureScore;
            if (inlierScore < 1.0)
            {
                outputs::log_warning("Could not find a transformation with enough inliers using RANSAC");
                return false;
            }

            // optimize on all inliers
            const bool isPoseValid = Pose_Optimization::compute_optimized_global_pose(bestPose, inlierMatchedPoints, inlierMatchedPlanes, finalPose);
            if (isPoseValid)
            {
                // Compute pose variance
                vector3 estimatedPoseVariance;
                if (not inlierMatchedPoints.empty() and utils::compute_pose_variance(finalPose, inlierMatchedPoints, estimatedPoseVariance))
                {
                    finalPose.set_position_variance( estimatedPoseVariance + currentPose.get_position_variance());
                }
                else
                {
                    outputs::log_warning("Could not compute pose variance, as we only work with 2D points");
                }
                return true;
            }

            outputs::log_warning("Could not compute a global pose, even though we found a valid inlier set");
            return false;
        }

        bool Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, const matches_containers::match_plane_container& matchedPlanes, utils::Pose& optimizedPose, matches_containers::match_point_container& outlierMatchedPoints) 
        {
            const bool isPoseValid = compute_pose_with_ransac(currentPose, matchedPoints, matchedPlanes, optimizedPose, outlierMatchedPoints);
            return isPoseValid;
        }


        bool Pose_Optimization::compute_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, const matches_containers::match_plane_container& matchedPlanes, utils::Pose& optimizedPose) 
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
                        matchedPoints,
                        matchedPlanes
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
                outputs::log("Failed to converge with " + std::to_string(matchedPoints.size()) + " points | Status " + message);
                return false;
            }

            // Update refined pose with optimized pose
            optimizedPose.set_parameters(endPosition, endRotation);
            return true;
        }

        bool Pose_Optimization::compute_p3p_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose)
        {
            assert(matchedPoints.size() == 3);
            // Do all operations on meters, while this SLAM uses millimeters
            const double multiplier = 1000.0;

            std::vector<vector3> cameraPoints;
            std::vector<vector3> worldPoints;

            for (const matches_containers::PointMatch& match : matchedPoints)
            {
                const vector3& cameraPoint = match._screenPoint.to_camera_coordinates().base();

                cameraPoints.push_back(cameraPoint.normalized());
                worldPoints.push_back(match._worldPoint / multiplier);
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
