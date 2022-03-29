#include "pose_optimization.hpp"

#include "camera_transformation.hpp"
#include "distance_utils.hpp"

#include "logger.hpp"
#include "levenberg_marquard_functors.hpp"
#include "parameters.hpp"
#include "p3p.hpp"

#include "ransac.hpp"

#include <Eigen/StdVector>

namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Compute the variance of the final pose in X Y Z
         */
        vector3 compute_pose_variance(const utils::Pose& optimizedPose, const matches_containers::match_point_container& matchedPoints)
        {
            assert(not matchedPoints.empty());

            const matrix44& transformationMatrix = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            vector3 sumOfErrors;
            vector3 sumOfSquaredErrors;
            sumOfErrors.setZero();
            sumOfSquaredErrors.setZero();
            // For each pair of points
            for (const matches_containers::Match& match : matchedPoints)
            {
                // Convert to world coordinates
                const vector3& matchedPoint3d = utils::screen_to_world_coordinates(match._screenPoint.x(), match._screenPoint.y(), match._screenPoint.z(), transformationMatrix);

                // absolute of (world map Point - new world point)
                const vector3& matchError = (match._worldPoint - matchedPoint3d).cwiseAbs();
                sumOfErrors += matchError;
                sumOfSquaredErrors += matchError.cwiseAbs2();
            }

            assert(sumOfErrors.x() >= 0 and sumOfErrors.y() >= 0 and sumOfErrors.z() >= 0);
            assert(sumOfSquaredErrors.x() >= 0 and sumOfSquaredErrors.y() >= 0 and sumOfSquaredErrors.z() >= 0);

            const double numberOfMatchesInverse = 1.0 / static_cast<double>(matchedPoints.size());
            const vector3& mean = sumOfErrors * numberOfMatchesInverse; 
            return (sumOfSquaredErrors * numberOfMatchesInverse) - mean.cwiseAbs2();
        }

        bool Pose_Optimization::compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& finalPose, matches_containers::match_point_container& outlierMatchedPoints) 
        {
            const uint minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();    // Selected set of random points
            const uint maxIterations = Parameters::get_maximum_ransac_iterations();
            const double maxThreshold = Parameters::get_ransac_maximum_retroprojection_error_for_inliers(); // maximum inlier threshold (in millimeters)
            const size_t matchedPointSize = matchedPoints.size();
            const double acceptableMinimumScore = matchedPointSize * Parameters::get_ransac_minimum_inliers_for_validation();       // RANSAC will stop if this mean score is reached
            double threshold = Parameters::get_ransac_initial_threshold();  // minimum retroprojection error to consider a match as an inlier (in millimeters)

            utils::Pose bestPose = currentPose;
            matches_containers::match_point_container inlierMatchedPoints;  // Contains the best pose inliers
            for(uint iteration = 0; iteration < maxIterations; ++iteration)
            {
                const matches_containers::match_point_container& selectedMatches = get_random_subset(matchedPoints, minimumPointsForOptimization);
                assert(selectedMatches.size() == minimumPointsForOptimization);
                utils::Pose pose; 
                const bool isPoseValid = Pose_Optimization::get_optimized_global_pose(currentPose, selectedMatches, pose);
                //const bool isPoseValid = Pose_Optimization::compute_p3p_pose(currentPose, selectedMatches, pose);
                if (not isPoseValid)
                    continue;

                const matrix44& transformationMatrix = utils::compute_world_to_camera_transform(pose.get_orientation_quaternion(), pose.get_position());

                // Select inliers by retroprojection threshold
                matches_containers::match_point_container potentialInliersContainer;
                matches_containers::match_point_container potentialOutliersContainer;
                for (const matches_containers::Match& match : matchedPoints)
                {
                    if (utils::get_3D_to_2D_distance(match._worldPoint, match._screenPoint, transformationMatrix) < threshold)
                    {
                        potentialInliersContainer.insert(potentialInliersContainer.end(), match);
                    }
                    else
                    {
                        potentialOutliersContainer.insert(potentialOutliersContainer.end(), match);
                    }
                }

                // We have a better score than the previous best one
                if (potentialInliersContainer.size() > inlierMatchedPoints.size())
                {
                    bestPose = pose;
                    inlierMatchedPoints.swap(potentialInliersContainer);
                    outlierMatchedPoints.swap(potentialOutliersContainer);

                    if (inlierMatchedPoints.size() >= acceptableMinimumScore)
                    {
                        // We can stop here, the optimization is good enough
                        break;
                    }
                }
                else if (iteration % 5 and threshold < maxThreshold)
                    // augment the error threshold
                    threshold += 10;
            }

            if (inlierMatchedPoints.size() < minimumPointsForOptimization)
            {
                utils::log_error("Could not find a transformation with enough inliers");
                // error case
                return false;
            }

            const bool isPoseValid = Pose_Optimization::get_optimized_global_pose(bestPose, inlierMatchedPoints, finalPose);
            // Compute pose variance
            if (isPoseValid)
            {
                finalPose.set_position_variance(compute_pose_variance(finalPose, inlierMatchedPoints));
                return true;
            }

            return false;
        }

        bool Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose, matches_containers::match_point_container& outlierMatchedPoints) 
        {
            utils::Pose newPose;
            const bool isPoseValid = compute_pose_with_ransac(currentPose, matchedPoints, newPose, outlierMatchedPoints);

            if (isPoseValid)
            {
                // compute pose covariance matrix
                optimizedPose = newPose;
                return true;
            }

            // error in transformation optimisation
            return false;
        }


        bool Pose_Optimization::get_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose) 
        {
            assert(matchedPoints.size() >= 6);

            const vector3& position = currentPose.get_position();    // Work in millimeters
            const quaternion& rotation = currentPose.get_orientation_quaternion();

            // Vector to optimize: (0, 1, 2) is position,
            // Vector (3, 4, 5) is a rotation parametrization, representing a delta in rotation in the tangential hyperplane -From Using Quaternions for Parametrizing 3-D Rotation in Unconstrained Nonlinear Optimization)
            Eigen::VectorXd input(6);
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
                        currentPose.get_position(),
                        currentPose.get_orientation_quaternion()
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
                utils::log("Failed to converge with " + std::to_string(matchedPoints.size()) + " points | Status " + message);
                return false;
            }

            // Update refined pose with optimized pose
            optimizedPose.set_parameters(endPosition, endRotation);
            return true;
        }

        bool Pose_Optimization::compute_p3p_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& optimizedPose)
        {
            assert(matchedPoints.size() == 3);

            std::vector<vector3> cameraPoints;
            std::vector<vector3> worldPoints;

            for (const matches_containers::Match& match : matchedPoints)
            {
                const vector3 cameraPoint (
                        (match._screenPoint.x() - Parameters::get_camera_1_center_x()) / Parameters::get_camera_1_focal_x(),
                        (match._screenPoint.y() - Parameters::get_camera_1_center_y()) / Parameters::get_camera_1_focal_y(),
                        1
                        );

                cameraPoints.push_back(cameraPoint.normalized());
                worldPoints.push_back(match._worldPoint);
            }

            const std::vector<lambdatwist::CameraPose>& finalCameraPoses = lambdatwist::p3p(cameraPoints, worldPoints);
            assert(finalCameraPoses.size() <= 4);

            double closestPoseDistance = std::numeric_limits<double>::max();
            for(const lambdatwist::CameraPose& cameraPose : finalCameraPoses)
            {
                const double poseDistance = utils::get_distance_euclidean(currentPose.get_position(), cameraPose.t);
                assert(poseDistance >= 0 and not std::isnan(poseDistance));

                if (poseDistance < closestPoseDistance)
                {
                    closestPoseDistance = poseDistance;
                    optimizedPose.set_parameters(cameraPose.t, quaternion(cameraPose.R));
                }
            }
            // At least one valid pose found
            return closestPoseDistance < std::numeric_limits<double>::max();
        }

    }   /* pose_optimization*/
}   /* rgbd_slam */
