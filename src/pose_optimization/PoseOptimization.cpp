#include "PoseOptimization.hpp"

#include "utils.hpp"
#include "LevenbergMarquardFunctors.hpp"
#include "parameters.hpp"

#include <Eigen/StdVector>

namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Compute the retroprojection distance between a mapPoint and  a cameraPoint
         */
        double get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& camToWorldMatrix)
        {
            const vector3& worldPoint = utils::screen_to_world_coordinates(matchedPoint.x(), matchedPoint.y(), matchedPoint.z(), camToWorldMatrix);
            return (mapPoint - worldPoint).norm();
        }

        /**
         * \brief Return a random subset of matches, of size n
         */
        matches_containers::match_point_container get_n_random_matches(const matches_containers::match_point_container& matchedPoints, const uint n)
        {
            const size_t maxIndex = matchedPoints.size();
            assert(n < maxIndex);

            // get a random subset of indexes
            matches_containers::match_point_container selectedMatches;
            while(selectedMatches.size() < n)
            {
                const uint index = rand() % maxIndex;
                matches_containers::match_point_container::const_iterator it = matchedPoints.cbegin();
                std::advance(it, index);

                selectedMatches.insert(selectedMatches.begin(), *it);
            }
            return selectedMatches;
        }

        bool Pose_Optimization::compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints, utils::Pose& finalPose, matches_containers::match_point_container& inlierMatchedPoints) 
        {
            const uint minimumPointsForOptimization = 3;    // Selected set of random points
            const uint maxIterations = Parameters::get_maximum_ransac_iterations();
            double threshold = 10;                   // minimum retroprojection error to consider a match as an inlier (in millimeters)
            const uint matchedPointSize = matchedPoints.size();
            const double acceptableMinimumScore = matchedPointSize * 0.9;       // RANSAC will stop if this mean score is reached

            uint iteration = 0;
            for(; iteration < maxIterations; ++iteration)
            {
                const matches_containers::match_point_container& selectedMatches = get_n_random_matches(matchedPoints, minimumPointsForOptimization);
                assert(selectedMatches.size() == minimumPointsForOptimization);
                const utils::Pose& pose = Pose_Optimization::get_optimized_global_pose(currentPose, matchedPoints);

                const matrix34& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());

                // Select inliers by retroprojection threshold
                matches_containers::match_point_container potentialInliersContainer;
                for (const matches_containers::point_pair& match : matchedPoints)
                {
                    if (get_distance_to_point(match.second, match.first, transformationMatrix) < threshold)
                    {
                        potentialInliersContainer.insert(potentialInliersContainer.end(), match);
                    }
                }

                // We have a better score than the previous best one
                if (potentialInliersContainer.size() > inlierMatchedPoints.size())
                {
                    finalPose = pose;
                    inlierMatchedPoints.swap(potentialInliersContainer);

                    if (inlierMatchedPoints.size() >= acceptableMinimumScore)
                        // We can stop here, the optimization is pretty good
                        break;
                }
                else
                    // augment the error threshold
                    threshold += 10;
            }

            //std::cout << iteration << " " << matchedPointSize << " " <<  inlierMatchedPoints.size() << std::endl;
            if (inlierMatchedPoints.size() < minimumPointsForOptimization)
            {
                utils::log_error("Could not find a transformation with enough inliers");
                // error case
                return false;
            }

            finalPose = Pose_Optimization::get_optimized_global_pose(finalPose, inlierMatchedPoints);
            return true;
        }

        const utils::Pose Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints) 
        {
            utils::Pose newPose;
            matches_containers::match_point_container matchPointInliers;
            const bool isPoseValid = compute_pose_with_ransac(currentPose, matchedPoints, newPose, matchPointInliers);
            
            if (isPoseValid)
            {
                // compute pose covariance matrix

                return newPose;
            }

            // error in transformation optimisation, return the input pose
            return currentPose;
        }


        const utils::Pose Pose_Optimization::get_optimized_global_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints) 
        {
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

            if (endStatus == Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation)
            {
                // Error: reached end of minimization without reaching a minimum
                const std::string message = get_human_readable_end_message(endStatus);
                utils::log("Failed to converge with " + std::to_string(matchedPoints.size()) + " points | Status " + message);
            }

            // Update refined pose with optimized pose
            return utils::Pose(endPosition, endRotation);
        }

    }   /* pose_optimization*/
}   /* rgbd_slam */
