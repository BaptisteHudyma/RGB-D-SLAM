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

            matches_containers::match_point_container selectedMatches;
            std::set<size_t> usedIndexes;

            // get a random subset of indexes
            while(usedIndexes.size() < n)
            {
                const uint index = rand() % maxIndex;
                if (not usedIndexes.contains(index))
                {
                    usedIndexes.insert(index);
                }
            }

            // get the corresponding matches
            for(const size_t& index : usedIndexes)
            {
                matches_containers::match_point_container::const_iterator it = matchedPoints.cbegin();
                std::advance(it, index);

                selectedMatches.insert(selectedMatches.begin(), *it);
            }
            return selectedMatches;
        }

        /**
         * \brief Compute a score on a matchedPoint dataset, for a given pose
         */
        double get_pose_score(const utils::Pose& pose, const matches_containers::match_point_container& matchedPoints)
        {
            const matrix34& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());
            double score = 0;
            for (const matches_containers::point_pair& match : matchedPoints)
            {
                score += get_distance_to_point(match.second, match.first, transformationMatrix);
            }
            return score / matchedPoints.size();
        }

        /**
         * \brief Compute an optimized pose, using a RANSAC methodology
         */
        utils::Pose compute_pose_with_ransac(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints) {
            const uint minimumPointsForOptimization = 5;    // Selected set of random points
            const uint maxIterations = 50;                  // max RANSAC iterations
            const double threshold = 50;                     // minimum retroprojection error to consider a match as an inlier (in millimeters)
            const double acceptableMinimumScore = 10;       // RANSAC will stop if this mean score is reached
            const uint matchedPointSize = matchedPoints.size();
            const uint minimumPointsForEvaluation = 0.4 * matchedPointSize; // minimum number of points before considering a global optimization

            double minScore = 10000000;
            utils::Pose bestPose = currentPose;
            size_t finalSetSize = matchedPointSize;

            uint iteration = 0; 
            while(iteration < maxIterations)
            {
                const matches_containers::match_point_container& selectedMatches = get_n_random_matches(matchedPoints, minimumPointsForOptimization);
                const utils::Pose& pose = Pose_Optimization::get_optimized_global_pose(currentPose, matchedPoints);

                const matrix34& transformationMatrix = utils::compute_camera_to_world_transform(pose.get_orientation_quaternion(), pose.get_position());

                // Select inliers by retroprojection threshold
                matches_containers::match_point_container inliersContainer;
                for (const matches_containers::point_pair& match : matchedPoints)
                {
                    if (get_distance_to_point(match.second, match.first, transformationMatrix) < threshold)
                    {
                        inliersContainer.insert(inliersContainer.end(), match);
                    }
                }

                // if inlier count is good enought, try a global optimization
                if (inliersContainer.size() > minimumPointsForOptimization and inliersContainer.size() >= minimumPointsForEvaluation)
                {
                    const utils::Pose& newPose = Pose_Optimization::get_optimized_global_pose(pose, inliersContainer);
                    const double poseScore = get_pose_score(newPose, inliersContainer);
                    if (poseScore < minScore)
                    {
                        minScore = poseScore;
                        bestPose = newPose;
                        finalSetSize = inliersContainer.size();

                        if (poseScore <= acceptableMinimumScore)
                            // We can stop here, the optimization is pretty good
                            break;
                    }
                }
                ++iteration;
            }

            if (matchedPointSize != finalSetSize)
                std::cout << matchedPointSize << " " <<  finalSetSize << std::endl;
            return bestPose;
        }

        /**
         *
         * \param[in] optimizedPose the pose obtained after one optimization
         * \param[in] matchedPoints the original matched point container
         * \param[out] retroprojectionErrorContainer A container with all the errors of the matched points
         *
         * \return the mean of the error vector
         */
        double get_retroprojection_error_vector(const utils::Pose& optimizedPose, const matches_containers::match_point_container& matchedPoints, std::vector<double>& retroprojectionErrorContainer)
        {
            assert(retroprojectionErrorContainer.size() == 0);
            assert(matchedPoints.size() > 0);

            retroprojectionErrorContainer.clear();
            retroprojectionErrorContainer.reserve(matchedPoints.size());

            double sumOfErrors = 0;
            const matrix34& newTransformationMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());
            for(matches_containers::match_point_container::const_iterator pointIterator = matchedPoints.cbegin(); pointIterator != matchedPoints.cend(); ++pointIterator) {
                const vector2& screenPoint = vector2(pointIterator->first.x(), pointIterator->first.y());
                const vector3& worldPoint = pointIterator->second;

                const vector2& projectedPoint = utils::world_to_screen_coordinates(worldPoint, newTransformationMatrix);
                const double projectionError = (projectedPoint - screenPoint).norm();
                sumOfErrors += projectionError;
                retroprojectionErrorContainer.push_back(projectionError);
            }
            return sumOfErrors / matchedPoints.size();
        }

        /**
         * \brief Remove the outliers of the matched point container by excluding the 5% of errors
         *
         * \param[in] matchedPoints the original matched point container
         * \param[in] retroprojectionErrorContainer error of the retroprojection
         * \param[in] retroprojectionErrorMean mean of the error vector
         *
         * \return a matched point container without outliers
         */
        const matches_containers::match_point_container remove_match_outliers(const matches_containers::match_point_container& matchedPoints, const std::vector<double>& retroprojectionErrorContainer, const double retroprojectionErrorMean)
        {
            assert(retroprojectionErrorContainer.size() > 0);
            assert(matchedPoints.size() > 0);

            double errorVariance = 0;
            for(const double& error : retroprojectionErrorContainer) 
            {
                errorVariance += std::pow(error - retroprojectionErrorMean, 2.0);
            }
            errorVariance /= retroprojectionErrorContainer.size();
            const double errorStandardDeviation = std::max(sqrt(errorVariance), 0.5);

            // standard threshold: 
            const double stdError = errorStandardDeviation * 1.5;  // 95% not outliers
            const double highThresh = retroprojectionErrorMean + stdError;

            matches_containers::match_point_container newMatchedPoints;
            size_t cnt = 0;
            for(matches_containers::match_point_container::const_iterator pointIterator = matchedPoints.cbegin(); pointIterator != matchedPoints.cend(); ++pointIterator, ++cnt) {
                if (retroprojectionErrorContainer[cnt] <= highThresh)
                {
                    const vector3& screenPoint = pointIterator->first;
                    const vector3& worldPoint = pointIterator->second;

                    newMatchedPoints.emplace(newMatchedPoints.end(), screenPoint, worldPoint);
                }
            }

            return newMatchedPoints;
        }

        const utils::Pose Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const matches_containers::match_point_container& matchedPoints) 
        {
            assert(matchedPoints.size() > 0);

            // get absolute displacement error
            const double maximumOptimizationRetroprojection = Parameters::get_maximum_optimization_retroprojection_error();
            const size_t maximumOptimizationReiteration = Parameters::get_maximum_optimization_reiteration();
            const size_t minimumPointsForOptimization = Parameters::get_minimum_point_count_for_optimization();

            assert(matchedPoints.size() >= minimumPointsForOptimization);

            utils::Pose finalPose = currentPose;
            matches_containers::match_point_container finalMatchedPoints = matchedPoints;
            double errorOfSet = 0.0;    // mean retroprojection error of this matches set. Should be close to zero
            size_t iterationCount = 0;

            // restart optimization while removing outliers from the next iteration
            // DO this until we reach a good state (low error), or the max iteration count is reached
            do
            {
                // Compute an optimized pose from this matched point set
                const utils::Pose& optimizedPose = get_optimized_global_pose(currentPose, finalMatchedPoints);

                // compute the retroprojection error of the matches with the optimized pose
                std::vector<double> retroprojectionErrorContainer;
                const double meanOfErrorContainer = get_retroprojection_error_vector(currentPose, finalMatchedPoints, retroprojectionErrorContainer);
                errorOfSet = meanOfErrorContainer;

                // get a new matched point set with outliers (hopefully) removed
                const matches_containers::match_point_container& newMatchedPoints = remove_match_outliers(finalMatchedPoints, retroprojectionErrorContainer, meanOfErrorContainer);

                finalPose = optimizedPose;
                if (errorOfSet < maximumOptimizationRetroprojection)
                {
                    // Reached a good optimization state
                    finalMatchedPoints = newMatchedPoints;
                    break;
                }
                else if (newMatchedPoints.size() == finalMatchedPoints.size())
                {
                    // No matches removed: reached a steady state (but the error is still too great)
                    break;
                }
                else if (newMatchedPoints.size() < minimumPointsForOptimization)
                {
                    // We removed too much points for the optimization process
                    utils::log_error("Not enough points for pose optimization after outlier removal");
                    break;
                }
                // update the matched point list
                finalMatchedPoints = newMatchedPoints;

                ++iterationCount;
            }  while (iterationCount <= maximumOptimizationReiteration);

            if (errorOfSet > maximumOptimizationRetroprojection)
            {
                // The optimized pose is still not good enough
                utils::log_error("Optimization error is too high");
                return finalPose;
            }
            return finalPose;
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
