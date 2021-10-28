#include "PoseOptimization.hpp"

#include "utils.hpp"
#include "LevenbergMarquardFunctors.hpp"
#include "parameters.hpp"

#include <Eigen/StdVector>

namespace rgbd_slam {
    namespace pose_optimization {

        const utils::Pose Pose_Optimization::compute_optimized_pose(const utils::Pose& currentPose, const match_point_container& matchedPoints) 
        {
            const utils::Pose& newGlobalPose = get_optimized_global_pose(currentPose, matchedPoints);

            return newGlobalPose;
        }


        const utils::Pose Pose_Optimization::get_optimized_global_pose(const utils::Pose& currentPose, const match_point_container& matchedPoints) 
        {
            const vector3& position = currentPose.get_position();    // Work in millimeters
            const quaternion& rotation = currentPose.get_orientation_quaternion();

            // Compute B matrix
            const matrix43& singularBValues = get_B_singular_values(rotation);

            // Vector to optimize: (0, 1, 2) is position,
            // Vector (3, 4, 5) is a rotation parametrization, representing a delta in rotation in the tangential hyperplane -From Using Quaternions for Parametrizing 3-D Rotation in Unconstrained Nonlinear Optimization)
            Eigen::VectorXd input(6);
            // 3D pose
            input[0] = position.x();
            input[1] = position.y();
            input[2] = position.z();
            // X Y Z of a quaternion representation (0, 0, 0) corresponds to the quaternion itself
            input[3] = 0;
            input[4] = 0;
            input[5] = 0;

            // Optimize function 
            Global_Pose_Functor pose_optimisation_functor(
                    Global_Pose_Estimator(
                        input.size(), 
                        matchedPoints, 
                        currentPose.get_position(),
                        currentPose.get_orientation_quaternion(),
                        singularBValues
                        )
                    );
            // Optimization algorithm
            Eigen::LevenbergMarquardt<Global_Pose_Functor, double> poseOptimizator( pose_optimisation_functor );

            // maxfev   : maximum number of function evaluation
            // xtol     : tolerance for the norm of the solution vector
            // ftol     : tolerance for the norm of the vector function
            // gtol     : tolerance for the norm of the gradient of the error function
            // factor   : step bound for the diagonal shift
            // epsfcn   : error precision
            poseOptimizator.parameters.maxfev = Parameters::get_optimization_maximum_iterations();
            poseOptimizator.parameters.epsfcn = Parameters::get_optimization_error_precision();
            poseOptimizator.parameters.xtol = Parameters::get_optimization_xtol();
            poseOptimizator.parameters.ftol = Parameters::get_optimization_ftol();
            poseOptimizator.parameters.gtol = Parameters::get_optimization_gtol();
            poseOptimizator.parameters.factor = Parameters::get_optimization_factor();


            const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimizator.minimize(input);

            const quaternion& endRotation = get_quaternion_from_original_quaternion(rotation, vector3(input[3], input[4], input[5]), singularBValues); 
            const vector3 endPosition(
                    input[0],
                    input[1],
                    input[2]
                    );

            if (endStatus == Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation)
            {
                // Error: reached end of minimization without reaching a minimum
                const std::string message = get_human_readable_end_message(endStatus);
                std::cerr << matchedPoints.size() << " pts | Position is " << endPosition.transpose() << " | Result " << endStatus << " (" << message << ")" << std::endl;
            }

            // Update refine pose with optimized pose
            return utils::Pose(endPosition, endRotation);
        }

    }   /* pose_optimization*/
}   /* rgbd_slam */
