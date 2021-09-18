#include "PoseOptimization.hpp"

#include "utils.hpp"
#include "LevenbergMarquardFunctors.hpp"
#include "parameters.hpp"

#include <Eigen/StdVector>

namespace rgbd_slam {
    namespace utils {

        const poseEstimation::Pose Pose_Optimization::compute_optimized_pose(const poseEstimation::Pose& currentPose, match_point_container& matchedPoints)
        {
            const poseEstimation::Pose& newGlobalPose = get_optimized_global_pose(currentPose, matchedPoints);

            return newGlobalPose;
        }


        const poseEstimation::Pose Pose_Optimization::get_optimized_global_pose(const poseEstimation::Pose& currentPose, match_point_container& matchedPoints)
        {
            const vector3& position = currentPose.get_position();
            const quaternion& rotation = currentPose.get_orientation_quaternion();

            // Compute B matrix
            const Eigen::MatrixXd BMatrix {
                {- rotation.x() / rotation.w(), - rotation.y() / rotation.w(), - rotation.z() / rotation.w()},
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1}
            };
            const matrix43& singularBValues = Eigen::JacobiSVD<Eigen::MatrixXd>(BMatrix, Eigen::ComputeThinU).matrixU();
            
            // Vector to optimize: (0, 1, 2) is position,
            // Vector (3, 4, 5) is a rotation parametrization, representing a delta in rotation in the tangential hyperplane -From Using Quaternions for Parametrizing 3-D Rotation in Unconstrained Nonlinear Optimization)
            Eigen::VectorXd input(6);
            // 3D pose
            input[0] = position.x();
            input[1] = position.y();
            input[2] = position.z();
            // X Y Z of a quaternion
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
            Eigen::LevenbergMarquardt<Global_Pose_Functor, double> poseOptimisator( pose_optimisation_functor );

            // xtol     : tolerance for the norm of the solution vector
            // ftol     : tolerance for the norm of the vector function
            // gtol     : tolerance for the norm of the gradient of the error function
            // factor   : step bound for the diagonal shift
            // epsfcn   : error precision
            // maxfev   : maximum number of function evaluation
            poseOptimisator.parameters.maxfev = Parameters::get_maximum_global_optimization_iterations();

            const Eigen::LevenbergMarquardtSpace::Status endStatus = poseOptimisator.minimize(input);

            const quaternion& endRotation = get_quaternion_from_original_quaternion(rotation, vector3(input[3], input[4], input[5]), singularBValues); 
            const vector3 endPosition(input[0], input[1], input[2]);

            if (endStatus == Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation)
            {
                // Error: reached end of minimization without reaching a minimum
                const std::string message = get_human_readable_end_message(endStatus);
                std::cerr << matchedPoints.size() << " pts " << endPosition.transpose() << ". Result " << endStatus << " (" << message << ")" << std::endl;
            }

            // Update refine pose with optimized pose
            return poseEstimation::Pose(endPosition, endRotation);
        }

    }   /* utils */
}   /* rgbd_slam */
