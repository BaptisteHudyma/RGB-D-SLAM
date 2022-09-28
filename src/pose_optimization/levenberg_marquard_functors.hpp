#ifndef RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP
#define RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP

#include "../types.hpp"
#include "../utils/matches_containers.hpp"
#include "../utils/pose.hpp"

// types
#include <unsupported/Eigen/NonLinearOptimization>

namespace rgbd_slam {
    namespace pose_optimization {

        /**
         * \brief Structure given to the Levenberg-Marquardt algorithm. It optimizes a rotation (quaternion) and a translation (vector3) using the matched features from a frame to the local map, using their distances to one another as the main metric.
         *
         */
        template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
            struct Levenberg_Marquardt_Functor 
            {

                //tell the called the numerical type and input/ouput size
                typedef _Scalar Scalar;
                enum {
                    InputsAtCompileTime = NX,
                    ValuesAtCompileTime = NY
                };

                // typedefs for the original functor
                typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
                typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
                typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

                Levenberg_Marquardt_Functor(const uint inputCount, const uint outputCount) :
                    _inputCount(inputCount), _outputCount(outputCount)
                {
                }

                uint values() const {
                    return _outputCount;
                }
                uint inputs() const {
                    return _inputCount;
                }

                uint _inputCount;
                uint _outputCount;
            };


        /**
         * \brief Compute a Lie projection of this quaternion for optimization purposes (Scaled Axis representation)
         */
        vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat);

        /**
         * \brief Compute a quaternion from the Lie projection (Scaled Axis representation)
         */
        quaternion get_quaternion_from_scale_axis_coefficients(const vector3& optimizationCoefficients);


        /**
         * \brief Implementation of the main pose and orientation optimisation, to be used by the Levenberg Marquard optimisator. 
         */
        struct Global_Pose_Estimator :
            Levenberg_Marquardt_Functor<double>
        {
            // Simple constructor
            /**
             * \param[in] inputParametersSize Number of input parameters 
             * \param[in,out] points Matched 2D (screen) to 3D (world) points
             */
            Global_Pose_Estimator(const size_t inputParametersSize, const matches_containers::match_point_container& points);

            /**
             * \brief Implementation of the objective function
             *
             * \param[in] optimizedParameters The vector of parameters to optimize (Size M)
             * \param[out] outputScores The vector of errors, of size N (N the number of points) 
             */
            int operator()(const Eigen::VectorXd& optimizedParameters, Eigen::VectorXd& outputScores) const;

            private:
            const matches_containers::match_point_container& _points;
        };

        struct Global_Pose_Functor : Eigen::NumericalDiff<Global_Pose_Estimator> {};

        /**
         * \brief Compute a mean transformation score for a pose and a set of point matches
         */
        double get_transformation_score(const matches_containers::match_point_container& points, const utils::Pose& finalPose);

        /**
         * \brief Use for debug.
         * \return Returns a string with the human readable version of Eigen LevenbergMarquardt output status
         */
        const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status);

    }       /* pose_optimization*/
}   /* rgbd_slam */

#endif
