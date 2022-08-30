#ifndef RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP
#define RGBDSLAM_POSEOPTIMIZATION_LMFUNCTORS_HPP

#include "../types.hpp"
#include "../utils/matches_containers.hpp"

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


        quaternion get_quaternion_exponential(const quaternion& quat);
        quaternion get_quaternion_logarithm(const quaternion& quat);



        /**
         * \brief Implementation of the main pose and orientation optimisation, to be used by the Levenberg Marquard optimisator. 
         */
        struct Global_Pose_Estimator :
            Levenberg_Marquardt_Functor<double>
        {
            // Simple constructor
            /**
             * \param[in] n Number of input parameters 
             * \param[in,out] points Matched 2D (screen) to 3D (world) points
             * \param[in] worldPosition Position of the observer in the world
             * \param[in] worldRotation Orientation of the observer in the world
             */
            Global_Pose_Estimator(const size_t n, const matches_containers::match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation);

            /**
             * \brief Implementation of the objective function
             *
             * \param[in] x The vector of parameters to optimize (Size M)
             * \param[out] fvec The vector of errors, of size N (N the number of points) 
             */
            int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const;

            private:
            const matches_containers::match_point_container& _points; 
            const quaternion _rotation;
            const vector3 _position;

            // Error function parameters
            const double _pointErrorMultiplier;
            const double _lossScale;
            const double _lossAlpha;
        };

        struct Global_Pose_Functor : Eigen::NumericalDiff<Global_Pose_Estimator> {};


        /**
         * \brief Use for debug.
         * \return Returns a string with the human readable version of Eigen LevenbergMarquardt output status
         */
        const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status);

    }       /* pose_optimization*/
}   /* rgbd_slam */

#endif
