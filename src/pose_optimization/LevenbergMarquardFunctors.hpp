#ifndef RGBDSLAM_UTILS_LM_FUNCTORS
#define RGBDSLAM_UTILS_LM_FUNCTORS

#include "types.hpp"

// types
#include "map_point.hpp"

#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

#include <iostream>

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

                Levenberg_Marquardt_Functor(const unsigned int inputCount, const unsigned int outputCount) :
                    _M(inputCount), _N(outputCount)
                {
                }

                unsigned int values() const {
                    return _N;
                }
                unsigned int inputs() const {
                    return _M;
                }

                unsigned int _M;
                unsigned int _N;
            };



        /**
         * \brief Compute a matrix parametrization of the quaternion, to be used with get_quaternion_from_original_quaternion
         *
         * \param[in] rotation The original quaternion
         *
         * \return A parametrization matrix
         */
        const matrix43 get_B_singular_values(const quaternion& rotation);

        /**
         * \brief Return a quaternion from an ideal parametrization estimationVector
         *
         * \param[in] originalQuaternion Original position used to compute transformationMatrixB
         * \param[in] estimationVector Estimated vector, optimized by levenberg marquardt, to turn back to a quaternion
         * \param[in] transformationMatrixB Transformation matrix used to transform estimationVector back to a quaternion
         */
        const quaternion get_quaternion_from_original_quaternion(const quaternion& originalQuaternion, const vector3& estimationVector, const matrix43& transformationMatrixB);



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
            Global_Pose_Estimator(const unsigned int n, const match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation, const matrix43& singularBvalues);

            /**
             * \brief Return te distance between the map point and the it's matched point
             *
             * \param[in] mapPoint The map point in 3D world coordinates
             * \param[in] matchedPoint The detected & matched point, in 3D screen coordinates
             * \param[in] worldToCamMatrix The matrix to make a transformation from world coordinates to screen coordinates
             *
             * \return The 2D screen distance between those two points
             */
            double get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const;

            /**
             * \brief Implementation of the objective function
             *
             * \param[in] x The vector of parameters to optimize (Size M)
             * \param[out] fvec The vector of errors, of size N (N the number of points) 
             */
            int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const;

            /**
             * \brief Compute the Jacobian matrix of the input parameters
             * 
             * \param[in] x A vector of dimension M, with M the number of parameters to optimize
             * \param[out] fjac A matrix NxM, with M the number of parameters to optimize. This is the Jacobian of the errors
             */
            int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const;

            private:
            const match_point_container& _points; 
            const quaternion _rotation;
            const vector3 _position;
            const matrix43 _singularBvalues;
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
