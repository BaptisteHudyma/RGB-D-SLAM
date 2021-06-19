#ifndef POSE_OPTIMISATION_HPP
#define POSE_OPTIMISATION_HPP 

#include "types.hpp"

#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

#include <iostream>

namespace poseEstimation {

        

    /**
     * \brief Structure given to the Levenberg-Marquardt algorithm. It optimizes a rotation (quaternion) and a translation (vector3) using the matched features from a frame to the local map, using their distances to one another as the main metric.
     *
     */
    template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
        struct Levenberg_Marquard_Functor {

            //tell the called the numerical type and input/ouput size
            typedef _Scalar Scalar;
            enum {
                InputsAtCompileTime = NX,
                ValuesAtCompileTime = NY
            };

            typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
            typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
            typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;


            /*Levenberg_Marquard_Functor() :
              _M(InputsAtCompileTime), _N(ValuesAtCompileTime)
              {}*/

            Levenberg_Marquard_Functor(unsigned int inputCount, unsigned int outputCount) :
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

    typedef std::pair<vector3, vector3> point_pair;
    typedef std::list<point_pair> matched_point_container;

    struct Pose_Estimator: Levenberg_Marquard_Functor<double> 
    {
        // Simple constructor
        Pose_Estimator(unsigned int n, const matched_point_container& points) :
            Levenberg_Marquard_Functor<double>(n, points.size()) 
        {
            //copy points
            _points = points;
        }

        // Implementation of the objective function
        int operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
            vector3 translation(z(0), z(1), z(2));
            quaternion rotation(z(3), z(4), z(5), z(6));
            rotation.normalize();
            matrix33 rotationMatrix = rotation.toRotationMatrix();

            unsigned int i = 0;
            for(const point_pair& pointPair : _points) {
                const vector3& p1 = pointPair.first;
                const vector3& p2 = pointPair.second;

                //pose error
                const vector3 dist = p1 - ( rotationMatrix *  p2 + translation );
                //fvec(i) = dist.squaredNorm(); 
                //fvec(i) = dist.norm(); 

                //manhattan
                fvec(i) = abs(dist[0]) + abs(dist[1]) + abs(dist[2]);

                ++i;
            }
            //std::cout << fvec.transpose() << std::endl;
            return 0;
        }

        private:
        matched_point_container _points; 

    };

    struct Pose_Functor : Eigen::NumericalDiff<Pose_Estimator> {};

}       /* poseEstimation */

#endif
