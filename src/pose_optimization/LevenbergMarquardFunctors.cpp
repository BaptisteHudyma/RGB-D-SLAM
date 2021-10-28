#include "LevenbergMarquardFunctors.hpp"

#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {
    namespace pose_optimization {


        double get_distance_manhattan(const vector2& pointA, const vector2& pointB) 
        { 
            return 
                abs(pointA.x() - pointB.x()) + 
                abs(pointA.y() - pointB.y());
        }
        double get_distance_manhattan(const vector3& pointA, const vector3& pointB) 
        { 
            return 
                abs(pointA.x() - pointB.x()) + 
                abs(pointA.y() - pointB.y()) +
                abs(pointA.z() - pointB.z());
        }
        double get_distance_squared(const vector2& pointA, const vector2& pointB) 
        { 
            return 
                pow(pointA.x() - pointB.x(), 2.0) + 
                pow(pointA.y() - pointB.y(), 2.0);
        }
        double get_distance_squared(const vector3& pointA, const vector3& pointB) 
        { 
            return 
                pow(pointA.x() - pointB.x(), 2.0) + 
                pow(pointA.y() - pointB.y(), 2.0) +
                pow(pointA.z() - pointB.z(), 2.0);
        }

        /**
         * \brief Implementation of "A General and Adaptive Robust Loss Function" (2019)
         * By Jonathan T. Barron
         *
         * \param[in] error The error to pass to the loss function
         * \param[in] apha The steepness of the loss function. For alpha == 2, this is a L2 loss, alpha == 1 is Charbonnier loss, alpha == 0 is Cauchy loss, alpha == 0 is a German MCClure and alpha == - infinity is Welsch loss
         * \param[in] scale Standard deviation of the error, as a scale parameter
         */
        double get_generalized_loss_estimator(const double error, const double alpha = 1, const double scale = 1)
        {
            const double scaledSquaredError = (error * error) / (scale * scale);

            if (alpha == 2)
            {
                return 0.5 * scaledSquaredError;
            }
            else if (alpha == 0)
            {
                return log(0.5 * scaledSquaredError + 1);
            }
            else if (alpha < -100)
            {
                return 1 - exp( -0.5 * scaledSquaredError);
            }
            else
            {
                const double internalTerm = scaledSquaredError / abs(alpha - 2) + 1;
                return (abs(alpha - 2) / alpha) * ( pow(internalTerm, alpha / 2.0) - 1);
            }
        }

        const matrix43 get_B_singular_values(const quaternion& rotation)
        {
            const Eigen::MatrixXd BMatrix {
                {- rotation.x() / rotation.w(), - rotation.y() / rotation.w(), - rotation.z() / rotation.w()},
                    {1, 0, 0},
                    {0, 1, 0},
                    {0, 0, 1}
            };
            return Eigen::JacobiSVD<Eigen::MatrixXd>(BMatrix, Eigen::ComputeThinU).matrixU();
        }


        const quaternion get_quaternion_from_original_quaternion(const quaternion& originalQuaternion, const vector3& estimationVector, const matrix43& transformationMatrixB)
        {
            vector4 transformedEstimationVector = transformationMatrixB * estimationVector;
            const double normOfV4 = transformedEstimationVector.norm();
            if (normOfV4 == 0)
                return originalQuaternion;

            // Normalize v4
            transformedEstimationVector /= normOfV4;

            const vector4 quaternionAsVector(originalQuaternion.x(), originalQuaternion.y(), originalQuaternion.z(), originalQuaternion.w());
            // Compute final quaternion
            const vector4 finalQuaternion = sin(normOfV4) * transformedEstimationVector + cos(normOfV4) * quaternionAsVector;
            return quaternion(finalQuaternion.w(), finalQuaternion.x(), finalQuaternion.y(), finalQuaternion.z());
        }


        Global_Pose_Estimator::Global_Pose_Estimator(const unsigned int n, const match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation, const matrix43& singularBvalues) :
            Levenberg_Marquardt_Functor<double>(n, points.size()),
            _points(points),
            _rotation(worldRotation),
            _position(worldPosition),
            _singularBvalues(singularBvalues)
        {
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const 
        {
            const quaternion& rotation = get_quaternion_from_original_quaternion(_rotation, vector3(x(3), x(4), x(5)), _singularBvalues);
            const vector3 translation(
                    x(0),
                    x(1),
                    x(2)
                    );

            const double sqrtOfErrorMultiplier = sqrt(Parameters::get_point_error_multiplier() / _points.size());
            const double lossAlpha = Parameters::get_point_loss_alpha();
            const double lossScale = Parameters::get_point_loss_scale();

            const matrix34& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);

            unsigned int pointIndex = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {

                // Compute distance
                const double distance = get_distance_to_point(pointIterator->second, pointIterator->first, transformationMatrix);

                // Pass it to loss function
                const double weightedLoss = get_generalized_loss_estimator(distance * distance, lossAlpha, lossScale);

                // Compute the final error
                fvec(pointIndex) = sqrtOfErrorMultiplier * weightedLoss; 
            }
            return 0;
        }


        /*int Global_Pose_Estimator::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
        {
            const double epsilon = Parameters::get_optimization_error_precision();
            for (int i = 0; i < x.size(); i++) {
                Eigen::VectorXd xPlus(x);
                xPlus(i) += epsilon;
                Eigen::VectorXd xMinus(x);
                xMinus(i) -= epsilon;

                Eigen::VectorXd fvecPlus(values());
                operator()(xPlus, fvecPlus);

                Eigen::VectorXd fvecMinus(values());
                operator()(xMinus, fvecMinus);

                Eigen::VectorXd fvecDiff(values());
                fvecDiff = (fvecPlus - fvecMinus) / (2.0 * epsilon);

                fjac.block(0, i, values(), 1) = fvecDiff;
            }

            return 0;
        }*/


        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
        {
            const vector2 matchedPointAs2D(matchedPoint.x(), matchedPoint.y());
            const vector2& mapPointAs2D = utils::world_to_screen_coordinates(mapPoint, worldToCamMatrix);

            const double distance = get_distance_manhattan(matchedPointAs2D, mapPointAs2D);
            return distance;
        }




        const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status) 
        {
            switch(status) {
                case Eigen::LevenbergMarquardtSpace::Status::NotStarted :
                    return "not started";
                case Eigen::LevenbergMarquardtSpace::Status::Running :
                    return "running";
                case Eigen::LevenbergMarquardtSpace::Status::ImproperInputParameters :
                    return "improper input parameters";
                case Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall :
                    return "relative reduction too small";
                case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall :
                    return "relative error too small";
                case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall :
                    return "relative error and reduction too small";
                case Eigen::LevenbergMarquardtSpace::Status::CosinusTooSmall :
                    return "cosinus too small";
                case Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation :
                    return "too many function evaluation";
                case Eigen::LevenbergMarquardtSpace::Status::FtolTooSmall :
                    return "xtol too small";
                case Eigen::LevenbergMarquardtSpace::Status::XtolTooSmall :
                    return "ftol too small";
                case Eigen::LevenbergMarquardtSpace::Status::GtolTooSmall :
                    return "gtol too small";
                case Eigen::LevenbergMarquardtSpace::Status::UserAsked :
                    return "user asked";
                default:
                    return "error: empty message";
            }
            return std::string("");
        }


    } /* pose_optimization */
} /* rgbd_slam */
