#include "LevenbergMarquardFunctors.hpp"

#include "utils.hpp"
#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {


        double get_distance_manhattan(const vector2& pointA, const vector2& pointB) 
        { 
            return 
                abs(pointA.x() - pointB.x()) + 
                abs(pointA.y() - pointB.y());
        }
        /**
         * \brief Compute a weight associated by this error, using a Hubert type loss
         */
        double get_weight(const double error, const double medianOfErrors)
        {
            const double pointWeightThreshold = Parameters::get_point_weight_threshold();
            const double weightCoefficient = Parameters::get_point_weight_coefficient();

            const double theta = weightCoefficient * abs(error - medianOfErrors); 
            const double score = abs(error / theta);
            if (score > pointWeightThreshold)
            {
                return pointWeightThreshold / score;
            }
            return 1;
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
            const double scaledSquaredError = pow(error / scale, 2.0);

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

        double get_median(std::vector<double>& inputVector)
        {
            std::sort(inputVector.begin(), inputVector.end());
            if (inputVector.size() % 2 == 0)
                return (inputVector[inputVector.size() / 2] + inputVector[inputVector.size() / 2 - 1]) / 2;
            else 
                return inputVector[inputVector.size() / 2];
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


        Global_Pose_Estimator::Global_Pose_Estimator(const unsigned int n, match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation, const matrix43& singularBvalues) :
            Levenberg_Marquard_Functor<double>(n, points.size()),
            _points(points),
            _rotation(worldRotation),
            _singularBvalues(singularBvalues)
        {
            _weights = std::vector<double>(points.size());

            const matrix34& transformationMatrix = compute_world_to_camera_transform(_rotation, worldPosition);

            std::vector<double> errors(points.size());
            std::vector<double> medianErrorVector(points.size());
            unsigned int pointCount = 0;
            // Compute the start error
            for(match_point_container::const_iterator pointIterator = points.cbegin(); pointIterator != points.cend(); ++pointIterator, ++pointCount) 
            {
                const double error = get_distance_to_point(pointIterator->second, pointIterator->first, transformationMatrix);

                errors[pointCount] = error;
                medianErrorVector[pointCount] = error;
            }
            // Compute median of all errors
            _medianOfDistances = get_median(medianErrorVector);;

            for(unsigned int i = 0; i < errors.size(); ++i)
            {
                medianErrorVector[i] = errors[i] - _medianOfDistances;
            }
            // Compute (error - median of errors) median
            const double medianOfErrorsMedian = get_median(medianErrorVector);

            // Fill weights
            for (unsigned int i = 0; i < points.size(); ++i)
            {
                _weights[i] = get_weight(errors[i] - _medianOfDistances, medianOfErrorsMedian);
            }
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const 
        {
            const quaternion& rotation = get_quaternion_from_original_quaternion(_rotation, vector3(z(3), z(4), z(5)), _singularBvalues);
            const vector3 translation(z(0), z(1), z(2));

            const double pointErrorMultiplier = sqrt(Parameters::get_point_error_multiplier() / _points.size());

            const matrix34& transformationMatrix = compute_world_to_camera_transform(rotation, translation);

            unsigned int pointIndex = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {

                // Use doubles to make it easier to optimize (smaller precision, but non ritical)

                // Compute distance
                const double distance = get_distance_to_point(pointIterator->second, pointIterator->first, transformationMatrix);
                
                // Pass it to loss function (cut some precision with a float cast)
                const float weightedLoss = get_generalized_loss_estimator(distance, Parameters::get_point_loss_alpha(), _medianOfDistances);

                // Compute the final error
                fvec(pointIndex) = _weights[pointIndex] * pointErrorMultiplier * weightedLoss ; 
            }
            return 0;
        }


        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
        {
            const vector2 matchedPointAs2D(matchedPoint.x(), matchedPoint.y());
            const vector2& mapPointAs2D = world_to_screen_coordinates(mapPoint, worldToCamMatrix);

            return get_distance_manhattan(matchedPointAs2D, mapPointAs2D);
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


    } /* utils */
} /* rgbd_slam */
