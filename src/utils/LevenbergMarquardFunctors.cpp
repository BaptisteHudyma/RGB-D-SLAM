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


        Local_Pose_Estimator::Local_Pose_Estimator(const unsigned int n, match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation, const matrix43& singularBvalues) :
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
                const vector2 detectedPoint(pointIterator->first.x(), pointIterator->first.y());
                const vector2& mapPoint = world_to_screen_coordinates(pointIterator->second, transformationMatrix);

                const double error = get_distance_manhattan(mapPoint, detectedPoint); 
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
        int Local_Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
            const quaternion& rotation = get_quaternion_from_original_quaternion(_rotation, vector3(z(3), z(4), z(5)), _singularBvalues);
            const vector3 translation(z(0), z(1), z(2));

            const double pointErrorMultiplier = sqrt(Parameters::get_point_error_multiplier() / _points.size());

            const matrix34& transformationMatrix = compute_world_to_camera_transform(rotation, translation);

            unsigned int pointIndex = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {
                // Project detected point to 3D space
                const vector2 detectedPoint(pointIterator->first.x(), pointIterator->first.y());
                const vector2& mapPoint = world_to_screen_coordinates(pointIterator->second, transformationMatrix);
                
                // Compute distance and pass it to loss function
                const double distance = get_distance_manhattan(mapPoint, detectedPoint);
                const double weightedLoss = get_generalized_loss_estimator(distance, Parameters::get_point_loss_alpha(), _medianOfDistances);

                // Compute the final error
                // sqrtf is faster than sqrtl, which is faster than sqrt
                // Maybe the lesser precision ? it's an advantage here
                fvec(pointIndex) = _weights[pointIndex] * pointErrorMultiplier * sqrtf(weightedLoss) ; 
            }
            return 0;
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
