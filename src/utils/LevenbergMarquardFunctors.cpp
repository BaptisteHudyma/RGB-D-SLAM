#include "LevenbergMarquardFunctors.hpp"

#include "parameters.hpp"

namespace rgbd_slam {
    namespace utils {


        double get_distance_manhattan(const vector3& pointA, const vector3& pointB) {
            return abs(pointA[0] - pointB[0]) + 
                abs(pointA[1] - pointB[1]) + 
                abs(pointA[2] - pointB[2]);
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

        double get_hubert_estimator(const double score)
        {
            const double absScore = abs(score);
            const double hubertThreshold = Parameters::get_point_Hubert_threshold();

            if (absScore < hubertThreshold)
            {
                return 0.5 * pow(score, 2.0);
            }
            else
            {
                return hubertThreshold * (absScore - 0.5 * hubertThreshold);
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


        Pose_Estimator::Pose_Estimator(const unsigned int n, match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation) :
            Levenberg_Marquard_Functor<double>(n, points.size()),
            _points(points),
            _position(worldPosition),
            _rotation(worldRotation)
        {
            _weights = std::vector<double>(points.size());

            matrix34 transformationMatrix;
            transformationMatrix << _rotation.toRotationMatrix(), _position;

            std::vector<double> errors(points.size());
            std::vector<double> medianErrorVector(points.size());
            unsigned int pointCount = 0;
            // Compute the start error
            for(match_point_container::const_iterator pointIterator = points.cbegin(); pointIterator != points.cend(); ++pointIterator, ++pointCount) {
                const vector3& detectedPoint = pointIterator->first;
                const vector3& point3D = utils::screen_to_world_coordinates(detectedPoint(0), detectedPoint(1), detectedPoint(2), transformationMatrix); 

                const double error = get_distance_manhattan(pointIterator->second, point3D); 
                errors[pointCount] = error;
                medianErrorVector[pointCount] = error;
            }
            // Compute median of all errors
            const double medianOfErrors = get_median(medianErrorVector);;
            
            for(unsigned int i = 0; i < errors.size(); ++i)
            {
                medianErrorVector[i] = errors[i] - medianOfErrors;
            }
            // Compute (error - median of errors) median
            const double medianOfErrorsMedian = get_median(medianErrorVector);

            // Fill weights
            for (unsigned int i = 0; i < points.size(); ++i)
            {
                _weights[i] = get_weight(errors[i] - medianOfErrors, medianOfErrorsMedian);
            }
        }

        // Implementation of the objective function
        int Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
            quaternion rotation(z(3), z(4), z(5), z(6));
            rotation.normalize();
            vector3 translation(z(0), z(1), z(2));

            // Convert to world coordinates
            rotation = _rotation * rotation;
            translation += _position;

            const double pointErrorMultiplier = Parameters::get_point_error_multiplier();

            matrix34 transformationMatrix;
            transformationMatrix << rotation.toRotationMatrix(), translation;

            unsigned int pointIndex = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {
                const vector3& detectedPoint = pointIterator->first;
                const vector3& point3D = utils::screen_to_world_coordinates(detectedPoint(0), detectedPoint(1), detectedPoint(2), transformationMatrix); 

                //Supposed Sum of squares: 
                // For some reason, sqrtf is waaaayyy faster than sqrtl, which is faster than sqrt
                // Maybe the lesser precision ? it's an advantage here
                fvec(pointIndex) = sqrtf(pointErrorMultiplier * _weights[pointIndex] * get_hubert_estimator(get_distance_manhattan(pointIterator->second, point3D))); 
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
