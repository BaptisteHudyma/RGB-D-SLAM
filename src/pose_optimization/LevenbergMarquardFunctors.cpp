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

            if (alpha > 2)
            {
                const double internalTerm = scaledSquaredError / abs(alpha - 2) + 1;
                return (abs(alpha - 2) / alpha) * ( pow(internalTerm, alpha / 2.0) - 1);
            }
            if (alpha <= 2 and alpha > 0)
            {
                return 0.5 * scaledSquaredError;
            }
            else if (alpha <= 0 and alpha > -100)
            {
                return log(0.5 * scaledSquaredError + 1);
            }
            else 
            {
                return 1 - exp( -0.5 * scaledSquaredError);
            }
        }

        vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat)
        {
            // forcing positive "w" to work from 0 to PI
            const quaternion& q = (quat.w() >= 0) ? quat : quaternion(-quat.coeffs());
            const vector3& qv = q.vec();

            const double sinha = qv.norm();
            if(sinha > 0)
            {
                const double angle = 2 * atan2(sinha, q.w()); //NOTE: signed
                return (qv * (angle / sinha));
            }
            else{
                // if l is too small, its norm can be equal 0 but norm_inf greater than 0
                // probably w is much bigger that vec, use it as length
                return (qv * (2 / q.w())); ////NOTE: signed
            }
        }

        quaternion get_quaternion_from_scale_axis_coefficients(const vector3 optimizationCoefficients)
        {
            const double a = optimizationCoefficients.norm();
            const double ha = a * 0.5;
            const double scale = (a > 0) ? (sin(ha) / a) : 0.5;
            return quaternion(cos(ha), optimizationCoefficients.x() * scale, optimizationCoefficients.y() * scale, optimizationCoefficients.z() * scale);
        }


        double sinc(double x)
        {
            return (x == 0) ? 1 : (sin(x) / x);
        }

        quaternion get_quaternion_exponential(const quaternion& quat)
        {
            const double a = quat.vec().norm();
            const double expW = exp(quat.w());
            if (a == 0)
            {
                return quaternion(expW, 0, 0, 0);
            }
            quaternion res;
            res.w() = expW * cos(a);
            res.vec() = expW * sinc(a) * quat.vec();
            return res;
        }

        quaternion get_quaternion_logarithm(const quaternion& quat)
        {
            const double expW = quat.norm();
            const double w = log(expW);
            const double a = acos(quat.w() / expW);
            if (a == 0)
            {
                return quaternion(w, 0, 0, 0);
            }
            quaternion res;
            res.w() = w;
            res.vec() = quat.vec() / expW / (sin(a) / a);
            return res;
        }



        Global_Pose_Estimator::Global_Pose_Estimator(const size_t n, const match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation) :
            Levenberg_Marquardt_Functor<double>(n, points.size()),
            _points(points),
            _rotation(worldRotation),
            _position(worldPosition)
        {
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const 
        {
            const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(vector3(x(3), x(4), x(5)));
            const vector3 translation(x(0), x(1), x(2));

            const size_t pointContainerSize = _points.size();

            const double sqrtOfErrorMultiplier = sqrt(Parameters::get_point_error_multiplier() / static_cast<double>(pointContainerSize));
            const double lossAlpha = Parameters::get_point_loss_alpha();
            const double lossScale = Parameters::get_point_loss_scale();

            const matrix34& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);

            double mean = 0;
            size_t pointIndex = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {
                // Compute distance
                const double distance = get_distance_to_point(pointIterator->second, pointIterator->first, transformationMatrix);
                mean += distance / static_cast<double>(pointContainerSize);
                fvec(pointIndex) = distance;
            }

            for(size_t i = 0; i < pointContainerSize; ++i)
            {
                // distance squared divided by mean of all distances
                const double distance = (fvec(i) * fvec(i)) / mean;

                // Pass it to loss function
                const double weightedLoss = get_generalized_loss_estimator(distance, lossAlpha, lossScale);

                // Compute the final error
                fvec(i) = sqrtOfErrorMultiplier * weightedLoss; 
            }
            return 0;
        }

        
        
        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
        {
            const vector2 matchedPointAs2D(matchedPoint.x(), matchedPoint.y());
            const vector2& mapPointAs2D = utils::world_to_screen_coordinates(mapPoint, worldToCamMatrix);

            return get_distance_manhattan(matchedPointAs2D, mapPointAs2D);
        }
        /*
        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
        {
            const vector2& mapPointAs2D = utils::world_to_screen_coordinates(mapPoint, worldToCamMatrix);
            const vector3 mapPointAs3D(mapPointAs2D.x(), mapPointAs2D.y(), mapPoint.z());

            return get_distance_manhattan(matchedPoint, mapPointAs3D);
        }
        */
        /*
        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& camToWorldMatrix) const
        {
            const vector3& matchedPointAs3D = utils::screen_to_world_coordinates( matchedPoint.x(), matchedPoint.y(), matchedPoint.z(), camToWorldMatrix);

            return get_distance_manhattan(matchedPointAs3D, mapPoint);
        }
        */


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
