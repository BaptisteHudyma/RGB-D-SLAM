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
            return (pointA - pointB).norm();
        }
        double get_distance_squared(const vector3& pointA, const vector3& pointB) 
        { 
            return (pointA - pointB).norm();
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
            assert(scale > 0);

            const double scaledSquaredError = (error * error) / (scale * scale);

            // ]2, oo[
            if (alpha > 2)
            {
                const double internalTerm = scaledSquaredError / abs(alpha - 2) + 1;
                return (abs(alpha - 2) / alpha) * ( pow(internalTerm, alpha / 2.0) - 1);
            }
            // ]0, 2]
            else if (alpha <= 2 and alpha > 0)
            {
                return 0.5 * scaledSquaredError;
            }
            // ]-100, 0]
            else if (alpha <= 0 and alpha > -100)
            {
                return log(0.5 * scaledSquaredError + 1);
            }
            // ]-oo, -100]
            else 
            {
                return 1 - exp( -0.5 * scaledSquaredError);
            }
        }

        /**
         * \brief Compute a scaled axis representation of a rotation quaternion. The scaled axis is easier to optimize for Levenberg-Marquardt algorithm
         */
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

        /**
         * \brief Compute a quaternion from it's scaled axis representation
         */
        quaternion get_quaternion_from_scale_axis_coefficients(const vector3 optimizationCoefficients)
        {
            const double a = optimizationCoefficients.norm();
            const double ha = a * 0.5;
            const double scale = (a > 0) ? (sin(ha) / a) : 0.5;
            return quaternion(cos(ha), optimizationCoefficients.x() * scale, optimizationCoefficients.y() * scale, optimizationCoefficients.z() * scale);
        }

        /**
         * GLOBAL POSE ESTIMATOR members
         */

        Global_Pose_Estimator::Global_Pose_Estimator(const size_t n, const matches_containers::match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation) :
            Levenberg_Marquardt_Functor<double>(n, points.size()),
            _points(points),
            _rotation(worldRotation),
            _position(worldPosition),
            _pointErrorMultiplier( sqrt(Parameters::get_point_error_multiplier() / static_cast<double>(points.size())) ),
            _lossScale(Parameters::get_point_loss_scale()),
            _lossAlpha(Parameters::get_point_loss_alpha())
        {
            assert(_lossScale > 0);
            assert(_pointErrorMultiplier > 0);
            assert(_points.size() > 0);
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const 
        {
            assert(_points.size() > 0);
            assert(x.size() == 6);
            assert(static_cast<size_t>(fvec.size()) == _points.size());

            // Get the new estimated pose
            const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(vector3(x(3), x(4), x(5)));
            const vector3 translation(x(0), x(1), x(2));

            const matrix34& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
            double meanOfDistances = 0;
            size_t pointIndex = 0;  // index of the match being treated

            // Compute retroprojection distances
            for(matches_containers::match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++pointIndex) {
                // Compute retroprojected distance
                const double distance = get_distance_to_point(pointIterator->second, pointIterator->first, transformationMatrix);
                assert(distance >= 0);

                meanOfDistances += distance; 
                fvec(pointIndex) = distance;
            }

            const size_t pointContainerSize = _points.size();
            meanOfDistances /= static_cast<double>(pointContainerSize);

            // If the mean of distance is 0, no need to continue 
            assert(meanOfDistances >= 0);
            if (meanOfDistances > 0)
            {
                // Compute distance scores
                for(size_t i = 0; i < pointContainerSize; ++i)
                {
                    // distance squared divided by mean of all distances
                    const double distance = (fvec(i) * fvec(i)) / meanOfDistances;

                    // Pass it to loss function
                    const double weightedLoss = get_generalized_loss_estimator(distance, _lossAlpha, _lossScale);

                    // Compute the final error
                    fvec(i) = _pointErrorMultiplier * weightedLoss; 
                }
            }
            return 0;
        }

        double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
        {
            const vector2 matchedPointAs2D(matchedPoint.x(), matchedPoint.y());
            vector2 mapPointAs2D; 
            const bool isCoordinatesValid = utils::world_to_screen_coordinates(mapPoint, worldToCamMatrix, mapPointAs2D);
            if(isCoordinatesValid)
                return get_distance_manhattan(matchedPointAs2D, mapPointAs2D);
            // randomly high number
            return 10000;
        }

        /*
           double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& worldToCamMatrix) const
           {
           vector2 mapPointAs2D;
           const bool isCoordinatesValid = utils::world_to_screen_coordinates(mapPoint, worldToCamMatrix, mapPointAs2D);
           if (isCoordinatesValid)
           {
           const vector3 mapPointAs3D(mapPointAs2D.x(), mapPointAs2D.y(), mapPoint.z());

           return get_distance_manhattan(matchedPoint, mapPointAs3D);
           }
           return 10000;
           }
         */
        /*
           double Global_Pose_Estimator::get_distance_to_point(const vector3& mapPoint, const vector3& matchedPoint, const matrix34& camToWorldMatrix) const
           {
           const vector3& matchedPointAs3D = utils::screen_to_world_coordinates( matchedPoint.x(), matchedPoint.y(), matchedPoint.z(), camToWorldMatrix);

           return get_distance_manhattan(matchedPointAs3D, mapPoint);
           }
         */


        /**
         * \brief Return a string corresponding to the end status of the optimization
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
