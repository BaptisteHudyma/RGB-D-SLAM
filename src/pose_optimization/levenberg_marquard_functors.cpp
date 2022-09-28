#include "levenberg_marquard_functors.hpp"

#include "../utils/camera_transformation.hpp"
#include "../utils/distance_utils.hpp"
#include "../parameters.hpp"

namespace rgbd_slam {
    namespace pose_optimization {

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
        quaternion get_quaternion_from_scale_axis_coefficients(const vector3& optimizationCoefficients)
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
            Levenberg_Marquardt_Functor<double>(n, points.size() * 2),
            _points(points),
            _rotation(worldRotation),
            _position(worldPosition)
        {
            assert(not _points.empty());
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const 
        {
            assert(not _points.empty());
            assert(x.size() == 6);
            assert(static_cast<size_t>(fvec.size()) == _points.size() * 2);

            // Get the new estimated pose
            const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(vector3(x(3), x(4), x(5)));
            const vector3 translation(x(0), x(1), x(2));

            const matrix44& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
            size_t pointIndex = 0;  // index of the match being treated
            // Compute retroprojection distances
            for(const matches_containers::Match& match : _points) {
                // Compute retroprojected distance
                const vector2& distance = utils::get_3D_to_2D_distance_2D(match._worldPoint, match._screenPoint, transformationMatrix);

                fvec(pointIndex++) = distance.x();
                fvec(pointIndex++) = distance.y();
            }
            return 0;
        }


        double get_transformation_score(const matches_containers::match_point_container& points, const utils::Pose& finalPose)
        {
            // Get the new estimated pose
            const quaternion& rotation = finalPose.get_orientation_quaternion();
            const vector3& translation = finalPose.get_position();

            const matrix44& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
            double meanOfDistances = 0;

            // Compute retroprojection distances
            for(const matches_containers::Match& match : points) {
                // Compute retroprojected distance
                const double distance = utils::get_3D_to_2D_distance(match._worldPoint, match._screenPoint, transformationMatrix);
                assert(distance >= 0.0);

                meanOfDistances += distance; 
            }
            return meanOfDistances / static_cast<double>(points.size());
        }


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
