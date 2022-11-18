#include "levenberg_marquard_functors.hpp"

#include "../utils/camera_transformation.hpp"
#include "coordinates.hpp"
#include <Eigen/src/Core/util/Meta.h>
#include <cmath>
#include <iostream>

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
            quaternion rotation(cos(ha), optimizationCoefficients.x() * scale, optimizationCoefficients.y() * scale, optimizationCoefficients.z() * scale);
            rotation.normalize();
            return rotation;
        }

        /**
         * GLOBAL POSE ESTIMATOR members
         */

        Global_Pose_Estimator::Global_Pose_Estimator(const size_t inputParametersSize, const matches_containers::match_point_container& points, const matches_containers::match_plane_container& planes) :
            Levenberg_Marquardt_Functor<double>(inputParametersSize, points.size() * 2 + planes.size() * 3),
            _points(points),
            _planes(planes)
        {
            assert(not _points.empty() or not _planes.empty());
        }

        // Implementation of the objective function
        int Global_Pose_Estimator::operator()(const vectorxd& optimizedParameters, vectorxd& outputScores) const 
        {
            assert(not _points.empty() or not _planes.empty());
            assert(optimizedParameters.size() == 6);
            assert(static_cast<size_t>(outputScores.size()) == (_points.size() * 2 + _planes.size() * 3));

            // Get the new estimated pose
            const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(
                vector3(optimizedParameters(3), optimizedParameters(4), optimizedParameters(5))
            );
            const vector3 translation(optimizedParameters(0), optimizedParameters(1), optimizedParameters(2));

            const worldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
            Eigen::Index pointIndex = 0;  // index of the match being treated
            // Compute retroprojection distances
            for(const matches_containers::PointMatch& match : _points) {
                // Compute retroprojected distance
                const vector2& distance = match._worldFeature.get_signed_distance_2D(match._screenFeature, transformationMatrix);

                outputScores(pointIndex++) = distance.x();
                outputScores(pointIndex++) = distance.y();
            }

            // add plane optimization vectors
            const planeWorldToCameraMatrix& planeTransformationMatrix = utils::compute_plane_world_to_camera_matrix(transformationMatrix);
            for(const matches_containers::PlaneMatch& match: _planes) {
                const vector3& planeProjectionError = 0.1 * match._worldFeature.get_reduced_signed_distance(match._screenFeature, planeTransformationMatrix);

                outputScores(pointIndex++) = planeProjectionError.x();
                outputScores(pointIndex++) = planeProjectionError.y();
                outputScores(pointIndex++) = planeProjectionError.z();
            }
            return 0;
        }

        double get_transformation_score(const matches_containers::match_point_container& points, const utils::Pose& finalPose)
        {
            // Get the new estimated pose
            const quaternion& rotation = finalPose.get_orientation_quaternion();
            const vector3& translation = finalPose.get_position();

            const worldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
            double meanOfDistances = 0;

            // Compute retroprojection distances
            for(const matches_containers::PointMatch& match : points) {
                // Compute retroprojected distance
                const double distance = match._worldFeature.get_distance(match._screenFeature, transformationMatrix);
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
