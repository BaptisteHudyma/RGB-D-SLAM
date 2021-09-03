#include "Pose_Optimisation.hpp"

namespace rgbd_slam {
    namespace poseOptimisation {


        double get_distance_manhattan(const vector3& pointA, const vector3& pointB) {
            return abs(pointA[0] - pointB[0]) + 
                abs(pointA[1] - pointB[1]) + 
                abs(pointA[2] - pointB[2]);
        }


        Pose_Estimator::Pose_Estimator(const unsigned int n, match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation) :
            Levenberg_Marquard_Functor<double>(n, points.size()),
            _points(points),
            _position(worldPosition),
            _rotation(worldRotation)
        {
            assert(_points.size() == points.size());
        }

        // Implementation of the objective function
        int Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
            quaternion rotation(z(3), z(4), z(5), z(6));
            //rotation.normalize();
            vector3 translation(z(0), z(1), z(2));

            // Convert to world coordinates
            rotation = _rotation * rotation;
            translation += _position;

            matrix34 transformationMatrix;
            transformationMatrix << rotation.toRotationMatrix(), translation;


            unsigned int i = 0;
            for(match_point_container::const_iterator pointIterator = _points.cbegin(); pointIterator != _points.cend(); ++pointIterator, ++i) {
                const vector3& detectedPoint = pointIterator->first;
                const vector3& point3D = utils::screen_to_world_coordinates(detectedPoint(0), detectedPoint(1), detectedPoint(2), transformationMatrix); 

                //Supposed Sum of squares: reduce the sum
                fvec(i) = sqrt(get_distance_manhattan(pointIterator->second, point3D)); 
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


    } /* poseOptimisation */
}
