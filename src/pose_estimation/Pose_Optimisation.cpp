#include "Pose_Optimisation.hpp"

namespace rgbd_slam {
    namespace poseOptimisation {



        Pose_Estimator::Pose_Estimator(unsigned int n, match_point_container& points, const vector3& worldPosition, const quaternion& worldRotation) :
            Levenberg_Marquard_Functor<double>(n, points.size()),
            _points(points),
            _position(worldPosition),
            _rotation(worldRotation)
        {
            assert(_points.size() == points.size());
        }


        double get_distance_manhattan(const vector2& pointA, const vector2& pointB) {
            return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1]);
        }
        double get_distance(const vector2& pointA, const vector2& pointB) {
            return std::sqrt(std::pow(pointA[0] - pointB[0], 2.0) + std::pow(pointA[1] - pointB[1], 2.0));
        }


        // Implementation of the objective function
        int Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
            quaternion rotation(z(3), z(4), z(5), z(6));
            rotation.normalize();
            vector3 translation(z(0), z(1), z(2));
            
            // Convert to world coordinates
            rotation = _rotation * rotation;
            translation = _position + translation;

            const matrix34& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);

            unsigned int i = 0;
            for (const point_pair& pointPair : _points) {
                const vector2& detectedPoint = pointPair.first;

                // convert map point to screen coordinates
                const vector2& screenCoordinates = utils::world_to_screen_coordinates(pointPair.second, transformationMatrix);

                double distance = get_distance(detectedPoint, screenCoordinates);

                //pose error
                //const vector3 dist = detectedPoint - ;
                //fvec(i) = dist.squaredNorm(); 
                //fvec(i) = dist.norm(); 

                //Manhattan
                fvec(i) = distance; 

                //std::cout << distance << " " << detectedPoint.transpose() << " | " << screenCoordinates.transpose() << std::endl;
                ++i;
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
