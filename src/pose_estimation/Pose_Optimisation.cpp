#include "Pose_Optimisation.hpp"

namespace poseOptimisation {



    Pose_Estimator::Pose_Estimator(unsigned int n, const matched_point_container& points) :
        Levenberg_Marquard_Functor<double>(n, points.size()) 
    {
        //copy points
        _points = points;
    }

    // Implementation of the objective function
    int Pose_Estimator::operator()(const Eigen::VectorXd& z, Eigen::VectorXd& fvec) const {
        quaternion rotation(z(3), z(4), z(5), z(6));
        rotation.normalize();

        const matrix33& rotationMatrix = rotation.toRotationMatrix();
        const vector3 translation(z(0), z(1), z(2));

        unsigned int i = 0;
        for(const point_pair& pointPair : _points) {
            const vector3& mapPoint = pointPair.first;
            const vector3& detectedPoint = pointPair.second;

            //pose error
            const vector3 dist = detectedPoint - ( rotationMatrix * mapPoint + translation );
            //fvec(i) = dist.squaredNorm(); 
            //fvec(i) = dist.norm(); 

            //Manhattan
            fvec(i) = abs(dist[0]) + abs(dist[1]) + abs(dist[2]);
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
