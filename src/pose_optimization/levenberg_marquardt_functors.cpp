#include "levenberg_marquardt_functors.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include <Eigen/src/Core/util/Meta.h>
#include <cmath>
#include <stdexcept>

namespace rgbd_slam::pose_optimization {

vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat)
{
    return vector3(quat.x(), quat.y(), quat.z());
}

quaternion get_quaternion_from_scale_axis_coefficients(const vector3& optimizationCoefficients)
{
    const double w = sqrt(1.0 - SQR(optimizationCoefficients.x()) - SQR(optimizationCoefficients.y()) -
                          SQR(optimizationCoefficients.z()));
    return quaternion(w, optimizationCoefficients.x(), optimizationCoefficients.y(), optimizationCoefficients.z());
}

vector6 get_optimization_coefficient_from_pose(const utils::PoseBase& pose)
{
    vector6 coeffs;
    coeffs.head<3>() = pose.get_position();
    coeffs.tail<3>() = get_scaled_axis_coefficients_from_quaternion(pose.get_orientation_quaternion());
    return coeffs;
}

utils::PoseBase get_pose_from_optimization_coeffiencients(const vector6& optimizationCoefficients)
{
    return utils::PoseBase(optimizationCoefficients.head<3>(),
                           get_quaternion_from_scale_axis_coefficients(optimizationCoefficients.tail<3>()));
}

/**
 * GLOBAL POSE ESTIMATOR members
 */
Global_Pose_Estimator::Global_Pose_Estimator(const size_t optimizationParts,
                                             const matches_containers::match_container* const features) :
    Levenberg_Marquardt_Functor<double>(6, optimizationParts),
    _optimizationParts(optimizationParts),
    _features(features)
{
    // parameter checks
    if (_features == nullptr or _features->empty() or _optimizationParts == 0)
    {
        throw std::logic_error("cannot optimize on empty vector");
    }

    // sanity check
    size_t featureParts = 0;
    for (const auto& feature: *_features)
    {
        featureParts += feature->get_feature_part_count();
    }
    if (featureParts != optimizationParts)
    {
        throw std::logic_error("optimization vector do not match the given vector size");
    }
}

// Implementation of the objective function
int Global_Pose_Estimator::operator()(const Eigen::Vector<double, 6>& optimizedParameters, vectorxd& outputScores) const
{
    // sanity checks
    assert(_features != nullptr);
    assert(not _features->empty());
    assert(static_cast<size_t>(outputScores.size()) == _optimizationParts);

    // Get the new estimated pose
    const utils::PoseBase& pose = get_pose_from_optimization_coeffiencients(optimizedParameters);

    // convert to optimization matrix
    const WorldToCameraMatrix& transformationMatrix =
            utils::compute_world_to_camera_transform(pose.get_orientation_quaternion(), pose.get_position());

    // Compute projection distances
    int featureScoreIndex = 0; // index of the match being treated
    for (const auto& feature: *_features)
    {
        const auto& distance = feature->get_distance(transformationMatrix);
        const auto partCount = feature->get_feature_part_count();

        assert(static_cast<int>(partCount) == distance.size());

        outputScores.segment(featureScoreIndex, partCount) =
                distance * feature->get_alpha_reduction() / static_cast<double>(partCount);
        featureScoreIndex += static_cast<int>(partCount);
    }
    return 0;
}

/**
 * \brief Return a string corresponding to the end status of the optimization
 */
std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status) noexcept
{
    switch (status)
    {
        case Eigen::LevenbergMarquardtSpace::Status::NotStarted:
            return "not started";
        case Eigen::LevenbergMarquardtSpace::Status::Running:
            return "running";
        case Eigen::LevenbergMarquardtSpace::Status::ImproperInputParameters:
            return "improper input parameters";
        case Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall:
            return "relative reduction too small";
        case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorTooSmall:
            return "relative error too small";
        case Eigen::LevenbergMarquardtSpace::Status::RelativeErrorAndReductionTooSmall:
            return "relative error and reduction too small";
        case Eigen::LevenbergMarquardtSpace::Status::CosinusTooSmall:
            return "cosinus too small";
        case Eigen::LevenbergMarquardtSpace::Status::TooManyFunctionEvaluation:
            return "too many function evaluation";
        case Eigen::LevenbergMarquardtSpace::Status::FtolTooSmall:
            return "xtol too small";
        case Eigen::LevenbergMarquardtSpace::Status::XtolTooSmall:
            return "ftol too small";
        case Eigen::LevenbergMarquardtSpace::Status::GtolTooSmall:
            return "gtol too small";
        case Eigen::LevenbergMarquardtSpace::Status::UserAsked:
            return "user asked";
        default:
            return "error: empty message";
    }
    return std::string("");
}

} // namespace rgbd_slam::pose_optimization
