#include "levenberg_marquardt_functors.hpp"
#include "matches_containers.hpp"
#include "utils/camera_transformation.hpp"
#include <Eigen/src/Core/util/Meta.h>
#include <cmath>
#include <stdexcept>

namespace rgbd_slam::pose_optimization {

vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat)
{
    // forcing positive "w" to work from 0 to PI
    const quaternion& q = (quat.w() >= 0) ? quat : quaternion(-quat.coeffs());
    const vector3& qv = q.vec();

    const double sinha = qv.norm();
    if (sinha > 0.001)
    {
        const double angle = 2 * atan2(sinha, q.w()); // NOTE: signed
        return (qv * (angle / sinha));
    }
    else
    {
        // if l is too small, its norm can be equal 0 but norm_inf greater than 0
        // probably w is much bigger that vec, use it as length
        return (qv * (2 / q.w())); ////NOTE: signed
    }
}

quaternion get_quaternion_from_scale_axis_coefficients(const vector3& optimizationCoefficients)
{
    const double a = optimizationCoefficients.norm();
    const double ha = a * 0.5;
    const double scale = (a > 0.001) ? (sin(ha) / a) : 0.5;
    quaternion rotation(cos(ha),
                        optimizationCoefficients.x() * scale,
                        optimizationCoefficients.y() * scale,
                        optimizationCoefficients.z() * scale);
    return rotation.normalized();
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
    const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(
            vector3(optimizedParameters(3), optimizedParameters(4), optimizedParameters(5)));
    const vector3 translation(optimizedParameters(0), optimizedParameters(1), optimizedParameters(2));

    // convert to optimization matrix
    const WorldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);

    // Compute projection distances
    int featureScoreIndex = 0; // index of the match being treated
    for (const auto& feature: *_features)
    {
        const auto& distance = feature->get_distance(transformationMatrix);
        const auto partCount = feature->get_feature_part_count();

        assert(static_cast<int>(partCount) == distance.size());

        outputScores.segment(featureScoreIndex, partCount) = distance * feature->get_alpha_reduction();
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
