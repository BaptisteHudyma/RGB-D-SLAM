#include "levenberg_marquardt_functors.hpp"
#include "covariances.hpp"
#include "logger.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "types.hpp"
#include "utils/camera_transformation.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/ForwardDeclarations.h>
#include <Eigen/src/Core/util/Meta.h>
#include <Eigen/Sparse>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <stdexcept>
#include <unsupported/Eigen/src/NonLinearOptimization/LevenbergMarquardt.h>

namespace rgbd_slam::pose_optimization {

vector3 get_optimization_coefficients_from_quaternion(const quaternion& quat)
{
    ///
    /// Ref: On quaternion based parameterization of orientation in computer vision and robotics
    ///
    /// by: G. Terzakis, P. Culverhouse, G. Bugmann, S. Sharma and R. Sutton
    ///

    // The max is an addition on my part to resolve the occasional case where we have a pure rotation that gives quat.z
    // at -1
    const double divider = 1.0 / std::max(1.0 + quat.z(), 0.001);
    return vector3(quat.w() * divider, quat.x() * divider, quat.y() * divider);
}

quaternion get_quaternion_from_optimization_coefficients(const vector3& optimizationCoefficients)
{
    const double alpha =
            SQR(optimizationCoefficients.x()) + SQR(optimizationCoefficients.y()) + SQR(optimizationCoefficients.z());
    const double divider = 1.0 / (alpha + 1);
    return quaternion(2.0 * optimizationCoefficients.x() * divider,
                      2.0 * optimizationCoefficients.y() * divider,
                      2.0 * optimizationCoefficients.z() * divider,
                      (1 - alpha) * divider);
}

Eigen::Matrix<double, 4, 3> get_quaternion_from_optimization_coefficients_jacobian(const vector3& optCoeff)
{
    Eigen::Matrix<double, 4, 3> jacobian;
    jacobian.setZero();

    const double theta6 = SQR(optCoeff.x()) + SQR(optCoeff.y()) + SQR(optCoeff.z());
    const double theta5 = theta6 + 1;
    const double theta1 = theta6 - 1;
    const double theta2 = -(4.0 * optCoeff.y() * optCoeff.z()) / SQR(theta5);
    const double theta3 = -(4.0 * optCoeff.x() * optCoeff.z()) / SQR(theta5);
    const double theta4 = -(4.0 * optCoeff.x() * optCoeff.y()) / SQR(theta5);

    const double multiplierDiag = -4.0 / SQR(theta5);

    jacobian(0, 0) = 2.0 / theta5 - SQR(optCoeff.x()) * multiplierDiag;
    jacobian(0, 1) = theta4;
    jacobian(0, 2) = theta3;

    jacobian(1, 0) = theta4;
    jacobian(1, 1) = 2.0 / theta5 - SQR(optCoeff.y()) * multiplierDiag;
    jacobian(1, 2) = theta2;

    jacobian(2, 0) = theta3;
    jacobian(2, 1) = theta2;
    jacobian(2, 2) = 2.0 / theta5 - SQR(optCoeff.z()) * multiplierDiag;

    const double multiA = 2.0 * theta1 / SQR(theta5);
    const double multiB = -2.0 / theta5;
    jacobian(3, 0) = optCoeff.x() * multiA + optCoeff.x() * multiB;
    jacobian(3, 1) = optCoeff.y() * multiA + optCoeff.y() * multiB;
    jacobian(3, 2) = optCoeff.z() * multiA + optCoeff.z() * multiB;

    return jacobian;
}

vector6 get_optimization_coefficient_from_pose(const utils::PoseBase& pose)
{
    vector6 coeffs;
    coeffs.head<3>() = pose.get_position();
    coeffs.tail<3>() = get_optimization_coefficients_from_quaternion(pose.get_orientation_quaternion());
    return coeffs;
}

utils::PoseBase get_pose_from_optimization_coefficients(const vector6& optimizationCoefficients)
{
    return utils::PoseBase(optimizationCoefficients.head<3>(),
                           get_quaternion_from_optimization_coefficients(optimizationCoefficients.tail<3>()));
}

utils::PoseBase get_pose_from_optimization_coefficients(const vector6& optimizationCoefficients,
                                                        Eigen::Matrix<double, 7, 6>& jacobian)
{
    jacobian.setZero();
    jacobian.block<3, 3>(0, 0) = matrix33::Identity();

    jacobian.block<4, 3>(3, 3) =
            get_quaternion_from_optimization_coefficients_jacobian(optimizationCoefficients.tail<3>());

    return get_pose_from_optimization_coefficients(optimizationCoefficients);
}

/**
 * RELATIVE POSE ESTIMATOR members
 */

Relative_Pose_Estimator::Relative_Pose_Estimator(const vector6& startParameters,
                                                 const size_t optimizationParts,
                                                 const matches_containers::match_container* const features) :
    Levenberg_Marquardt_Functor<double>(6, optimizationParts + feedbackOptimizationPart),
    _startParameters(startParameters),
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

vectorxd get_score(matches_containers::feat_ptr feature, vectorxd distance)
{
    const auto partCount = feature->get_feature_part_count();
    return vectorxd(partCount).setConstant(feature->get_alpha_reduction() / static_cast<double>(partCount));
}

// Implementation of the objective function
int Relative_Pose_Estimator::operator()(const vector6& optimizedParameters, vectorxd& outputScores) const
{
    // sanity checks
    assert(_features != nullptr);
    assert(not _features->empty());
    assert(static_cast<size_t>(outputScores.size()) == (_optimizationParts + 2));

    // Get the new estimated pose
    const utils::PoseBase& pose = get_pose_from_optimization_coefficients(optimizedParameters);

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
                distance.array().colwise() * get_score(feature, distance).array();
        featureScoreIndex += static_cast<int>(partCount);
    }

    // feedback, to garanty to prevent the system from going to infinity when the features are all scaless parameters
    // (eg: inverse depth points)
    const auto res = (_startParameters - optimizedParameters);
    outputScores.segment(featureScoreIndex, feedbackOptimizationPart) =
            vector2(res.head<3>().norm(), res.tail<3>().norm()) / outputScores.size();
    return 0;
}

bool Relative_Pose_Estimator::get_input_covariance(const matches_containers::match_container& features,
                                                   const utils::PoseBase& optimizedPose,
                                                   const matrixd& jacobian,
                                                   matrix66& inputCovariance) noexcept
{
    // Ignore the feedback part of the jacobian
    if (jacobian.hasNaN() or not jacobian.allFinite() or jacobian.isConstant(0.0))
    {
        outputs::log_error("Input jacobian is all invalid");
        return false;
    }

    // convert to optimization matrix
    const WorldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(
            optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

    matrixd distancesCovariances(jacobian.rows(), jacobian.rows());
    distancesCovariances.setZero();

    matrixd newJac = jacobian;

    int featureScoreIndex = 0; // index of the match being treated
    for (const auto& feature: features)
    {
        const auto partCount = feature->get_feature_part_count();

        // remove the score applied to this part of the jacobian
        auto part = newJac.block(featureScoreIndex, 0, partCount, 6);
        part = (part.array().colwise() / get_score(feature, vectorxd()).array()).eval();

        const matrixd& distCov = feature->get_distance_covariance(transformationMatrix).selfadjointView<Eigen::Lower>();
        // set covariance of this feature in world space
        distancesCovariances.block(featureScoreIndex, featureScoreIndex, partCount, partCount) = distCov;

        std::string failureReason;
        if (not utils::is_covariance_valid(distCov, failureReason))
        {
            outputs::log_warning("a distance covariance is invalid: " + failureReason + ": feature type " +
                                 to_string(feature->get_feature_type()));
        }
        featureScoreIndex += static_cast<int>(partCount);
    }
    std::string failureReason;
    if (not utils::is_covariance_valid(distancesCovariances, failureReason))
    {
        outputs::log_error("Distances covariances are invalid: " + failureReason);
        return false;
    }

    // Retrieve the covariance of the parameters
    // get the jacobian of the transformation
    matrix66 cov =
            pseudoInverse(newJac.transpose() * distancesCovariances.inverse().selfadjointView<Eigen::Lower>() * newJac)
                    .selfadjointView<Eigen::Lower>(); // force symetrical
    cov.diagonal().head<3>() += vector3::Constant(1e-6);
    if (utils::is_covariance_valid(cov, failureReason))
    {
        inputCovariance = cov;
        return true;
    }
    outputs::log("Distances covariances after transfo is invalid: " + failureReason);
    return false;
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
