#include "levenberg_marquard_functors.hpp"
#include "../utils/camera_transformation.hpp"
#include "coordinates.hpp"
#include <Eigen/src/Core/util/Meta.h>
#include <cmath>

namespace rgbd_slam {
namespace pose_optimization {

vector3 get_scaled_axis_coefficients_from_quaternion(const quaternion& quat)
{
    // forcing positive "w" to work from 0 to PI
    const quaternion& q = (quat.w() >= 0) ? quat : quaternion(-quat.coeffs());
    const vector3& qv = q.vec();

    const double sinha = qv.norm();
    if (sinha > 0)
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
    const double scale = (a > 0) ? (sin(ha) / a) : 0.5;
    quaternion rotation(cos(ha),
                        optimizationCoefficients.x() * scale,
                        optimizationCoefficients.y() * scale,
                        optimizationCoefficients.z() * scale);
    rotation.normalize();
    return rotation;
}

/**
 * GLOBAL POSE ESTIMATOR members
 */

Global_Pose_Estimator::Global_Pose_Estimator(const matches_containers::match_point_container& points,
                                             const matches_containers::match_plane_container& planes) :
    Levenberg_Marquardt_Functor<double>(6, points.size() * 2 + planes.size() * 3),
    // TODO: optimize below: we copy the containers instead of referencing them
    _points(points),
    _planes(planes)
{
    assert(not _points.empty() or not _planes.empty());

    _dividers.reserve(_points.size() + _planes.size());
    for (const matches_containers::PointMatch& match: _points)
    {
        _dividers.emplace(_dividers.end(), 1.0 / sqrt(match._projectedWorldfeatureCovariance.x()));
        _dividers.emplace(_dividers.end(), 1.0 / sqrt(match._projectedWorldfeatureCovariance.y()));
    }
    for (const matches_containers::PlaneMatch& match: _planes)
    {
        // TODO: replace with a true covariance value
        _dividers.emplace(_dividers.end(), 0.01);
        _dividers.emplace(_dividers.end(), 0.01);
        _dividers.emplace(_dividers.end(), 0.01);
    }
}

// Implementation of the objective function
int Global_Pose_Estimator::operator()(const vectorxd& optimizedParameters, vectorxd& outputScores) const
{
    assert(not _points.empty() or not _planes.empty());
    assert(optimizedParameters.size() == 6);
    assert(Eigen::Index(_dividers.size()) == outputScores.size());
    assert(static_cast<size_t>(outputScores.size()) == (_points.size() * 2 + _planes.size() * 3));

    // Get the new estimated pose
    const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(
            vector3(optimizedParameters(3), optimizedParameters(4), optimizedParameters(5)));
    const vector3 translation(optimizedParameters(0), optimizedParameters(1), optimizedParameters(2));

    const worldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
    Eigen::Index pointIndex = 0; // index of the match being treated
    // Compute retroprojection distances
    for (const matches_containers::PointMatch& match: _points)
    {
        // Compute retroprojected distance
        const vector2& distance =
                match._worldFeature.get_signed_distance_2D(match._screenFeature, transformationMatrix);

        outputScores(pointIndex) = distance.x() * _dividers[pointIndex];
        ++pointIndex;
        outputScores(pointIndex) = distance.y() * _dividers[pointIndex];
        ++pointIndex;
    }

    // add plane optimization vectors
    const planeWorldToCameraMatrix& planeTransformationMatrix =
            utils::compute_plane_world_to_camera_matrix(transformationMatrix);
    for (const matches_containers::PlaneMatch& match: _planes)
    {
        const vector3& planeProjectionError =
                match._worldFeature.get_reduced_signed_distance(match._screenFeature, planeTransformationMatrix);

        outputScores(pointIndex) = planeProjectionError.x() * _dividers[pointIndex];
        ++pointIndex;
        outputScores(pointIndex) = planeProjectionError.y() * _dividers[pointIndex];
        ++pointIndex;
        outputScores(pointIndex) = planeProjectionError.z() * _dividers[pointIndex];
        ++pointIndex;
    }
    return 0;
}

/**
 * \brief Return a string corresponding to the end status of the optimization
 */
const std::string get_human_readable_end_message(Eigen::LevenbergMarquardtSpace::Status status)
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

} // namespace pose_optimization
} // namespace rgbd_slam
