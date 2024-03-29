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
constexpr uint scoreCountPerPoints = 2;
constexpr uint scoreCountPer2DPoints = 2;
constexpr uint scoreCountPerPlanes = 3;

Global_Pose_Estimator::Global_Pose_Estimator(const matches_containers::match_point2D_container* const points2d,
                                             const matches_containers::match_point_container* const points,
                                             const matches_containers::match_plane_container* const planes) :
    Levenberg_Marquardt_Functor<double>(6,
                                        points2d->size() * scoreCountPer2DPoints +
                                                points->size() * scoreCountPerPoints +
                                                planes->size() * scoreCountPerPlanes),
    _points2d(points2d),
    _points(points),
    _planes(planes)
{
    assert(_points2d != nullptr and _points != nullptr and _planes != nullptr);
    if (_points2d->empty() and _points->empty() and _planes->empty())
    {
        throw std::logic_error("cannot optimize on empty vectors");
    }
}

// Implementation of the objective function
int Global_Pose_Estimator::operator()(const Eigen::Vector<double, 6>& optimizedParameters, vectorxd& outputScores) const
{
    assert(_points2d != nullptr and _points != nullptr and _planes != nullptr);
    assert(not _points2d->empty() or not _points->empty() or not _planes->empty());
    assert(static_cast<size_t>(outputScores.size()) ==
           (_points2d->size() * scoreCountPer2DPoints + _points->size() * scoreCountPerPoints +
            _planes->size() * scoreCountPerPlanes));

    // Get the new estimated pose
    const quaternion& rotation = get_quaternion_from_scale_axis_coefficients(
            vector3(optimizedParameters(3), optimizedParameters(4), optimizedParameters(5)));
    const vector3 translation(optimizedParameters(0), optimizedParameters(1), optimizedParameters(2));

    const WorldToCameraMatrix& transformationMatrix = utils::compute_world_to_camera_transform(rotation, translation);
    int featureScoreIndex = 0; // index of the match being treated

    // Compute retroprojection distances
    static constexpr double point2dAlphaReduction = 0.3; // multiplier for points parameters in the equation
    for (const matches_containers::PointMatch2D& match: *_points2d)
    {
        const vector2& distance = match._worldFeature.compute_signed_screen_distance(
                match._screenFeature, match._worldFeatureCovariance.diagonal()(3), transformationMatrix);

        outputScores.segment<scoreCountPer2DPoints>(featureScoreIndex) = distance * point2dAlphaReduction;
        featureScoreIndex += scoreCountPer2DPoints;
    }

    static constexpr double pointAlphaReduction = 1.0; // multiplier for points parameters in the equation
    for (const matches_containers::PointMatch& match: *_points)
    {
        // Compute retroprojected distance
        const Eigen::Vector<double, scoreCountPerPoints>& distance =
                match._worldFeature.get_signed_distance_2D_px(match._screenFeature, transformationMatrix);

        outputScores.segment<scoreCountPerPoints>(featureScoreIndex) = distance * pointAlphaReduction;
        featureScoreIndex += scoreCountPerPoints;
    }

    // add plane optimization vectors
    static constexpr double planeAlphaReduction = 1.0; // multiplier for plane parameters in the equation
    const PlaneWorldToCameraMatrix& planeTransformationMatrix =
            utils::compute_plane_world_to_camera_matrix(transformationMatrix);
    for (const matches_containers::PlaneMatch& match: *_planes)
    {
        // TODO remove d from optimization, replace with boundary optimization
        const Eigen::Vector<double, scoreCountPerPlanes>& planeProjectionError =
                match._worldFeature.get_reduced_signed_distance(match._screenFeature, planeTransformationMatrix);

        outputScores.segment<scoreCountPerPlanes>(featureScoreIndex) = planeProjectionError * planeAlphaReduction;
        featureScoreIndex += scoreCountPerPlanes;
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
