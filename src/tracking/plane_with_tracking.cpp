#include "plane_with_tracking.hpp"

namespace rgbd_slam::tracking {

/**
 *  Plane
 */

Plane::Plane()
{
    _covariance.setZero();

    build_kalman_filter();
}

double Plane::track(const CameraToWorldMatrix& cameraToWorld,
                    const features::primitives::Plane& matchedFeature,
                    const utils::PlaneWorldCoordinates& newDetectionParameters,
                    const matrix44& newDetectionCovariance)
{
    assert(_kalmanFilter != nullptr);
    if (not utils::is_covariance_valid(newDetectionCovariance))
    {
        outputs::log_error("newDetectionCovariance is an invalid covariance matrix");
        return false;
    }
    if (not utils::is_covariance_valid(_covariance))
    {
        outputs::log_error("_covariance is an invalid covariance matrix");
        exit(-1);
    }

    const std::pair<vector4, matrix44>& res = _kalmanFilter->get_new_state(_parametrization.get_parametrization(),
                                                                           _covariance,
                                                                           newDetectionParameters.get_parametrization(),
                                                                           newDetectionCovariance);
    const utils::PlaneWorldCoordinates newEstimatedParameters(res.first);
    const matrix44& newEstimatedCovariance = res.second;
    const double score = (_parametrization.get_parametrization() - newEstimatedParameters.get_parametrization()).norm();

    // covariance update
    _covariance = newEstimatedCovariance;

    // parameters update
    _parametrization = utils::PlaneWorldCoordinates(newEstimatedParameters);

    // merge the boundary polygon (after optimization) with the observed polygon
    if (not update_boundary_polygon(cameraToWorld, matchedFeature.get_boundary_polygon()))
    {
        throw std::logic_error("Could not merge the polygons");
    }

    // static sanity checks
    assert(utils::double_equal(_parametrization.get_normal().norm(), 1.0));
    assert(not _covariance.hasNaN());
    assert(utils::is_covariance_valid(_covariance));
    assert(not _parametrization.hasNaN());
    return score;
}

bool Plane::update_boundary_polygon(const CameraToWorldMatrix& cameraToWorld,
                                    const utils::CameraPolygon& detectedPolygon) noexcept
{
    // correct the projection of the boundary polygon to correspond to the parametrization
    const vector3& worldPolygonNormal = _parametrization.get_normal();
    const vector3& worldPolygonCenter = _parametrization.get_center();
    _boundaryPolygon = _boundaryPolygon.project(worldPolygonNormal, worldPolygonCenter);
    if (not _boundaryPolygon.get_center().isApprox(worldPolygonCenter))
    {
        return false;
    }

    // convert detected polygon to world space, it is supposed to be aligned with the world polygon
    const utils::WorldPolygon& projectedPolygon = detectedPolygon.to_world_space(cameraToWorld);

    // merge the projected observed polygon with optimized parameters with the current world polygon
    _boundaryPolygon.merge(projectedPolygon);
    return true;
}

void Plane::build_kalman_filter() noexcept
{
    if (_kalmanFilter == nullptr)
    {
        const matrix44 systemDynamics = matrix44::Identity(); // planes are not supposed to move, so no dynamics
        const matrix44 outputMatrix = matrix44::Identity();   // we need all positions

        const double parametersProcessNoise = 0.000001; // TODO set in parameters
        const matrix44 processNoiseCovariance =
                matrix44::Identity() * parametersProcessNoise; // Process noise covariance

        _kalmanFilter = std::make_unique<tracking::SharedKalmanFilter<4, 4>>(
                systemDynamics, outputMatrix, processNoiseCovariance);
    }
}

} // namespace rgbd_slam::tracking