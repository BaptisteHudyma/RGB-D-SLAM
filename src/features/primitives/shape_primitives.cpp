#include "shape_primitives.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "covariances.hpp"
#include "cylinder_segment.hpp"
#include "distance_utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/VectorBlock.h>

namespace rgbd_slam::features::primitives {

/*
 *
 *      CYLINDER
 *
 */
Cylinder::Cylinder(const Cylinder_Segment& cylinderSeg) : _radius(0)
{
    for (uint i = 0; i < cylinderSeg.get_segment_count(); ++i)
    {
        _radius += cylinderSeg.get_radius(i);
    }
    _radius /= cylinderSeg.get_segment_count();
    _normal = cylinderSeg.get_normal();
}

Cylinder::Cylinder(const Cylinder& cylinder) : _normal(cylinder._normal), _radius(cylinder._radius) {}

bool Cylinder::is_similar(const Cylinder& cylinder) const noexcept
{
    static const double minimumNormalDotDiff =
            abs(cos(parameters::matching::maximumAngleForPlaneMatch_d * M_PI / 180.0));
    return std::abs(_normal.dot(cylinder._normal)) > minimumNormalDotDiff;
}

double Cylinder::get_distance(const vector3& point) const noexcept
{
    // TODO implement
    outputs::log_error("Error: get_point_distance is not implemented for Cylinder objects");
    return 0;
}

/*
 *
 *        PLANE
 *
 */
Plane::Plane(const Plane_Segment& planeSeg, const CameraPolygon& boundaryPolygon) :
    _parametrization(planeSeg.get_normal(), planeSeg.get_plane_d()),
    _pointCloudCovariance(planeSeg.get_point_cloud_covariance()),
    _boundaryPolygon(boundaryPolygon)
{
    assert(utils::double_equal(get_normal().norm(), 1.0));
    assert(utils::is_covariance_valid(_pointCloudCovariance));
    assert(_boundaryPolygon.boundary_length() >= 3);
}

Plane::Plane(const Plane& plane) :
    _parametrization(plane._parametrization),
    _pointCloudCovariance(plane._pointCloudCovariance),
    _boundaryPolygon(plane._boundaryPolygon)
{
    assert(utils::double_equal(get_normal().norm(), 1.0));
    assert(utils::is_covariance_valid(_pointCloudCovariance));
    assert(_boundaryPolygon.boundary_length() >= 3);
}

bool Plane::is_normal_similar(const Plane& plane) const noexcept { return is_normal_similar(plane._parametrization); }

bool Plane::is_normal_similar(const PlaneCameraCoordinates& planeParametrization) const noexcept
{
    static const double minimumNormalDotDiff =
            abs(cos(parameters::matching::maximumAngleForPlaneMatch_d * M_PI / 180.0));
    return abs(_parametrization.get_cos_angle(planeParametrization)) > minimumNormalDotDiff;
}

bool Plane::is_distance_similar(const Plane& plane) const noexcept
{
    return is_distance_similar(plane._parametrization);
}

bool Plane::is_distance_similar(const PlaneCameraCoordinates& planeParametrization) const noexcept
{
    constexpr double maximumPlaneMatchDistance = parameters::matching::maximumDistanceForPlaneMatch_mm;
    return abs(_parametrization.get_d() - planeParametrization.get_d()) < maximumPlaneMatchDistance;
}

bool Plane::is_similar(const Cylinder& cylinder) const noexcept
{
    // TODO: not implemented
    outputs::log_error("is_similar is not implemented between plane and cylinder");
    return false;
}

double Plane::get_distance(const vector3& point) const noexcept
{
    return get_parametrization().get_point_distance(point);
}

} // namespace rgbd_slam::features::primitives
