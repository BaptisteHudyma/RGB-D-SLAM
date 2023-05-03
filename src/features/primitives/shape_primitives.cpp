#include "shape_primitives.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "covariances.hpp"
#include "cylinder_segment.hpp"
#include "distance_utils.hpp"
#include "polygon.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/VectorBlock.h>

namespace rgbd_slam::features::primitives {

/*
 *
 *      PRIMITIVE
 *
 */

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

bool Cylinder::is_similar(const Cylinder& cylinder) const
{
    const static double minimumNormalDotDiff = Parameters::get_maximum_plane_normals_angle_for_match();
    return std::abs(_normal.dot(cylinder._normal)) > minimumNormalDotDiff;
}

double Cylinder::get_distance(const vector3& point) const
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
Plane::Plane(const Plane_Segment& planeSeg, const utils::CameraPolygon& boundaryPolygon) :
    _parametrization(planeSeg.get_normal(), planeSeg.get_plane_d()),
    _centroid(planeSeg.get_centroid()),
    _pointCloudCovariance(planeSeg.get_point_cloud_covariance()),
    _boundaryPolygon(boundaryPolygon)
{
    assert(utils::double_equal(get_normal().norm(), 1.0));
    assert(utils::is_covariance_valid(_pointCloudCovariance));
    assert(_boundaryPolygon.boundary_lentgh() >= 3);
}

Plane::Plane(const Plane& plane) :
    _parametrization(plane._parametrization),
    _centroid(plane._centroid),
    _pointCloudCovariance(plane._pointCloudCovariance),
    _boundaryPolygon(plane._boundaryPolygon)
{
    assert(utils::double_equal(get_normal().norm(), 1.0));
    assert(utils::is_covariance_valid(_pointCloudCovariance));
    assert(_boundaryPolygon.boundary_lentgh() >= 3);
}

bool Plane::is_normal_similar(const Plane& plane) const { return is_normal_similar(plane._parametrization); }

bool Plane::is_normal_similar(const utils::PlaneCameraCoordinates& planeParametrization) const
{
    const static double minimumNormalDotDiff =
            cos(Parameters::get_maximum_plane_normals_angle_for_match() * M_PI / 180.0);
    return abs(get_normal().dot(planeParametrization.get_normal())) > minimumNormalDotDiff;
}

bool Plane::is_distance_similar(const Plane& plane) const { return is_distance_similar(plane._parametrization); }

bool Plane::is_distance_similar(const utils::PlaneCameraCoordinates& planeParametrization) const
{
    const static double maximumPlanMatchDistance = Parameters::get_maximum_plane_distance_for_match();
    return abs(_parametrization.get_d() - planeParametrization.get_d()) < maximumPlanMatchDistance;
}

bool Plane::is_similar(const Cylinder& cylinder) const
{
    // TODO: not implemented
    outputs::log_error("is_similar is not implemented between plane and cylinder");
    return false;
}

double Plane::get_distance(const vector3& point) const { return get_normal().dot(point - _centroid); }

} // namespace rgbd_slam::features::primitives
