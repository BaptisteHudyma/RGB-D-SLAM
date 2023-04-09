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
Plane::Plane(const Plane_Segment& planeSeg, const utils::Polygon& boundaryPolygon) :
    _parametrization(planeSeg.get_normal(), planeSeg.get_plane_d()),
    _centroid(planeSeg.get_centroid()),
    _pointCloudCovariance(planeSeg.get_point_cloud_covariance()),
    _boundaryPolygon(boundaryPolygon),
    _descriptor(compute_descriptor())
{
    assert(utils::double_equal(planeSeg.get_normal().norm(), 1.0));
    assert(utils::double_equal(get_normal().norm(), 1.0));
}

Plane::Plane(const Plane& plane) :
    _parametrization(plane._parametrization),
    _centroid(plane._centroid),
    _pointCloudCovariance(plane._pointCloudCovariance),
    _boundaryPolygon(plane._boundaryPolygon),
    _descriptor(plane._descriptor)
{
}

vector6 Plane::compute_descriptor(const utils::PlaneCameraCoordinates& parametrization,
                                  const utils::CameraCoordinate& planeCentroid,
                                  const uint pixelCount)
{
    const vector3& normal = parametrization.head(3);
    vector6 descriptor({normal.x(),
                        normal.y(),
                        normal.z(),
                        abs(planeCentroid.x() / pixelCount),
                        abs(planeCentroid.y() / pixelCount),
                        abs(planeCentroid.z() / pixelCount)});
    return descriptor;
};

vector6 Plane::compute_descriptor() const
{
    return compute_descriptor(get_parametrization(), get_centroid(), _boundaryPolygon.area());
}

bool Plane::is_similar(const Plane& plane) const { return is_similar(plane._boundaryPolygon, plane._parametrization); }

bool Plane::is_similar(const utils::Polygon& planePolygon,
                       const utils::PlaneCameraCoordinates& planeParametrization) const
{
    const static double minimumIOUForMatch = Parameters::get_minimum_iou_for_match();
    const static double minimumNormalDotDiff =
            cos(Parameters::get_maximum_plane_normals_angle_for_match() * M_PI / 180.0);
    if (_boundaryPolygon.inter_over_union(planePolygon) < minimumIOUForMatch)
        return false;
    return abs(get_normal().dot(planeParametrization.head(3))) > minimumNormalDotDiff;
}

bool Plane::is_similar(const Cylinder& cylinder) const
{
    // TODO: not implemented
    outputs::log_error("is_similar is not implemented between plane and cylinder");
    return false;
}

double Plane::get_distance(const vector3& point) const { return get_normal().dot(point - _centroid); }

} // namespace rgbd_slam::features::primitives
