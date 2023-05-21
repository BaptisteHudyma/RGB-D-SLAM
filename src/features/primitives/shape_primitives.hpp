#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP

// cv:Mat
#include "../../types.hpp"
#include "../../utils/coordinates.hpp"
#include "cylinder_segment.hpp"
#include "plane_segment.hpp"
#include "polygon.hpp"
#include <opencv2/opencv.hpp>

namespace rgbd_slam::features::primitives {

/**
 * \brief A base class used to compute the tracking analysis.
 * It is a pure virtual class.
 */
class IPrimitive
{
  public:
    IPrimitive() = default;

    bool can_add_to_map() const
    {
        // TODO
        return true;
    }

  private:
    // remove copy functions
    IPrimitive& operator=(const IPrimitive&) = delete;
};

/**
 * \brief Handles cylinder primitives.
 */
class Cylinder : public IPrimitive
{
  public:
    /**
     * \brief Construct a cylinder object
     *
     * \param[in] cylinderSeg Cylinder segment to copy
     */
    Cylinder(const Cylinder_Segment& cylinderSeg);
    Cylinder(const Cylinder& cylinder);

    /**
     * \brief Get the similarity of two cylinders, based on normal direction and radius
     *
     * \param[in] prim Another cylinder to compare to
     *
     * \return A double between 0 and 1, with 1 indicating identical cylinders
     */
    bool is_similar(const Cylinder& prim) const;

    /**
     * \brief Get the distance of a point to the surface of the cylinder
     *
     * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point
     * is inside the cylinder
     */
    double get_distance(const vector3& point) const;

    vector3 _normal;
    double _radius;

    ~Cylinder() = default;

  private:
    // remove copy functions
    Cylinder() = delete;
    Cylinder& operator=(const Cylinder&) = delete;
};

/**
 * \brief Handles planes.
 */
class Plane : public IPrimitive
{
  public:
    /**
     * \brief Construct a plane object
     *
     * \param[in] planeSeg Plane to copy
     * \param[in] boundaryPolygon Polygon describing the boundary of the plane, in plane coordiates
     */
    Plane(const Plane_Segment& planeSeg, const utils::CameraPolygon& boundaryPolygon);

    Plane(const Plane& plane);

    /**
     * \brief Get the similarity of two planes, based on normal direction
     * \param[in] prim Another primitive to compare to
     * \return A true if those shapes are similar
     */
    bool is_normal_similar(const Plane& prim) const;
    bool is_normal_similar(const utils::PlaneCameraCoordinates& planeParametrization) const;

    /**
     * \brief Check that the distance between the two plane d component is less than a threshold
     * \param[in] prim Another primitive to compare to
     */
    bool is_distance_similar(const Plane& prim) const;
    bool is_distance_similar(const utils::PlaneCameraCoordinates& planeParametrization) const;

    bool is_similar(const Cylinder& prim) const;

    vector3 get_normal() const { return _parametrization.get_normal(); };
    double get_d() const { return _parametrization.get_d(); };
    utils::PlaneCameraCoordinates get_parametrization() const { return _parametrization; };
    utils::CameraCoordinate get_center() const { return get_normal() * (-get_d()); };
    matrix33 get_point_cloud_covariance() const { return _pointCloudCovariance; };

    utils::CameraPolygon get_boundary_polygon() const { return _boundaryPolygon; };

    ~Plane() = default;

  private:
    /**
     * Return the distance of this primitive to a point
     */
    double get_distance(const vector3& point) const;

    utils::PlaneCameraCoordinates _parametrization; // infinite plane representation
    matrix33 _pointCloudCovariance;                 // the covariance of point cloud that this plane is fitted from
    const utils::CameraPolygon _boundaryPolygon;

    // remove copy functions
    Plane() = delete;
    Plane& operator=(const Plane&) = delete;
};

// types for detected primitives
using cylinder_container = std::vector<Cylinder>;
using plane_container = std::vector<Plane>;

} // namespace rgbd_slam::features::primitives

#endif
