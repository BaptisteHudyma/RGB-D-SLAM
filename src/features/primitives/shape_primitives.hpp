#ifndef RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP
#define RGBDSLAM_FEATURES_PRIMITIVES_PRIMITIVES_HPP

// cv:Mat
#include "../../types.hpp"
#include "coordinates/point_coordinates.hpp"
#include "cylinder_segment.hpp"
#include "plane_segment.hpp"
#include "coordinates/polygon_coordinates.hpp"
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
    [[nodiscard]] bool is_similar(const Cylinder& prim) const noexcept;

    /**
     * \brief Get the distance of a point to the surface of the cylinder
     *
     * \return The signed distance of the point to the surface, 0 if the point is on the surface, and < 0 if the point
     * is inside the cylinder
     */
    [[nodiscard]] double get_distance(const vector3& point) const noexcept;

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
    Plane(const Plane_Segment& planeSeg, const CameraPolygon& boundaryPolygon);

    Plane(const Plane& plane);

    /**
     * \brief Get the similarity of two planes, based on normal direction
     * \param[in] prim Another primitive to compare to
     * \return A true if those shapes are similar
     */
    [[nodiscard]] bool is_normal_similar(const Plane& prim) const noexcept;
    [[nodiscard]] bool is_normal_similar(const PlaneCameraCoordinates& planeParametrization) const noexcept;

    /**
     * \brief Check that the distance between the two plane d component is less than a threshold
     * \param[in] prim Another primitive to compare to
     */
    [[nodiscard]] bool is_distance_similar(const Plane& prim) const noexcept;
    [[nodiscard]] bool is_distance_similar(const PlaneCameraCoordinates& planeParametrization) const noexcept;

    [[nodiscard]] bool is_similar(const Cylinder& prim) const noexcept;

    [[nodiscard]] vector3 get_normal() const noexcept { return _parametrization.get_normal(); };
    [[nodiscard]] double get_d() const noexcept { return _parametrization.get_d(); };
    [[nodiscard]] PlaneCameraCoordinates get_parametrization() const noexcept { return _parametrization; };
    [[nodiscard]] CameraCoordinate get_center() const noexcept { return get_normal() * (-get_d()); };
    [[nodiscard]] matrix33 get_point_cloud_covariance() const noexcept { return _pointCloudCovariance; };

    [[nodiscard]] CameraPolygon get_boundary_polygon() const noexcept { return _boundaryPolygon; };

    ~Plane() = default;

  private:
    /**
     * Return the distance of this primitive to a point
     */
    [[nodiscard]] double get_distance(const vector3& point) const noexcept;

    PlaneCameraCoordinates _parametrization; // infinite plane representation
    matrix33 _pointCloudCovariance;          // the covariance of point cloud that this plane is fitted from
    const CameraPolygon _boundaryPolygon;

    // remove copy functions
    Plane() = delete;
    Plane& operator=(const Plane&) = delete;
};

// types for detected primitives
using cylinder_container = std::vector<Cylinder>;
using plane_container = std::vector<Plane>;

} // namespace rgbd_slam::features::primitives

#endif
