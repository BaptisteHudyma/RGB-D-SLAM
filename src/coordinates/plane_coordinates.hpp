#ifndef RGBDSLAM_PLANE_COORDINATES_HPP
#define RGBDSLAM_PLANE_COORDINATES_HPP

#include "types.hpp"
#include <cmath>
#include "point_coordinates.hpp"

namespace rgbd_slam {

struct PlaneCameraCoordinates;
struct PlaneWorldCoordinates;

/**
 * \brief Base class to handle plane coordinates
 */
struct PlaneCoordinates
{
    PlaneCoordinates() : _normal(vector3::Zero()), _d(0.0) {};
    PlaneCoordinates(const vector4& parametrization) : _normal(parametrization.head<3>()), _d(parametrization(3))
    {
        _normal.normalize();
    };
    PlaneCoordinates(const vector3& normal, const double d) : _normal(normal), _d(d) { _normal.normalize(); };
    PlaneCoordinates(const PlaneCoordinates& other) : _normal(other.get_normal()), _d(other.get_d())
    {
        _normal.normalize();
    }

    PlaneCoordinates& operator=(const PlaneCoordinates& other) noexcept
    {
        // Guard self assignment
        if (this == &other)
            return *this;

        _normal = other._normal;
        _normal.normalize();

        _d = other._d;
        return *this;
    }

    [[nodiscard]] vector4 get_parametrization() const noexcept
    {
        return vector4(_normal.x(), _normal.y(), _normal.z(), _d);
    };
    [[nodiscard]] vector3 get_normal() const noexcept { return _normal; };
    [[nodiscard]] vector3& normal() noexcept { return _normal; };

    [[nodiscard]] double get_d() const noexcept { return _d; };
    [[nodiscard]] double& d() noexcept { return _d; };

    [[nodiscard]] WorldCoordinate get_center() const noexcept { return WorldCoordinate(_normal * (-_d)); };
    [[nodiscard]] double get_point_distance(const vector3& point) const noexcept
    {
        // distance can be negative depending on the plane
        return abs(_normal.transpose() * point + _d);
    }
    [[nodiscard]] double get_point_distance_squared(const vector3& point) const noexcept
    {
        // distance can be negative depending on the plane
        return SQR(_normal.transpose() * point + _d);
    }
    [[nodiscard]] double get_cos_angle(const PlaneCoordinates& other) const noexcept
    {
        return _normal.dot(other._normal);
    };

    [[nodiscard]] bool hasNaN() const noexcept { return std::isnan(_d) or _normal.hasNaN(); };

  private:
    vector3 _normal;
    double _d;
};

/**
 * \brief The parametrization of a plane in camera coordinates
 */
struct PlaneCameraCoordinates : public PlaneCoordinates
{
    using PlaneCoordinates::PlaneCoordinates;

    /**
     * \brief project to world coordinates
     */
    [[nodiscard]] PlaneWorldCoordinates to_world_coordinates(
            const PlaneCameraToWorldMatrix& cameraToWorld) const noexcept;
};

/**
 * \brief The parametrization of a plane in world coordinates
 */
struct PlaneWorldCoordinates : public PlaneCoordinates
{
    using PlaneCoordinates::PlaneCoordinates;

    /**
     * \brief project to camera coordinates
     */
    [[nodiscard]] PlaneCameraCoordinates to_camera_coordinates(
            const PlaneWorldToCameraMatrix& worldToCamera) const noexcept;

    /**
     * \brief Compute a distance between two planes, by retroprojecting a world plane to camera space
     * \param[in] cameraPlane A plane in camera coordinates
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     *
     * \return A 3D vector of the error between the two planes. The x and y are angle distances, the z is in millimeters
     */
    [[nodiscard]] vector4 get_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                              const PlaneWorldToCameraMatrix& worldToCamera) const noexcept;
    /**
     * \brief Compute a distance between two planes, by retroprojecting a world plane to camera space. Result is reduced
     * to two angles and a distance
     * \param[in] cameraPlane A plane in camera coordinates
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return A 3D vector of the error between the two planes. The x and y are angle distances, the z is in millimeters
     */
    [[nodiscard]] vector3 get_reduced_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                                      const PlaneWorldToCameraMatrix& worldToCamera) const noexcept;
};

} // namespace rgbd_slam

#endif