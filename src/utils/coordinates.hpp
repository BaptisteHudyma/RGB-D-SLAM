#ifndef RGBDSLAM_UTILS_COORDINATES_HPP
#define RGBDSLAM_UTILS_COORDINATES_HPP

#include "../types.hpp"
#include <cmath>
#include <ostream>

namespace rgbd_slam::utils {

struct ScreenCoordinate2D;
struct ScreenCoordinate;

struct CameraCoordinate2D;
struct CameraCoordinate;

struct WorldCoordinate;

struct PlaneCameraCoordinates;
struct PlaneWorldCoordinates;

/**
 * \brief Return true is a measurement is in the measurement range
 */
[[nodiscard]] bool is_depth_valid(const double depth) noexcept;

/**
 * \brief Compute the transformation matrix between two coordinate systems
 * \param[in] xFrom The x axis of the original coordinate system
 * \param[in] yFrom The y axis of the original coordinate system
 * \param[in] centerFrom  The center of the original coordinate system
 * \param[in] xTo The x axis of the next coordinate system
 * \param[in] yTo The y axis of the next coordinate system
 * \param[in] centerTo The center of the next coordinate system
 * \return The transformation matrix between those coordinate systems
 */
[[nodiscard]] matrix44 get_transformation_matrix(const vector3& xFrom,
                                                 const vector3& yFrom,
                                                 const vector3& centerFrom,
                                                 const vector3& xTo,
                                                 const vector3& yTo,
                                                 const vector3& centerTo) noexcept;

/**
 * \brief Contains a single of coordinate in screen space.
 * Screen space is defined as (x, y) in pixels
 */
struct ScreenCoordinate2D : public vector2
{
    using vector2::vector2;

    ScreenCoordinate2D() : vector2(vector2::Zero()) {};

    /**
     * \brief Transform a screen point with a depth value to a 3D camera point
     * \return A 3D point in camera coordinates
     */
    [[nodiscard]] CameraCoordinate2D to_camera_coordinates() const noexcept;

    /**
     * \brief Compute a covariance in screen space
     */
    [[nodiscard]] matrix22 get_covariance() const noexcept;

    /**
     * \return true if this point is in the visible screen space
     */
    [[nodiscard]] bool is_in_screen_boundaries() const noexcept;
};

/**
 * \brief Contains a single of coordinate in screen space.
 * Screen space is defined as (x, y) in pixels, and z in distance (millimeters)
 */
struct ScreenCoordinate : public vector3
{
    using vector3::vector3;

    ScreenCoordinate() : vector3(vector3::Zero()) {};

    /**
     * \brief Transform a screen point with a depth value to a 3D world point
     * \param[in] cameraToWorld Matrix to transform local to world coordinates
     * \return A 3D point in world coordinates
     */
    [[nodiscard]] WorldCoordinate to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const noexcept;

    /**
     * \brief Transform a screen point with a depth value to a 3D camera point
     * \return A 3D point in camera coordinates
     */
    [[nodiscard]] CameraCoordinate to_camera_coordinates() const noexcept;

    /**
     * \brief Compute a covariance in screen space
     */
    [[nodiscard]] ScreenCoordinateCovariance get_covariance() const noexcept;

    [[nodiscard]] ScreenCoordinate2D get_2D() const noexcept { return ScreenCoordinate2D(x(), y()); }

    /**
     * \return true if this point is in the visible screen space
     */
    [[nodiscard]] bool is_in_screen_boundaries() const noexcept;
};

/**
 * \brief Contains a single of coordinate in camera space.
 * Camera space is defined as (x, y), relative to the camera center
 */
struct CameraCoordinate2D : public vector2
{
    using vector2::vector2;

    CameraCoordinate2D() : vector2(vector2::Zero()) {};

    /**
     * \brief Transform a point from camera to screen coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    [[nodiscard]] bool to_screen_coordinates(ScreenCoordinate2D& screenPoint) const noexcept;
};

/**
 * \brief Contains a single of coordinate in camera space.
 * Camera space is defined as (x, y, z) in distance (millimeters), relative to the camera center
 */
struct CameraCoordinate : public vector3
{
    using vector3::vector3;

    CameraCoordinate() : vector3(vector3::Zero()) {};
    CameraCoordinate(const vector4& homegenousCoordinates) :
        vector3(homegenousCoordinates.x() / homegenousCoordinates[3],
                homegenousCoordinates.y() / homegenousCoordinates[3],
                homegenousCoordinates.z() / homegenousCoordinates[3]) {};
    CameraCoordinate(const CameraCoordinate2D& other, const double z) : vector3(other.x(), other.y(), z) {};

    /**
     * \brief Transform a camera point to a 3D world point
     * \param[in] cameraToWorld Matrix to transform local to world coordinates
     * \return A 3D point in world coordinates
     */
    [[nodiscard]] WorldCoordinate to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const noexcept;

    /**
     * \brief Transform a point from camera to screen coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    [[nodiscard]] bool to_screen_coordinates(ScreenCoordinate& screenPoint) const noexcept;
    [[nodiscard]] bool to_screen_coordinates(ScreenCoordinate2D& screenPoint) const noexcept;
};

/**
 * \brief Contains a single of coordinate in world space.
 * World space is defined as (x, y, z) in distance (millimeters), relative to the world center
 */
struct WorldCoordinate : public vector3
{
    using vector3::vector3;

    WorldCoordinate() : vector3(vector3::Zero()) {};

    /**
     * \brief Transform a point from world to screen coordinate system
     * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                             ScreenCoordinate& screenPoint) const noexcept;
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& worldToCamera,
                                             ScreenCoordinate2D& screenPoint) const noexcept;

    /**
     * \brief Transform a vector in world space to a vector in camera space
     * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
     * \return The input vector transformed to camera space
     */
    [[nodiscard]] CameraCoordinate to_camera_coordinates(const WorldToCameraMatrix& worldToCamera) const noexcept;

    /**
     * \brief Compute a signed 2D distance between this world point and a screen point, by retroprojecting the world
     * point to screen space
     * \param[in] screenPoint A point in screen space. Only the x and y components will be used
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return a 2D signed distance in camera space (pixels)
     */
    [[nodiscard]] vector2 get_signed_distance_2D_px(const ScreenCoordinate2D& screenPoint,
                                                    const WorldToCameraMatrix& worldToCamera) const noexcept;
    /**
     * \brief Compute a distance between this world point and a screen point, by retroprojecting the world point to
     * screen space.
     * \param[in] screenPoint A point in screen space. Only the x and y components will be used
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return an unsigned distance in camera space (pixels)
     */
    [[nodiscard]] double get_distance_px(const ScreenCoordinate2D& screenPoint,
                                         const WorldToCameraMatrix& worldToCamera) const noexcept;
    /**
     * \brief Compute a signed distance between a world point and a 3D point in screen space, by projecting the screen
     * point to world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The 3D signed distance in world space
     */
    [[nodiscard]] vector3 get_signed_distance_mm(const ScreenCoordinate& screenPoint,
                                                 const CameraToWorldMatrix& cameraToWorld) const noexcept;
    /**
     * \brief Compute a distance between a world point and a 3D point in screen space, by projecting the screen point to
     * world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The unsigned distance in world space
     */
    [[nodiscard]] double get_distance_mm(const ScreenCoordinate& screenPoint,
                                         const CameraToWorldMatrix& cameraToWorld) const noexcept;
    /**
     * \brief Compute a signed distance with another world point
     */
    [[nodiscard]] vector3 get_signed_distance_mm(const WorldCoordinate& worldPoint) const noexcept
    {
        return this->base() - worldPoint;
    };
    [[nodiscard]] double get_distance_mm(const WorldCoordinate& worldPoint) const noexcept
    {
        return get_signed_distance_mm(worldPoint).lpNorm<1>();
    };
};

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
        return pow(_normal.transpose() * point + _d, 2.0);
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

} // namespace rgbd_slam::utils

#endif