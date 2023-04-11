#ifndef RGBDSLAM_UTILS_COORDINATES_HPP
#define RGBDSLAM_UTILS_COORDINATES_HPP

#include "../types.hpp"
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
bool is_depth_valid(const double depth);

/**
 * \brief Contains a single of coordinate in screen space.
 * Screen space is defined as (x, y) in pixels
 */
struct ScreenCoordinate2D : public vector2
{
    ScreenCoordinate2D() : vector2(vector2::Zero()) {};
    ScreenCoordinate2D(const double x, const double y) : vector2(x, y) {};
    ScreenCoordinate2D(const vector2& other) : vector2(other) {};

    /**
     * \brief Transform a screen point with a depth value to a 3D camera point
     * \return A 3D point in camera coordinates
     */
    CameraCoordinate2D to_camera_coordinates() const;

    /**
     * \brief Compute a covariance in screen space
     */
    matrix22 get_covariance() const;
};

/**
 * \brief Contains a single of coordinate in screen space.
 * Screen space is defined as (x, y) in pixels, and z in distance (millimeters)
 */
struct ScreenCoordinate : public vector3
{
    ScreenCoordinate() : vector3(vector3::Zero()) {};
    ScreenCoordinate(const vector3& other) : vector3(other) {};
    ScreenCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};

    /**
     * \brief Transform a screen point with a depth value to a 3D world point
     * \param[in] cameraToWorld Matrix to transform local to world coordinates
     * \return A 3D point in world coordinates
     */
    WorldCoordinate to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const;

    /**
     * \brief Transform a screen point with a depth value to a 3D camera point
     * \return A 3D point in camera coordinates
     */
    CameraCoordinate to_camera_coordinates() const;

    /**
     * \brief Compute a covariance in screen space
     */
    ScreenCoordinateCovariance get_covariance() const;

    ScreenCoordinate2D get_2D() const { return ScreenCoordinate2D(x(), y()); }
};

/**
 * \brief Contains a single of coordinate in camera space.
 * Camera space is defined as (x, y), relative to the camera center
 */
struct CameraCoordinate2D : public vector2
{
    CameraCoordinate2D() : vector2(vector2::Zero()) {};
    CameraCoordinate2D(const vector2& other) : vector2(other) {};
    CameraCoordinate2D(const double x, const double y) : vector2(x, y) {};

    /**
     * \brief Transform a point from camera to screen coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    bool to_screen_coordinates(ScreenCoordinate2D& screenPoint) const;
};

/**
 * \brief Contains a single of coordinate in camera space.
 * Camera space is defined as (x, y, z) in distance (millimeters), relative to the camera center
 */
struct CameraCoordinate : public vector3
{
    /**
     * \brief Scores a 3D coordinate in camera (x, y, depth). It can be projected to world space using a pose
     * transformation
     */
    CameraCoordinate() : vector3(vector3::Zero()) {};
    CameraCoordinate(const vector3& coords) : vector3(coords) {};
    CameraCoordinate(const vector4& homegenousCoordinates) :
        vector3(homegenousCoordinates.x() / homegenousCoordinates[3],
                homegenousCoordinates.y() / homegenousCoordinates[3],
                homegenousCoordinates.z() / homegenousCoordinates[3]) {};
    CameraCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};
    CameraCoordinate(const CameraCoordinate2D& other, const double z) : vector3(other.x(), other.y(), z) {};
    vector4 get_homogenous() const { return vector4(x(), y(), z(), 1); };

    /**
     * \brief Transform a camera point to a 3D world point
     * \param[in] cameraToWorld Matrix to transform local to world coordinates
     * \return A 3D point in world coordinates
     */
    WorldCoordinate to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const;

    /**
     * \brief Transform a point from camera to screen coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    bool to_screen_coordinates(ScreenCoordinate& screenPoint) const;
    bool to_screen_coordinates(ScreenCoordinate2D& screenPoint) const;
};

/**
 * \brief Contains a single of coordinate in world space.
 * World space is defined as (x, y, z) in distance (millimeters), relative to the world center
 */
struct WorldCoordinate : public vector3
{
    /**
     * \brief Scores a 3D coordinate in world space (x, y, depth).
     */
    WorldCoordinate() : vector3(vector3::Zero()) {};
    WorldCoordinate(const vector3& coords) : vector3(coords) {};
    WorldCoordinate(const double x, const double y, const double z) : vector3(x, y, z) {};

    /**
     * \brief Transform a point from world to screen coordinate system
     * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
     * \param[out] screenPoint The point screen coordinates, if the function returned true
     * \return True if the screen position is valid
     */
    bool to_screen_coordinates(const WorldToCameraMatrix& worldToCamera, ScreenCoordinate& screenPoint) const;
    bool to_screen_coordinates(const WorldToCameraMatrix& worldToCamera, ScreenCoordinate2D& screenPoint) const;

    /**
     * \brief Transform a vector in world space to a vector in camera space
     * \param[in] worldToCamera Matrix to transform the world to a local coordinate system
     * \return The input vector transformed to camera space
     */
    CameraCoordinate to_camera_coordinates(const WorldToCameraMatrix& worldToCamera) const;

    /**
     * \brief Compute a signed 2D distance between this world point and a screen point, by retroprojecting the world
     * point to screen space
     * \param[in] screenPoint A point in screen space. Only the x and y components will be used
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return a 2D signed distance in camera space (pixels)
     */
    vector2 get_signed_distance_2D(const ScreenCoordinate2D& screenPoint,
                                   const WorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief Compute a distance between this world point and a screen point, by retroprojecting the world point to
     * screen space.
     * \param[in] screenPoint A point in screen space. Only the x and y components will be used
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return an unsigned distance in camera space (pixels)
     */
    double get_distance(const ScreenCoordinate2D& screenPoint, const WorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief Compute a signed distance between a world point and a 3D point in screen space, by projecting the screen
     * point to world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The 3D signed distance in world space
     */
    vector3 get_signed_distance(const ScreenCoordinate& screenPoint, const CameraToWorldMatrix& cameraToWorld) const;
    /**
     * \brief Compute a distance between a world point and a 3D point in screen space, by projecting the screen point to
     * world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The unsigned distance in world space
     */
    double get_distance(const ScreenCoordinate& screenPoint, const CameraToWorldMatrix& cameraToWorld) const;
    /**
     * \brief Compute a signed distance with another world point
     */
    vector3 get_signed_distance(const WorldCoordinate& worldPoint) const { return this->base() - worldPoint; };
    double get_distance(const WorldCoordinate& worldPoint) const
    {
        return get_signed_distance(worldPoint).lpNorm<1>();
    };
};

struct PlaneCameraCoordinates : vector4
{
    PlaneCameraCoordinates() : vector4(vector4::Zero()) {};
    PlaneCameraCoordinates(const vector4& plane) : vector4(plane) {};
    PlaneCameraCoordinates(const vector3& normal, const double d) : vector4(normal.x(), normal.y(), normal.z(), d) {};
    PlaneCameraCoordinates(const double x, const double y, const double z, const double d) : vector4(x, y, z, d) {};

    PlaneWorldCoordinates to_world_coordinates(const PlaneCameraToWorldMatrix& cameraToWorld) const;
    /**
     * \brief project to world coordinates, with a renormalization of the normal. This is necessary because of
     * floating point errors that accumulates
     */
    PlaneWorldCoordinates to_world_coordinates_renormalized(const PlaneCameraToWorldMatrix& cameraToWorld) const;
};

struct PlaneWorldCoordinates : public vector4
{
    PlaneWorldCoordinates() : vector4(vector4::Zero()) {};
    PlaneWorldCoordinates(const vector4& plane) : vector4(plane) {};
    PlaneWorldCoordinates(const double x, const double y, const double z, const double d) : vector4(x, y, z, d) {};

    PlaneCameraCoordinates to_camera_coordinates(const PlaneWorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief project to camera coordinates, with a renormalization of the normal. This is necessary because of
     * floating point errors that accumulates
     */
    PlaneCameraCoordinates to_camera_coordinates_renormalized(const PlaneWorldToCameraMatrix& worldToCamera) const;

    /**
     * \brief Compute a distance between two planes, by retroprojecting a world plane to camera space
     * \param[in] cameraPlane A plane in camera coordinates
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     *
     * \return A 3D vector of the error between the two planes. The x and y are angle distances, the z is in millimeters
     */
    vector4 get_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                const PlaneWorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief Compute a distance between two planes, by retroprojecting a world plane to camera space. Result is reduced
     * to two angles and a distance
     * \param[in] cameraPlane A plane in camera coordinates
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return A 3D vector of the error between the two planes. The x and y are angle distances, the z is in millimeters
     */
    vector3 get_reduced_signed_distance(const PlaneCameraCoordinates& cameraPlane,
                                        const PlaneWorldToCameraMatrix& worldToCamera) const;
};

/**
 * \brief Compute the position of a point in the plane coordinate system
 * \param[in] pointToProject The point to project to plane, in world coordinates
 * \param[in] planeCenter The center point of the plane
 * \param[in] xAxis The unit y vector of the plane, othogonal to the normal
 * \param[in] yAxis The unit x vector of the plane, othogonal to the normal and u
 * \return A 2D point corresponding to pointToProject, in plane coordinate system
 */
vector2 get_projected_plan_coordinates(const vector3& pointToProject,
                                       const vector3& planeCenter,
                                       const vector3& xAxis,
                                       const vector3& yAxis);

/**
 * \brief Compute the projection of a point from the plane coordinate system to world
 * \param[in] pointToProject The point to project to world, in plane coordinates
 * \param[in] planeCenter The center point of the plane
 * \param[in] xAxis The unit x vector of the plane, othogonal to the normal
 * \param[in] yAxis The unit y vector of the plane, othogonal to the normal and u
 * \return A 3D point corresponding to pointToProject, in world coordinate system
 */
vector3 get_point_from_plane_coordinates(const vector2& pointToProject,
                                         const vector3& planeCenter,
                                         const vector3& xAxis,
                                         const vector3& yAxis);

} // namespace rgbd_slam::utils

#endif