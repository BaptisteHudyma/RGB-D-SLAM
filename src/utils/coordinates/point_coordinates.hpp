#ifndef RGBDSLAM_UTILS_POINT_COORDINATES_HPP
#define RGBDSLAM_UTILS_POINT_COORDINATES_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

struct ScreenCoordinate2D;
struct ScreenCoordinate;

struct CameraCoordinate2D;
struct CameraCoordinate;

struct WorldCoordinate;

struct InverseDepthWorldPoint;

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
                                                 const vector3& centerTo);

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
    [[nodiscard]] CameraCoordinate2D to_camera_coordinates() const;

    /**
     * \brief Compute a covariance in screen space
     */
    [[nodiscard]] matrix22 get_covariance() const;

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
    [[nodiscard]] WorldCoordinate to_world_coordinates(const CameraToWorldMatrix& cameraToWorld) const;

    /**
     * \brief Transform a screen point with a depth value to a 3D camera point
     * \return A 3D point in camera coordinates
     */
    [[nodiscard]] CameraCoordinate to_camera_coordinates() const;

    /**
     * \brief Compute a covariance in screen space
     */
    [[nodiscard]] ScreenCoordinateCovariance get_covariance() const;

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
                                                    const WorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief Compute a distance between this world point and a screen point, by retroprojecting the world point to
     * screen space.
     * \param[in] screenPoint A point in screen space. Only the x and y components will be used
     * \param[in] worldToCamera A transformation matrix to convert from world to camera space
     * \return an unsigned distance in camera space (pixels)
     */
    [[nodiscard]] double get_distance_px(const ScreenCoordinate2D& screenPoint,
                                         const WorldToCameraMatrix& worldToCamera) const;
    /**
     * \brief Compute a signed distance between a world point and a 3D point in screen space, by projecting the screen
     * point to world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The 3D signed distance in world space
     */
    [[nodiscard]] vector3 get_signed_distance_mm(const ScreenCoordinate& screenPoint,
                                                 const CameraToWorldMatrix& cameraToWorld) const;
    /**
     * \brief Compute a distance between a world point and a 3D point in screen space, by projecting the screen point to
     * world space
     * \param[in] screenPoint A 3D point in screen space
     * \param[in] cameraToWorld A matrix to convert from camera to world space
     * \return The unsigned distance in world space
     */
    [[nodiscard]] double get_distance_mm(const ScreenCoordinate& screenPoint,
                                         const CameraToWorldMatrix& cameraToWorld) const;
    /**
     * \brief Compute a signed distance with another world point
     */
    [[nodiscard]] vector3 get_signed_distance_mm(const WorldCoordinate& worldPoint) const
    {
        return this->base() - worldPoint;
    };
    [[nodiscard]] double get_distance_mm(const WorldCoordinate& worldPoint) const
    {
        return get_signed_distance_mm(worldPoint).lpNorm<1>();
    };
};

/**
 * \brief Contain an inverse depth representation of a world point.
 * This is used to represent point with an unknown depth
 */
struct InverseDepthWorldPoint
{
  public:
    InverseDepthWorldPoint(const WorldCoordinate& firstPose,
                           const double inverseDepth,
                           const double theta,
                           const double phi);
    InverseDepthWorldPoint(const ScreenCoordinate2D& observation, const CameraToWorldMatrix& c2w);
    InverseDepthWorldPoint(const CameraCoordinate& observation, const CameraToWorldMatrix& c2w);

    /**
     * \brief signed Line to line distance
     */
    vector3 compute_signed_distance(const InverseDepthWorldPoint& other) const;
    vector3 compute_signed_distance(const ScreenCoordinate2D& other, const WorldToCameraMatrix& c2w) const;

    /**
     * \brief Set the parameters of this instance from a cartesian point
     * \param[in] point The observed point in world coordinates
     * \param[in] origin The point where the point was observed
     */
    void from_cartesian(const WorldCoordinate& point, const WorldCoordinate& origin) noexcept;

    /**
     * \brief Set the parameters of this instance from a cartesian point
     * \param[in] point The observed point in world coordinates
     * \param[in] origin The point where the point was observed
     * \param[out] jacobian The jacobian of this transformation
     */
    void from_cartesian(const WorldCoordinate& point,
                        const WorldCoordinate& origin,
                        Eigen::Matrix<double, 6, 6>& jacobian) noexcept;

    /**
     * \brief compute the cartesian projection of this point in world space.
     * \return The point in camera coordinates (the associated covariance can be huge)
     */
    [[nodiscard]] WorldCoordinate to_world_coordinates() const noexcept;

    /**
     * \brief compute the cartesian projection of this point in world space.
     * \param[out] jacobian The jacobian of this transformation
     * \return The point in camera coordinates (the associated covariance can be huge)
     */
    WorldCoordinate to_world_coordinates(Eigen::Matrix<double, 3, 6>& jacobian) const noexcept;

    /**
     * \brief Compute the projected coordinates of this point to camera space
     * \param[in] w2c The matrix to go from world to camera space
     * \return The point in camera coordinates (the associated covariance can be huge)
     */
    [[nodiscard]] CameraCoordinate to_camera_coordinates(const WorldToCameraMatrix& w2c) const noexcept;

    /**
     * \brief Compute the projected coordinates of this point to screen space
     * \param[in] w2c The matrix to go from world to camera space
     * \param[out] screenCoordinates The projected coordinates, only valid if the function returned true
     * \return True if the process succeeded (the associated covariance can be huge)
     */
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                             ScreenCoordinate2D& screenCoordinates) const noexcept;

    // get the bearing vector that point from _firstObservation to the point
    vector3 get_bearing_vector() const noexcept;

    [[nodiscard]] vector6 get_vector_state() const noexcept;
    void from_vector_state(const vector6& state) noexcept;

    WorldCoordinate _firstObservation; // position of the camera for the first observation
    double _inverseDepth_mm = 0.0;     // inverse of the depth (>= 0)
    double _theta_rad = 0.0;           // elevation angle of the first observation, in world space
    double _phi_rad = 0.0;             // heading angle of the first observation, in world space

    // changing this implies that all computations should be changed, handle with care. Those should be
    // always in [0, 5]
    static constexpr uint firstPoseIndex = 0; // takes 3 spaces
    static constexpr uint inverseDepthIndex = 3;
    static constexpr uint thetaIndex = 4;
    static constexpr uint phiIndex = 5;
};

} // namespace rgbd_slam::utils

#endif