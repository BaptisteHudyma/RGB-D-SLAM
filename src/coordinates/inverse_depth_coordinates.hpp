#ifndef RGBDSLAM_INVERSE_DEPTH_COORDINATES_HPP
#define RGBDSLAM_INVERSE_DEPTH_COORDINATES_HPP

#include "types.hpp"
#include "line.hpp"
#include "point_coordinates.hpp"

namespace rgbd_slam {

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
     * \param[in] other
     * \return The distance between the two closest points on the lines
     */
    [[nodiscard]] vector3 compute_signed_distance(const InverseDepthWorldPoint& other) const;

    /**
     * \brief signed Line to line distance
     * \param[in] other The 2d observation to put in inverse depth coordinates
     * \param[in] w2c The matrix to go from world to camera space
     * \return The distance between the two closest points on the lines
     */
    [[nodiscard]] vector3 compute_signed_distance(const ScreenCoordinate2D& other,
                                                  const WorldToCameraMatrix& w2c) const;
    /**
     * \brief compute distance of the screen projections
     * \param[in] other The 2d observation in the new image
     * \param[in] inverseDepthCovariance The covariance of the inverse depth
     * \param[in] w2c The matrix to go from world to camera space
     * \return The distance between the two observations, in pixels
     */
    [[nodiscard]] vector2 compute_signed_screen_distance(const ScreenCoordinate2D& other,
                                                         const double inverseDepthCovariance,
                                                         const WorldToCameraMatrix& w2c) const;

    /**
     * \brief Set the parameters of this instance from a cartesian point
     * \param[in] point The observed point in world coordinates
     * \param[in] origin The point where the point was observed
     */
    [[nodiscard]] static InverseDepthWorldPoint from_cartesian(const WorldCoordinate& point,
                                                               const WorldCoordinate& origin) noexcept;

    /**
     * \brief Set the parameters of this instance from a cartesian point
     * \param[in] point The observed point in world coordinates
     * \param[in] origin The point where the point was observed
     * \param[out] jacobian The jacobian of this transformation
     */
    [[nodiscard]] static InverseDepthWorldPoint from_cartesian(const WorldCoordinate& point,
                                                               const WorldCoordinate& origin,
                                                               Eigen::Matrix<double, 6, 3>& jacobian) noexcept;

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
    [[nodiscard]] WorldCoordinate to_world_coordinates(Eigen::Matrix<double, 3, 6>& jacobian) const noexcept;

    /**
     * \brief Compute a line that represent the potential position of the inverse depth point, taking into account the
     * uncertainty
     * \param[in] w2c The world to camera matrix
     * \param[in] inverseDepthCovariance The covariance of the inverse depth
     * \param[out] screenSegment the projection of this point as a segment
     * \return true if screenSegment is valid
     */
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                             const double inverseDepthCovariance,
                                             utils::Segment<2>& screenSegment) const noexcept;

    /**
     * \brief Compute a line that represent the potential position of the inverse depth point, taking into account the
     * uncertainty
     * \param[in] w2c The world to camera matrix
     * \param[in] inverseDepthCovariance The covariance of the inverse depth
     * \param[out] screenSegment the projection of this point as a segment (with depth !)
     * \return true if screenSegment is valid
     */
    [[nodiscard]] bool to_screen_coordinates(const WorldToCameraMatrix& w2c,
                                             const double inverseDepthCovariance,
                                             utils::Segment<3>& screenSegment) const noexcept;

    WorldCoordinate get_closest_estimation(const double inverseDepthStandardDev) const;
    WorldCoordinate get_furthest_estimation(const double inverseDepthStandardDev) const;

    // changing this implies that all computations should be changed, handle with care. Those should be
    // always in [0, 5]
    static constexpr uint firstPoseIndex = 0; // takes 3 spaces
    static constexpr uint inverseDepthIndex = 3;
    static constexpr uint thetaIndex = 4;
    static constexpr uint phiIndex = 5;

    /**
     * GETTERS
     */

    [[nodiscard]] WorldCoordinate get_first_observation() const noexcept { return _firstObservation; };
    [[nodiscard]] double get_inverse_depth() const noexcept { return _inverseDepth_mm; };
    [[nodiscard]] double get_theta() const noexcept { return _theta_rad; };
    [[nodiscard]] double get_phi() const noexcept { return _phi_rad; };
    [[nodiscard]] vector3 get_bearing_vector() const noexcept { return _bearingVector; };

  private:
    WorldCoordinate _firstObservation; // position of the camera for the first observation
    double _inverseDepth_mm = 0.0;     // inverse of the depth (>= 0)
    double _theta_rad = 0.0;           // elevation angle of the first observation, in world space
    double _phi_rad = 0.0;             // heading angle of the first observation, in world space

    vector3 _bearingVector; // get the bearing vector that point from _firstObservation to the point
};

} // namespace rgbd_slam

#endif
