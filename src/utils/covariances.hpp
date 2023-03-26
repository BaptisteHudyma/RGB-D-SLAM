#ifndef RGBDSLAM_UTILS_COVARIANCES_HPP
#define RGBDSLAM_UTILS_COVARIANCES_HPP

#include "coordinates.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "types.hpp"

namespace rgbd_slam {
namespace utils {

/**
 * \brief Return the expected depth quantization at this depth value.
 * \param[in] depht The measured depth value, in millimeters
 * \return The smallest possible measure in millimeters (caped at 0.5 mm)
 */
double get_depth_quantization(const double depht);

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (world space)
 * \param[in] pointCovariance The covariance associated with this point (world space)
 */
const ScreenCoordinateCovariance get_screen_point_covariance(const WorldCoordinate& point,
                                                             const WorldCoordinateCovariance& pointCovariance);

/**
 * \brief Compute a screen point covariance from a given point
 *
 * \param[in] point The coordinates of this 3D point (camera space)
 * \param[in] pointCovariance The covariance associated with this point (camera space)
 */
const ScreenCoordinateCovariance get_screen_point_covariance(const CameraCoordinate& point,
                                                             const CameraCoordinateCovariance& pointCovariance);

/**
 * \brief Compute the covariance of a world point
 */
const WorldCoordinateCovariance get_world_point_covariance(const CameraCoordinateCovariance& cameraPointCovariance,
                                                           const matrix33& poseCovariance);
/**
 * \brief Compute covariance of a screen point in world state
 */
const WorldCoordinateCovariance get_world_point_covariance(const ScreenCoordinate& screenPoint,
                                                           const matrix33& poseCovariance);

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point. This
 * function will internaly compute the covariance of the screen point.
 * \param[in] screenPoint The 2D point in screen coordinates
 * \return the covariance of the 3D camera point
 */
const CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint);

/**
 * \brief Compute the associated Gaussian error of a screen point when it will be transformed to camera point
 * \param[in] screenPoint The 2D point in screen coordinates
 * \param[in] screenPointCovariance The covariance matrix associated with a point in screen space
 * \return the covariance of the 3D camera point
 */
const CameraCoordinateCovariance get_camera_point_covariance(const ScreenCoordinate& screenPoint,
                                                             const ScreenCoordinateCovariance& screenPointCovariance);

/**
 * \brief Compute the covariance of the plane parameters
 * Inspired by : revisiting uncertainty analysis for optimum planes extracted from 3d range sensor point-cloud
 * \param[in] parametersMatrix The matrix used to compute the plane parameters
 * \param[in] normal Normal vector of this plane
 * \param[in] centroid Centroid of this plane. Will be used as the base for the plane covariance
 * \param[in] centroidError optionnal. if given, will be added to the centroid error.
 * \return the covariance of the plane. If centroidError is not given, it is in camera space.
 */
const matrix44 compute_plane_covariance(const matrix33& parametersMatrix,
                                        const vector3& normal,
                                        const vector3& centroid,
                                        const matrix33& centroidError = matrix33::Zero());

} // namespace utils
} // namespace rgbd_slam

#endif
