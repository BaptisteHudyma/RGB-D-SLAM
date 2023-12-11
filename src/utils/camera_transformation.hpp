#ifndef RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP
#define RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP

#include "../types.hpp"

namespace rgbd_slam::utils {

[[nodiscard]] matrix44 get_transformation_matrix(const quaternion& rotation, const vector3& position) noexcept;

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a world point (xyz) to camera point (uvd)
 */
[[nodiscard]] WorldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation,
                                                                    const vector3& position) noexcept;

[[nodiscard]] WorldToCameraMatrix compute_world_to_camera_transform(const CameraToWorldMatrix& cameraToWorld) noexcept;

/**
 * \brief DO NOT USE EXPECT FOR TESTING.
 * This returns the transformation matrix UNRECTIFIED.
 */
[[nodiscard]] WorldToCameraMatrix compute_world_to_camera_transform_no_correction(const quaternion& rotation,
                                                                                  const vector3& position) noexcept;

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a camera point (uvd) to world point (xyz)
 */
[[nodiscard]] CameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation,
                                                                    const vector3& position) noexcept;

[[nodiscard]] CameraToWorldMatrix compute_camera_to_world_transform(const WorldToCameraMatrix& worldToCamera) noexcept;

/**
 * \brief DO NOT USE EXPECT FOR TESTING.
 * This returns the transformation matrix UNRECTIFIED.
 */
[[nodiscard]] CameraToWorldMatrix compute_camera_to_world_transform_no_correction(const quaternion& rotation,
                                                                                  const vector3& position) noexcept;

/**
 * \brief Transform a CameraToWorldMatrix to a special plane cameraToWorld matrix
 */
[[nodiscard]] PlaneCameraToWorldMatrix compute_plane_camera_to_world_matrix(
        const CameraToWorldMatrix& cameraToWorld) noexcept;
/**
 * \brief Transform a WorldToCameraMatrix to a special plane worldToCamera matrix
 */
[[nodiscard]] PlaneWorldToCameraMatrix compute_plane_world_to_camera_matrix(
        const WorldToCameraMatrix& worldToCamera) noexcept;

} // namespace rgbd_slam::utils

#endif
