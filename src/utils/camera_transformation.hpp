#ifndef RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP
#define RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP

#include "../types.hpp"

namespace rgbd_slam {
namespace utils {

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a world point (xyz) to camera point (uvd)
 */
const worldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position);

const worldToCameraMatrix compute_world_to_camera_transform(const cameraToWorldMatrix& cameraToWorld);

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a camera point (uvd) to world point (xyz)
 */
const cameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position);

const cameraToWorldMatrix compute_camera_to_world_transform(const worldToCameraMatrix& worldToCamera);

/**
 * \brief Transform a cameraToWorldMatrix to a special plane cameraToWorld matrix
 */
planeCameraToWorldMatrix compute_plane_camera_to_world_matrix(const cameraToWorldMatrix& cameraToWorld);
/**
 * \brief Transform a worldToCameraMatrix to a special plane worldToCamera matrix
 */
planeWorldToCameraMatrix compute_plane_world_to_camera_matrix(const worldToCameraMatrix& worldToCamera);

} // namespace utils
} // namespace rgbd_slam

#endif
