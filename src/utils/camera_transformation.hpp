#ifndef RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP
#define RGBDSLAM_UTILS_CAMERA_TRANSFORMATION_HPP

#include "../types.hpp"

namespace rgbd_slam {
namespace utils {

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a world point (xyz) to camera point (uvd)
 */
const WorldToCameraMatrix compute_world_to_camera_transform(const quaternion& rotation, const vector3& position);

const WorldToCameraMatrix compute_world_to_camera_transform(const CameraToWorldMatrix& cameraToWorld);

/**
 * \brief Given a camera pose, returns a transformation matrix to convert a camera point (uvd) to world point (xyz)
 */
const CameraToWorldMatrix compute_camera_to_world_transform(const quaternion& rotation, const vector3& position);

const CameraToWorldMatrix compute_camera_to_world_transform(const WorldToCameraMatrix& worldToCamera);

/**
 * \brief Transform a CameraToWorldMatrix to a special plane cameraToWorld matrix
 */
PlaneCameraToWorldMatrix compute_plane_camera_to_world_matrix(const CameraToWorldMatrix& cameraToWorld);
/**
 * \brief Transform a WorldToCameraMatrix to a special plane worldToCamera matrix
 */
PlaneWorldToCameraMatrix compute_plane_world_to_camera_matrix(const WorldToCameraMatrix& worldToCamera);

} // namespace utils
} // namespace rgbd_slam

#endif
