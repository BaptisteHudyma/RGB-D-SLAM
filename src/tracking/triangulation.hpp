#ifndef RGBDSLAM_UTILS_TRIANGULATION_HPP
#define RGBDSLAM_UTILS_TRIANGULATION_HPP

#include "coordinates/point_coordinates.hpp"
#include "utils/pose.hpp"

namespace rgbd_slam::tracking {

/**
 * \brief Triangulate 2D point (screen space) to find 3D correspondances (camera space)
 */
class Triangulation
{
  public:
    /**
     * \brief Triangulate a world point from two successive 2D matches, with an already known pose
     *
     * \param[in] currentWorldToCamera The world to camera transform matrix of the current optimised pose of the
     * observer
     * \param[in] newWorldToCamera The world to camera transform matrix of the new observer pose, already optimized
     * \param[in] point2Da A 2D point observed with the reference pose
     * \param[in] point2Db A 2D point, observed with the new pose, matched with point2Da
     * \param[out] triangulatedPoint The 3D world point, if this function returned true
     *
     * \return True is the triangulation was successful
     */
    [[nodiscard]] static bool triangulate(const WorldToCameraMatrix& currentWorldToCamera,
                                          const WorldToCameraMatrix& newWorldToCamera,
                                          const ScreenCoordinate2D& point2Da,
                                          const ScreenCoordinate2D& point2Db,
                                          WorldCoordinate& triangulatedPoint) noexcept;

    /**
     * \brief Return a weak supposition of a new pose, from an optimized pose
     */
    [[nodiscard]] static utils::Pose get_supposed_pose(const utils::Pose& pose,
                                                       const double baselinePoseSupposition) noexcept;

  private:
    /**
     * \brief Indicates if a retroprojection is valid (eg: under a threshold)
     *
     * \return True if the retroprojection is valid
     */
    [[nodiscard]] static bool is_retroprojection_valid(const WorldCoordinate& worldPoint,
                                                       const ScreenCoordinate2D& screenPoint,
                                                       const WorldToCameraMatrix& worldToCamera,
                                                       const double maximumRetroprojectionErrorSqr_px) noexcept;
};

} // namespace rgbd_slam::tracking

#endif
