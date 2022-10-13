#ifndef RGBDSLAM_UTILS_TRIANGULATION_HPP
#define RGBDSLAM_UTILS_TRIANGULATION_HPP

#include "../types.hpp"
#include "utils/pose.hpp"

namespace rgbd_slam {
    namespace tracking {

        class Triangulation
        {
            public:

                /**
                 * \brief Triangulate a world point from two successive 2D matches, with an already known pose
                 *
                 * \param[in] currentWorldToCamera The world to camera transform matrix of the current optimised pose of the observer
                 * \param[in] newWorldToCamera The world to camera transform matrix of the new observer pose, already optimized
                 * \param[in] point2Da A 2D point observed with the reference pose
                 * \param[in] point2Db A 2D point, observed with the new pose, matched with point2Da 
                 * \param[out] triangulatedPoint The 3D world point, if this function returned true
                 *
                 * \return True is the triangulation was successful
                 */
                static bool triangulate(const worldToCameraMatrix& currentWorldToCamera, const worldToCameraMatrix& newWorldToCamera, const vector2& point2Da, const vector2& point2Db, vector3& triangulatedPoint);

                /**
                 * \brief Return a weak supposition of a new pose, from an optimized pose
                 */
                static utils::Pose get_supposed_pose(const utils::Pose& pose, const double baselinePoseSupposition);

            private:
                
                /**
                  * \brief Indicates if a retroprojection is valid (eg: under a threshold)
                  *
                  * \return True if the retroprojection is valid
                  */
                static bool is_retroprojection_valid(const vector3& worldPoint, const vector2& screenPoint, const worldToCameraMatrix& worldToCamera, const double& maximumRetroprojectionError);

        };

    }
}

#endif
