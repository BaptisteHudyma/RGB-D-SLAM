#ifndef SLAM_LOCAL_MAP_HPP
#define SLAM_LOCAL_MAP_HPP 

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "types.hpp"
#include "map_point.hpp"
#include "KeyPointDetection.hpp"
#include "Pose.hpp"


namespace rgbd_slam {
    namespace map_management {

        /**
         * \brief Maintain a local map around the camera. Can return matched features, and update the global map when features are estimated to be reliable. For now we dont have a global map
         */
        class Local_Map {
            public:
                Local_Map();

                /**
                 * \brief Compute the point feature matches between the local map and a given set of points. Update the staged point list matched points
                 *
                 * \param[in] currentPose The current observer pose.
                 * \param[in] detectedKeypoint An object containing the detected key points in the rgbd frame
                 *
                 * \return A container associating the map/staged points to detected key points
                 */
                match_point_container find_matches(const poseEstimation::Pose currentPose, const utils::Keypoint_Handler& detectedKeypoint); 

                /**
                 * \brief Update the local and global map. Add new points to staged and map container 
                 *
                 * \param[in] optimizedPose The clean true pose of the observer, after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject);


                /**
                 * \brief Hard clean the local and staged map
                 */
                void reset();


                /**
                 * \brief Compute a debug image
                 *
                 * \param[in] camPose Pose of the camera in world coordinates
                 * \param[in, out] debugImage Output image
                 */

                void get_debug_image(const poseEstimation::Pose& camPose, cv::Mat& debugImage) const;

            protected:
                /**
                 * \brief Add previously uncertain features to the local map
                 *
                 * \param[in] camToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz)
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_staged(const matrix34& camToWorldMatrix, const utils::Keypoint_Handler& keypointObject);

                /**
                 * \brief Clean the local map so it stays local, and update the global map with the good features
                 */
                void update_local_to_global();

            private:
                // local map point container
                typedef std::list<Map_Point> point_map_container;
                // staged points container
                typedef std::list<Staged_Point> staged_point_container;

                // Local map contains world points with a good confidence
                point_map_container _localMap;
                // Staged points are potential new map points, waiting to confirm confidence
                staged_point_container _stagedPoints;

                // Hold unmatched detected point indexes, to add in the staged point container
                std::vector<bool> _isPointMatched;

                //local primitive map


        };

    }
}

#endif
