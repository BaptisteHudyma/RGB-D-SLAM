#ifndef SLAM_LOCAL_MAP_HPP
#define SLAM_LOCAL_MAP_HPP 

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "types.hpp"
#include "matches_containers.hpp"
#include "map_point.hpp"
#include "KeyPointDetection.hpp"
#include "PrimitiveDetection.hpp"
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
                matches_containers::match_point_container find_keypoint_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypoint); 

                /**
                 * \brief Compute the primitive matches
                 *
                 * \param[in] currentPose The current observer pose.
                 * \param[in] detectedPrimitives The primitives (planes, cylinders, ...) detected in the image
                 *
                 * \return A container associating the map primitives to detected primitives
                 */
                matches_containers::match_primitive_container find_primitive_matches(const utils::Pose& currentPose, const features::primitives::primitive_container& detectedPrimitives);


                /**
                 * \brief Update the local and global map. Add new points to staged and map container
                 *
                 * \param[in] optimizedPose The clean true pose of the observer, after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_keypoint_matches
                 */
                void update(const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Return an object containing the tracked keypoint features in screen space (2D), with the associated global ids 
                 *
                 * \param[in] pose The current pose of the observer
                 */
                const features::keypoints::KeypointsWithIdStruct get_tracked_keypoints_features() const;

                /**
                 * \brief Hard clean the local and staged map
                 */
                void reset();


                /**
                 * \brief Compute a debug image to display the keypoints & primitives
                 *
                 * \param[in] camPose Pose of the camera in world coordinates
                 * \param[in, out] debugImage Output image
                 */

                void get_debug_image(const utils::Pose& camPose, cv::Mat& debugImage) const;

            protected:

                /**
                 * \brief Compute a match for a given point, and update this point match index. It will update the _isPointMatched object if a point is matched
                 *
                 * \param[in, out] point A map point that we want to match to detected points
                 * \param[in] detectedKeypoint An object to handle all detected points in an image
                 * \param[in] worldToCamMatrix A matrix to transform a world point to a camera point
                 * \param[int, out] matchedPoints A container associating the detected to the map points
                 *
                 * \return A boolean indicating if this point was matched or not
                 */
                bool find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypoint, const matrix34& worldToCamMatrix, matches_containers::match_point_container& matchedPoints);

                /**
                  * \brief Update the Matched/Unmatched status of a map point
                  *
                  * \param[in, out] mapPoint the map point to update
                  * \param[in] keypointObject An object to handle all detected points in an image
                  * \param[in] camToWorldMatrix A transformation matrix to convert a camera point to a world point
                  */
                void update_point_match_status(IMap_Point_With_Tracking& mapPoint, const features::keypoints::Keypoint_Handler& keypointObject, const matrix34& camToWorldMatrix);

                /**
                 * \brief Update local keypoint map features 
                 *
                 * \param[in] camToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz)
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_local_keypoint_map(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Add previously uncertain keypoint features to the local map
                 *
                 * \param[in] camToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz)
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_staged_keypoints_map(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Clean the local map so it stays local, and update the global map with the good features
                 */
                void update_local_to_global();

            private:
                // local map point container
                typedef std::list<Map_Point> point_map_container;
                // staged points container
                typedef std::list<Staged_Point> staged_point_container;
                // local shape primitive map container
                typedef std::list<features::primitives::primitive_uniq_ptr> primitive_map_container; 

                // Local map contains world points with a good confidence
                point_map_container _localPointMap;
                // Staged points are potential new map points, waiting to confirm confidence
                staged_point_container _stagedPoints;
                // Hold unmatched detected point indexes, to add in the staged point container
                std::vector<bool> _isPointMatched;

                //local primitive map
                primitive_map_container _localPrimitiveMap;
        };

    }
}

#endif
