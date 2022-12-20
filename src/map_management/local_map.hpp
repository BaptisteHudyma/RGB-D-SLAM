#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "map_point.hpp"
#include "map_primitive.hpp"

#include "../outputs/map_writer.hpp"

#include "../utils/pose.hpp"
#include "../utils/matches_containers.hpp"

#include "../features/keypoints/keypoint_handler.hpp"
#include "../features/primitives/primitive_detection.hpp"



namespace rgbd_slam {
    namespace map_management {

        /**
         * \brief Maintain a local (around the camera) map.
         * Handle the feature association and tracking in local space. 
         * Can return matched features, and update the global map when features are estimated to be reliable.
         */
        class Local_Map {
            public:
                Local_Map();
                ~Local_Map();

                /**
                 * \brief Return an object containing the tracked keypoint features in screen space (2D), with the associated global ids 
                 *
                 * \param[in] pose The current pose of the observer
                 */
                const features::keypoints::KeypointsWithIdStruct get_tracked_keypoints_features(const utils::Pose& lastpose) const;

                /**
                 * \brief Find all matches for the given detected features
                 * \param[in] currentPose The pose of the observer
                 * \param[in] detectedKeypointsObject An object that contains the detected keypoints in the new input
                 * \param[in] detectedPlanes An object that contains the detected planes in the new input
                 */
                matches_containers::matchContainer find_feature_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const features::primitives::plane_container& detectedPlanes);

                /**
                 * \brief Update the local and global map. Add new points to staged and map container
                 *
                 * \param[in] optimizedPose The clean true pose of the observer, after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_keypoint_matches
                 * \param[in] detectedPlanes A container for all detected planes in the depth image
                 * \param[in] outlierMatchedPoints A container for all the wrongly associated points detected in the pose optimization process. They should be marked as invalid matches
                 * \param[in] outlierMatchedPlanes A container for all the wrongly associated planes detected in the pose optimization process. They should be marked as invalid matches
                 */
                void update(const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::plane_container& detectedPlanes, const matches_containers::match_point_container& outlierMatchedPoints, const matches_containers::match_plane_container& outlierMatchedPlanes);

                /**
                 * \brief Update the local map when no pose could be estimated. Consider all features as unmatched
                 */
                void update_no_pose();

                /**
                 * \brief Hard clean the local and staged map
                 */
                void reset();


                /**
                 * \brief Compute a debug image to display the keypoints & planes
                 *
                 * \param[in] camPose Pose of the camera in world coordinates
                 * \param[in] shouldDisplayStaged If true, will also display the content of the staged keypoint map
                 * \param[in] shouldDisplayPlaneMasks If true, will also display the planes in local map
                 * \param[in, out] debugImage Output image
                 */
                void get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, const bool shouldDisplayPlaneMasks, cv::Mat& debugImage) const;

            protected:

                /**
                 * \brief Compute the point feature matches between the local map and a given set of points. Update the staged point list matched points
                 *
                 * \param[in] currentPose The current observer pose.
                 * \param[in] detectedKeypointsObject An object containing the detected key points in the rgbd frame
                 *
                 * \return A container associating the map/staged points to detected key points
                 */
                matches_containers::match_point_container find_keypoint_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject); 

                /**
                 * \brief Compute the plane matches
                 *
                 * \param[in] currentPose The current observer pose.
                 * \param[in] detectedPlanes The planes detected in the image
                 *
                 * \return A container associating the map planes to detected planes
                 */
                matches_containers::match_plane_container find_plane_matches(const utils::Pose& currentPose, const features::primitives::plane_container& detectedPlanes);


                /**
                 * \brief Compute a match for a given point, and update this point match index. It will update the _isPointMatched object if a point is matched
                 *
                 * \param[in, out] point A map point that we want to match to detected points
                 * \param[in] detectedKeypointsObject An object to handle all detected points in an image
                 * \param[in] worldToCamera A matrix to transform a world point to a camera point
                 * \param[in, out] matchedPoints A container associating the detected to the map points
                 * \param[in] shouldAddMatchToContainer If this flag is false, this point will not be inserted in matchedPoints.
                 *
                 * \return A boolean indicating if this point was matched or not
                 */
                bool find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const worldToCameraMatrix& worldToCamera, matches_containers::match_point_container& matchedPoints, const bool shouldAddMatchToContainer=true);

                /**
                 * \brief Compute a match for a given plane, and update this plane match status.
                 *
                 * \param[in, out] mapPlane A map plane that we want to match to a detected plane
                 * \param[in] detectedPlanes A container that stores all the detected planes
                 * \param[in] worldToCamera A matrix to transform a world plane to a camera plane
                 * \param[in, out] matchedPlanes A container associating the detected planes to the map planes
                 *
                 * \return A boolean indicating if this plane was matched or not
                 */
                bool find_match(MapPlane& mapPlane, const features::primitives::plane_container& detectedPlanes, const planeWorldToCameraMatrix& worldToCamera, matches_containers::match_plane_container& matchedPlanes);

                /**
                 * \brief Update the Matched/Unmatched status of a map point
                 *
                 * \param[in, out] mapPoint the map point to update
                 * \param[in] keypointObject An object to handle all detected points in an image
                 * \param[in] cameraToWorld A transformation matrix to convert a camera point to a world point
                 */
                void update_point_match_status(IMap_Point_With_Tracking& mapPoint, const features::keypoints::Keypoint_Handler& keypointObject, const cameraToWorldMatrix& cameraToWorld);

                /**
                 * \brief Update local keypoint map features 
                 *
                 * \param[in] cameraToWorld A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz) It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_local_keypoint_map(const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Update local keypoint map features when no optimized pose 
                 */
                void update_local_keypoint_map_with_tracking_lost();

                /**
                 * \brief Update the local plane map features
                 *
                 * \param[in] cameraToWorld A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz) It represent the current pose after optimization
                 * \param[in] detectedPlanes A container that stores the detected planes in the depth image
                 */
                void update_local_plane_map(const cameraToWorldMatrix& cameraToWorld, const features::primitives::plane_container& detectedPlanes);
                
                /**
                 * \brief Update the local plane map features when no pose optimization have been made
                 *
                 * \param[in] detectedPlanes A container that stores the detected planes in the depth image
                 */
                void update_local_plane_map_with_tracking_lost();

                /**
                 * \brief Add previously uncertain keypoint features to the local map
                 *
                 * \param[in] cameraToWorld A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_staged_keypoints_map(const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Remove unmtached staged keypoints that are too old
                 */
                void update_staged_keypoints_map_with_tracking_lost();

                /**
                 * \brief Add unmatched detected points to the staged map
                 *
                 * \param[in] poseCovariance The covariance matrix of the optimized position of the observer
                 * \param[in] cameraToWorld A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void add_umatched_keypoints_to_staged_map(const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Clean the local map so it stays local, and update the global map with the good features
                 */
                void update_local_to_global();


                /**
                 * \brief Draw a given map point on the given debug image
                 *
                 * \param[in] mapPoint The 3D world point
                 * \param[in] worldToCameraMatrix A matrix to transforme a world point to a camera point
                 * \param[in] pointColor The color of the point to draw
                 * \param[in] radius The radius of the point to display
                 * \param[out] debugImage The image to draw the points modify
                 */
                static void draw_point_on_image(const IMap_Point_With_Tracking& mapPoint, const worldToCameraMatrix& worldToCameraMatrix, const cv::Scalar& pointColor, cv::Mat& debugImage, const size_t radius = 3);

                void draw_planes_on_image(const worldToCameraMatrix& worldToCamera, cv::Mat& debugImage) const;


                /**
                 * \brief Mark all the outliers detected during optimization as unmatched
                 * \param[in] outlierMatchedPoints A container of the wrong matches detected after the optimization process
                 */
                void mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints);
                /**
                 * \brief Mark all the outliers detected during optimization as unmatched
                 * \param[in] outlierMatchedPlanes A container of the wrong matches detected after the optimization process
                 */
                void mark_outliers_as_unmatched(const matches_containers::match_plane_container& outlierMatchedPlanes);

                /**
                 * \brief Mark a point with the id pointId as unmatched. Will search the staged and local map.
                 *
                 * \param[in] pointId The uniq id of the point to unmatch
                 * 
                 * \return true if the point was found and updated
                 */
                bool mark_point_with_id_as_unmatched(const size_t pointId);

                /**
                 * \brief mark a point as unmatched
                 * \param[in] point The point to mark as unmatched
                 */
                void mark_point_with_id_as_unmatched(IMap_Point_With_Tracking& point);

            private:
                // Define types

                // local map point container
                typedef std::map<size_t, Map_Point> point_map_container;
                // staged points container
                typedef std::map<size_t, Staged_Point> staged_point_container;
                // local shape plane map container
                typedef std::map<size_t, MapPlane> plane_map_container; 

                // Local map contains world points with a good confidence
                point_map_container _localPointMap;
                // Staged points are potential new map points, waiting to confirm confidence
                staged_point_container _stagedPoints;
                // Hold unmatched detected point indexes, to add in the staged point container
                std::vector<bool> _isPointMatched;
                // Hold unmatched plane ids, to add to the local map
                std::set<uchar> _unmatchedPlaneIds;

                //local plane map
                plane_map_container _localPlaneMap;

                outputs::XYZ_Map_Writer* _mapWriter; 

                // Remove copy operators
                Local_Map(const Local_Map& map) = delete;
                void operator=(const Local_Map& map) = delete;
        };

    }
}

#endif
