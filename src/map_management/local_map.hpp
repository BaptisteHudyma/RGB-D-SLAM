#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "map_point.hpp"
#include "map_primitive.hpp"

#include "../types.hpp"

#include "../outputs/map_writer.hpp"

#include "../utils/pose.hpp"
#include "../utils/matches_containers.hpp"

#include "../features/keypoints/keypoint_detection.hpp"
#include "../features/primitives/primitive_detection.hpp"



namespace rgbd_slam {
    namespace map_management {

        /**
         * \brief Maintain a local map around the camera. Can return matched features, and update the global map when features are estimated to be reliable. For now we dont have a global map
         */
        class Local_Map {
            public:
                Local_Map();
                ~Local_Map();

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
                 * \param[in] previousPose The clean true pose of the observer, before the new measurements
                 * \param[in] optimizedPose The clean true pose of the observer, after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_keypoint_matches
                 * \param[in] detectedPrimitives A container for all detected primitives in the depth image
                 * \param[in] outlierMatchedPoints A container for all the wrongly associated points detected in the pose optimization process. They should be marked as invalid matches
                 */
                void update(const utils::Pose& previousPose, const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::primitive_container& detectedPrimitives, const matches_containers::match_point_container& outlierMatchedPoints);

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
                 * \param[in] shouldDisplayStaged If true, will also display the content of the staged keypoint map
                 * \param[in] shouldDisplayPrimitiveMasks If true, will also display the primitives in local map
                 * \param[in, out] debugImage Output image
                 */
                void get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, const bool shouldDisplayPrimitiveMasks, cv::Mat& debugImage) const;

            protected:

                // Define types

                // local map point container
                typedef std::map<size_t, Map_Point> point_map_container;
                // staged points container
                typedef std::map<size_t, Staged_Point> staged_point_container;
                // local shape primitive map container
                typedef std::map<size_t, Primitive> primitive_map_container; 


                /**
                 * \brief Compute a match for a given point, and update this point match index. It will update the _isPointMatched object if a point is matched
                 *
                 * \param[in, out] point A map point that we want to match to detected points
                 * \param[in] detectedKeypointsObject An object to handle all detected points in an image
                 * \param[in] worldToCamMatrix A matrix to transform a world point to a camera point
                 * \param[in, out] matchedPoints A container associating the detected to the map points
                 * \param[in] shouldAddMatchToContainer If this flag is false, this point will not be inserted in matchedPoints.
                 *
                 * \return A boolean indicating if this point was matched or not
                 */
                bool find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const matrix44& worldToCamMatrix, matches_containers::match_point_container& matchedPoints, const bool shouldAddMatchToContainer=true);

                /**
                 * \brief Compute a match for a given primitive, and update this primitive match status.
                 *
                 * \param[in, out] mapPrimitive A map primitive  that we want to match to a detected primitive
                 * \param[in] detectedPrimitives A container that stores all the detected primitives
                 * \param[in] worldToCameraMatrix A matrix to transform a world point to a camera point
                 * \param[in, out] matchedPrimitives A container associating the detected primitives to the map primitives
                 *
                 * \return A boolean indicating if this primitive was matched or not
                 */
                bool find_match(Primitive& mapPrimitive, const features::primitives::primitive_container& detectedPrimitives, const matrix44& worldToCameraMatrix, matches_containers::match_primitive_container& matchedPrimitives);

                /**
                 * \brief Update the Matched/Unmatched status of a map point
                 *
                 * \param[in, out] mapPoint the map point to update
                 * \param[in] poseCovariance The covariance matrix of the optimized position of the observer
                 * \param[in] keypointObject An object to handle all detected points in an image
                 * \param[in] previousCameraToWorldMatrix A transformation matrix to convert a camera point to a world point. It represent the last optimized camera pose
                 * \param[in] cameraToWorldMatrix A transformation matrix to convert a camera point to a world point
                 */
                void update_point_match_status(IMap_Point_With_Tracking& mapPoint, const matrix33& poseCovariance, const features::keypoints::Keypoint_Handler& keypointObject, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix);

                /**
                 * \brief Update local keypoint map features 
                 *
                 * \param[in] poseCovariance The covariance matrix of the optimized position of the observer
                 * \param[in] previousCameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represents the last pose after optimization 
                 * \param[in] cameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz) It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_local_keypoint_map(const matrix33& poseCovariance, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Update the local primitive map features
                 *
                 * \param[in] previousCameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represents the last pose after optimization 
                 * \param[in] cameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz) It represent the current pose after optimization
                 * \param[in] A container that stores the detected primitives in the depth image
                 */
                void update_local_primitive_map(const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::primitives::primitive_container& detectedPrimitives);

                /**
                 * \brief Add previously uncertain keypoint features to the local map
                 *
                 * \param[in] poseCovariance The covariance matrix of the optimized position of the observer
                 * \param[in] previousCameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the last pose after optimization
                 * \param[in] cameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void update_staged_keypoints_map(const matrix33& poseCovariance, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject);

                /**
                 * \brief Add unmatched detected points to the staged map
                 *
                 * \param[in] poseCovariance The covariance matrix of the optimized position of the observer
                 * \param[in] cameraToWorldMatrix A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the current pose after optimization
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_matches
                 */
                void add_umatched_keypoints_to_staged_map(const matrix33& poseCovariance, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject);

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
                 * \param[out] debugImage The image to draw the points modify
                 */
                static void draw_point_on_image(const IMap_Point_With_Tracking& mapPoint, const matrix44& worldToCameraMatrix, const cv::Scalar& pointColor, cv::Mat& debugImage);

                void draw_primitives_on_image(const matrix44& worldToCameraMatrix, cv::Mat& debugImage) const;


                /**
                 * \brief Mark all the outliers detected during optimization as unmatched
                 */
                void mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints);

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
                 */
                void mark_point_with_id_as_unmatched(const size_t pointId, IMap_Point_With_Tracking& point);

            private:
                // Local map contains world points with a good confidence
                point_map_container _localPointMap;
                // Staged points are potential new map points, waiting to confirm confidence
                staged_point_container _stagedPoints;
                // Hold unmatched detected point indexes, to add in the staged point container
                std::vector<bool> _isPointMatched;
                // Hold unmatched primitive ids
                std::set<uchar> _unmatchedPrimitiveIds;

                //local primitive map
                primitive_map_container _localPrimitiveMap;
                std::unordered_map<int, uint> _previousPrimitiveAssociation;

                outputs::XYZ_Map_Writer* _mapWriter; 

                // Remove copy operators
                Local_Map(const Local_Map& map) = delete;
                void operator=(const Local_Map& map) = delete;
        };

    }
}

#endif
