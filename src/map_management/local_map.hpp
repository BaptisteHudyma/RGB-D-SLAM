#ifndef RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP
#define RGBDSLAM_MAPMANAGEMENT_LOCALMAP_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "map_point.hpp"
#include "map_primitive.hpp"

#include "../outputs/map_writer.hpp"

#include "../utils/pose.hpp"
#include "../utils/matches_containers.hpp"

#include "../features/keypoints/keypoint_handler.hpp"
#include "../features/primitives/shape_primitives.hpp"

#include "feature_map.hpp" 


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
                 * \brief Add features to staged map
                 * \param[in] poseCovariance The pose covariance of the observer, after optimization
                 * \param[in] cameraToWorld The matrix to go from camera to world space
                 * \param[in] keypointObject An object containing the detected key points in the rgbd frame. Must be the same as in find_keypoint_matches
                 * \param[in] detectedPlanes A container for all detected planes in the depth image
                 * \param[in] addAllFeatures If false, will add all non matched features, if true, add all features regardless of the match status
                 */
                void add_features_to_map(const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::plane_container& detectedPlanes, const bool addAllFeatures);

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
                 * \brief Compute the plane matches
                 *
                 * \param[in] currentPose The current observer pose.
                 * \param[in] detectedPlanes The planes detected in the image
                 *
                 * \return A container associating the map planes to detected planes
                 */
                matches_containers::match_plane_container find_plane_matches(const utils::Pose& currentPose, const features::primitives::plane_container& detectedPlanes);

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
                bool find_match(MapPlane& mapPlane, const features::primitives::plane_container& detectedPlanes, const worldToCameraMatrix& w2c, const planeWorldToCameraMatrix& worldToCamera, matches_containers::match_plane_container& matchedPlanes);


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
                 * \brief Add unmatched detected planes to the map
                 * \param[in] cameraToWorld A transformation matrix to go from a screen point (UVD) to a 3D world point (xyz). It represent the current pose after optimization
                 * \param[in] detectedPlanes An object containing the detected planes in the depth frame. Must be the same as in find_matches
                 * \param[in] addAllFeatures If false, will add all non matched features, if true, add all features regardless of the match status
                 */
                void add_planes_to_map(const cameraToWorldMatrix& cameraToWorld, const features::primitives::plane_container& detectedPlanes, const bool addAllFeatures);

                /**
                 * \brief Clean the local map so it stays local, and update the global map with the good features
                 */
                void update_local_to_global();


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

            private:
                // Define types

                typedef Feature_Map<LocalMapPoint, StagedMapPoint, features::keypoints::Keypoint_Handler, features::keypoints::DetectedKeyPoint, matches_containers::PointMatch, features::keypoints::KeypointsWithIdStruct> localPointMap;
                localPointMap _localPointMap;

                // local shape plane map container
                typedef std::unordered_map<size_t, MapPlane> plane_map_container; 


                //local plane map
                plane_map_container _localPlaneMap;
                // Hold unmatched plane ids, to add to the local map
                vectorb _isPlaneMatched;

                outputs::XYZ_Map_Writer* _mapWriter; 

                // Remove copy operators
                Local_Map(const Local_Map& map) = delete;
                void operator=(const Local_Map& map) = delete;
        };

    }
}

#endif
