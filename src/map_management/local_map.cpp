#include "local_map.hpp"

#include "../parameters.hpp"
#include "../tracking/triangulation.hpp"
#include "../utils/camera_transformation.hpp"
#include "../utils/covariances.hpp"
#include "../utils/coordinates.hpp"
#include "../utils/random.hpp"
#include "../outputs/logger.hpp"
#include "map_point.hpp"
#include "map_primitive.hpp"
#include "matches_containers.hpp"
#include "pose.hpp"
#include "shape_primitives.hpp"
#include "types.hpp"
#include <iostream>

namespace rgbd_slam {
    namespace map_management {


        /**
         * LOCAL UTILS FUNCTIONS
         */

        /**
         * PUBLIC
         */

        Local_Map::Local_Map()
        {
            // Check constants
            assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);

            _mapWriter = new outputs::XYZ_Map_Writer("out");
        }

        Local_Map::~Local_Map()
        {
            delete _mapWriter;
        }

        const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features(const utils::Pose& lastPose) const
        {
            const size_t numberOfNewKeypoints = _localPointMap.get_local_map_size() + _localPointMap.get_staged_map_size();

            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(lastPose.get_orientation_quaternion(), lastPose.get_position());

            // initialize output structure
            features::keypoints::KeypointsWithIdStruct keypointsWithIds; 

            // TODO: check the efficiency gain of those reserve calls
            keypointsWithIds.reserve(numberOfNewKeypoints);

            const static uint refreshFrequency = Parameters::get_keypoint_refresh_frequency() * 2;
            _localPointMap.get_tracked_features(worldToCamera, keypointsWithIds, refreshFrequency);
            return keypointsWithIds;
        }

        matches_containers::matchContainer Local_Map::find_feature_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypointsObject, const features::primitives::plane_container& detectedPlanes)
        {
            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            matches_containers::matchContainer matchSets;

            matchSets._points = _localPointMap.get_matches(detectedKeypointsObject, worldToCamera);
            matchSets._planes = find_plane_matches(currentPose, detectedPlanes);

            return matchSets;
        }

        void Local_Map::update(const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::plane_container& detectedPlanes, const matches_containers::match_point_container& outlierMatchedPoints, const matches_containers::match_plane_container& outlierMatchedPlanes)
        {
            // TODO find a better way to display trajectory than just a new map point
            // _mapWriter->add_point(optimizedPose.get_position());

            // Unmatch detected outliers
            mark_outliers_as_unmatched(outlierMatchedPoints);
            mark_outliers_as_unmatched(outlierMatchedPlanes);

            const matrix33& poseCovariance = utils::compute_pose_covariance(optimizedPose);
            const cameraToWorldMatrix& cameraToWorld = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            _localPointMap.update_map(cameraToWorld, poseCovariance, keypointObject);

            // add planes to local map
            update_local_plane_map(cameraToWorld, detectedPlanes);

            const bool addAllFeatures = false;  // only add unmatched features
            add_features_to_map(poseCovariance, cameraToWorld, keypointObject, detectedPlanes, addAllFeatures);

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::add_features_to_map(const matrix33& poseCovariance, const cameraToWorldMatrix& cameraToWorld, const features::keypoints::Keypoint_Handler& keypointObject, const features::primitives::plane_container& detectedPlanes, const bool addAllFeatures)
        {
            _localPointMap.add_features_to_staged_map(poseCovariance, cameraToWorld, keypointObject, addAllFeatures);

            // Add unmatched poins to the staged map, to unsure tracking of new features
            add_planes_to_map(cameraToWorld, detectedPlanes, addAllFeatures);
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map
        }

        void Local_Map::update_no_pose()
        {
            // add local map points
            _localPointMap.update_with_no_tracking();

            // add planes to local map
            update_local_plane_map_with_tracking_lost();
        }

        void Local_Map::reset()
        {
            _localPointMap.reset();
        }



        /**
         * PROTECTED
         */

        bool Local_Map::find_match(MapPlane& mapPlane, const features::primitives::plane_container& detectedPlanes, const worldToCameraMatrix& w2c, const planeWorldToCameraMatrix& worldToCamera, matches_containers::match_plane_container& matchedPlanes)
        {
            // start by flagging it as unmatched
            mapPlane._matchedPlane.mark_unmatched();

            // project plane in camera space
            const utils::PlaneCameraCoordinates& projectedPlane = mapPlane.get_parametrization().to_camera_coordinates(worldToCamera);
            const utils::CameraCoordinate& planeCentroid = mapPlane.get_centroid().to_camera_coordinates(w2c);
            const vector6& descriptor = features::primitives::Plane::compute_descriptor(projectedPlane, planeCentroid, mapPlane.get_contained_pixels());

            double smallestSimilarity = std::numeric_limits<double>::max();
            size_t selectedIndex = 0;

            const size_t detectedPlaneSize = detectedPlanes.size();
            for(size_t planeIndex = 0; planeIndex < detectedPlaneSize; ++planeIndex)
            {
                if (_isPlaneMatched[planeIndex])
                    // Does not allow multiple removal of a single match
                    // TODO: change this
                    continue;

                const features::primitives::Plane& shapePlane = detectedPlanes[selectedIndex];
                const double descriptorSim  = shapePlane.get_similarity(descriptor);
                if (descriptorSim < smallestSimilarity)// and shapePlane.is_similar(mapPlane.get_mask(), projectedPlane))
                {
                    selectedIndex = planeIndex;
                    smallestSimilarity = descriptorSim;
                }
            }

            if(false and smallestSimilarity < 0.2)
            {
                const features::primitives::Plane& shapePlane = detectedPlanes[selectedIndex];
                mapPlane._matchedPlane.mark_matched(selectedIndex);
                // TODO: replace nullptr by the plane covariance in camera space
                matchedPlanes.emplace(matchedPlanes.end(), shapePlane.get_parametrization(), mapPlane.get_parametrization(), nullptr, mapPlane._id);

                _isPlaneMatched[selectedIndex] = true;
                return true;
            }

            return false;
        }

        void Local_Map::update_local_plane_map(const cameraToWorldMatrix& cameraToWorld, const features::primitives::plane_container& detectedPlanes)
        {
            std::set<size_t> planesToRemove;
            const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

            // Update planes
            for (auto& [planeId, mapPlane] : _localPlaneMap)
            {
                if (mapPlane._matchedPlane.is_matched())
                {
                    const int matchedPlaneIndex = mapPlane._matchedPlane.get_match_index();
                    assert(matchedPlaneIndex != UNMATCHED_PRIMITIVE_ID);
                    assert(matchedPlaneIndex >= 0);
                    assert(static_cast<size_t>(matchedPlaneIndex) < detectedPlanes.size());

                    const features::primitives::Plane& detectedPlane = detectedPlanes[matchedPlaneIndex];
                    mapPlane.update(detectedPlane, planeCameraToWorld, cameraToWorld);
                }
                else
                {
                    mapPlane.update_unmatched();
                    if (mapPlane.is_lost())
                    {
                        // add to planes to remove
                        planesToRemove.emplace(planeId);
                    }
                }
            }

            // Remove umatched
            for(const size_t planeId : planesToRemove)
            {
                _localPlaneMap.erase(planeId);
            }
        }

        void Local_Map::update_local_plane_map_with_tracking_lost()
        {
            std::set<size_t> planesToRemove;
            // Update planes
            for (auto& [planeId, mapPlane] : _localPlaneMap)
            {
                mapPlane.update_unmatched();
                if (mapPlane.is_lost())
                {
                    // add to planes to remove
                    planesToRemove.emplace(planeId);
                }
            }

            // Remove umatched
            for(const size_t planeId : planesToRemove)
            {
                _localPlaneMap.erase(planeId);
            }
        }

        void Local_Map::add_planes_to_map(const cameraToWorldMatrix& cameraToWorld, const features::primitives::plane_container& detectedPlanes, const bool addAllFeatures)
        {
            const planeCameraToWorldMatrix& planeCameraToWorld = utils::compute_plane_camera_to_world_matrix(cameraToWorld);

            // add unmatched planes to local map
            const size_t detectedplaneCount = _isPlaneMatched.size();
            assert(detectedplaneCount == detectedPlanes.size());
            for(size_t planeIndex = 0; planeIndex < detectedplaneCount; ++planeIndex)
            {
                // Add all features, or add only the unmatched plane
                if (addAllFeatures or not _isPlaneMatched[planeIndex])
                {
                    const features::primitives::Plane& detectedPlane = detectedPlanes[planeIndex];

                    const MapPlane newMapPlane(
                        detectedPlane.get_parametrization().to_world_coordinates(planeCameraToWorld),
                        detectedPlane.get_centroid().to_world_coordinates(cameraToWorld),
                        detectedPlane.get_shape_mask()
                    );
                    _localPlaneMap.emplace(newMapPlane._id, newMapPlane);
                }
            }
        }

        void Local_Map::draw_planes_on_image(const worldToCameraMatrix& worldToCamera, cv::Mat& debugImage) const
        {
            assert(not debugImage.empty());

            const cv::Size& debugImageSize = debugImage.size();
            const uint imageWidth = debugImageSize.width;

            const double maskAlpha = 0.3;
            // 20 pixels
            const uint bandSize = 20;
            const uint placeInBand = bandSize * 0.75;

            std::stringstream textPoints;
            textPoints << "Points:" << _localPointMap.get_staged_map_size() << "|" << _localPointMap.get_local_map_size();
            const int plointLabelPosition = static_cast<int>(imageWidth * 0.15);
            cv::putText(debugImage, textPoints.str(), cv::Point(plointLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            std::stringstream text1;
            const double planeOffset = 0.35;
            text1 << "Planes:";
            const int planeLabelPosition = static_cast<int>(imageWidth * planeOffset);
            cv::putText(debugImage, text1.str(), cv::Point(planeLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            std::stringstream text2;
            const double cylinderOffset = 0.70;
            text2 << "Cylinders:";
            const int cylinderLabelPosition = static_cast<int>(imageWidth * 0.70);
            cv::putText(debugImage, text2.str(), cv::Point(cylinderLabelPosition, placeInBand), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

            // Tracking variables
            uint cylinderCount = 0;
            uint planeCount = 0;
            std::set<size_t> alreadyDisplayedIds;

            if (_localPlaneMap.size() == 0)
                return;

            cv::Mat allPlaneMasks = cv::Mat::zeros(debugImageSize, debugImage.type());
            for(const auto& [planeId, mapPlane]: _localPlaneMap)
            {
                if (not mapPlane._matchedPlane.is_matched())
                    continue;

                const cv::Scalar& planeColor = mapPlane.get_color();

                cv::Mat planeMask;
                // Resize with no interpolation
                cv::resize(mapPlane.get_mask() * 255, planeMask, debugImageSize, 0, 0, cv::INTER_NEAREST);
                cv::cvtColor(planeMask, planeMask, cv::COLOR_GRAY2BGR);
                assert(planeMask.size == debugImage.size);
                assert(planeMask.type() == debugImage.type());

                // merge with debug image
                planeMask.setTo(planeColor, planeMask);
                allPlaneMasks += planeMask;

                // Add color codes in label bar
                if (alreadyDisplayedIds.contains(planeId))
                    continue;   // already shown

                alreadyDisplayedIds.insert(planeId);

                double labelPosition = imageWidth;
                uint finalPlaceInBand = placeInBand;

                // plane
                labelPosition *= planeOffset;
                finalPlaceInBand *= planeCount;
                ++planeCount;

                    /*case features::primitives::PrimitiveType::Cylinder:
                        labelPosition *= cylinderOffset;
                        finalPlaceInBand *= cylinderCount;
                        ++cylinderCount;
                        break;*/

                if (labelPosition >= 0)
                {
                    // make a
                    const uint labelSquareSize = bandSize * 0.5;
                    cv::rectangle(debugImage, 
                            cv::Point(static_cast<int>(labelPosition + 80 + finalPlaceInBand), 6),
                            cv::Point(static_cast<int>(labelPosition + 80 + labelSquareSize + finalPlaceInBand), 6 + labelSquareSize), 
                            planeColor,
                            -1);
                }
            }

            cv::addWeighted(debugImage, (1 - maskAlpha), allPlaneMasks, maskAlpha, 0.0, debugImage);
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, const bool shouldDisplayPlaneMasks, cv::Mat& debugImage)  const
        {
            const worldToCameraMatrix& worldToCamMatrix = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            // Display planes
            if (shouldDisplayPlaneMasks)
                draw_planes_on_image(worldToCamMatrix, debugImage);

            _localPointMap.draw_on_image(worldToCamMatrix, debugImage, shouldDisplayStaged);
        }

        matches_containers::match_plane_container Local_Map::find_plane_matches(const utils::Pose& currentPose, const features::primitives::plane_container& detectedPlanes)
        {
            _isPlaneMatched = vectorb::Zero(detectedPlanes.size());

            // Compute a world to camera transformation matrix
            const worldToCameraMatrix& worldToCamera = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());
            const planeWorldToCameraMatrix& planeWorldToCamera = utils::compute_plane_world_to_camera_matrix(worldToCamera);

            // Search for matches
            matches_containers::match_plane_container matchedPlaneContainer;
            for(auto& [planeId, mapPlane] : _localPlaneMap)
            {
                // start by resetting the match status 
                mapPlane._matchedPlane.mark_unmatched();
                find_match(mapPlane, detectedPlanes, worldToCamera, planeWorldToCamera, matchedPlaneContainer);
            }

            return matchedPlaneContainer;
        }


        void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_point_container& outlierMatchedPoints)
        {
            // Mark outliers as unmatched
            for (const matches_containers::PointMatch& match : outlierMatchedPoints)
            {
                const bool isOutlierRemoved = _localPointMap.mark_feature_with_id_as_unmatched(match._idInMap);
                // If no points were found, this is bad. A match marked as outliers must be in the local map or staged points
                if(not isOutlierRemoved)
                {
                    outputs::log_error("Could not find the target point with id " + std::to_string(match._idInMap));
                }
            }
        }

        void Local_Map::mark_outliers_as_unmatched(const matches_containers::match_plane_container& outlierMatchedPlanes)
        {
            // Mark outliers as unmatched
            for (const matches_containers::PlaneMatch& match : outlierMatchedPlanes)
            {
                // Check if id is in local map
                const size_t planeId = match._idInMap;
                plane_map_container::iterator planeMapIterator = _localPlaneMap.find(planeId);
                if (planeMapIterator != _localPlaneMap.end())
                {
                    MapPlane& mapPlane = planeMapIterator->second;
                    assert(mapPlane._id == planeId);
                    // add detected plane id to unmatched set
                    _isPlaneMatched[mapPlane._matchedPlane.get_match_index()] = false;
                    // unmatch
                    mapPlane._matchedPlane.mark_unmatched();
                }
                else
                {
                    outputs::log_error("Could not find the target plane with id " + std::to_string(planeId));
                }
            }
        }

    }   /* map_management */
}   /* rgbd_slam */
