#include "local_map.hpp"

#include "parameters.hpp"
#include "utils.hpp"

namespace rgbd_slam {
    namespace map_management {

        const double MIN_DEPTH_DISTANCE = 40;   // 40 millimeters is the camera minimum detection distance

        /**
          * LOCAL UTILS FUNCTIONS
          */

        /**
          * \brief compute a covariance matrix for a point associated with a depth measurement
          */
        matrix33 get_screen_point_covariance(const double depth) 
        {
            // Quadratic error model (uses depth as meters)
            const double depthMeters = depth / 1000;
            // If depth is less than the min distance, covariance is set to a high value
            const double depthVariance = (depth > MIN_DEPTH_DISTANCE) ? std::max(0.0, -0.58 + 0.74 * depthMeters + 2.73 * pow(depthMeters, 2.0)) : 1000.0;

            // TODO xy variance should also depend on the placement of the pixel in x and y
            const double xyVariance = 0.25; // 0.5 pixel error

            matrix33 screenPointCovariance {
                {xyVariance, 0,          0},
                    {0,          xyVariance, 0},
                    {0,          0,          depthVariance * depthVariance},
            };
            return screenPointCovariance;
        }


        /**
          * \brief My add a point to the tracked feature object, used to add optical flow tracking
          */
        void add_point_to_tracked_features(const IMap_Point_With_Tracking& mapPoint, const matrix34& worldToCamMatrix, features::keypoints::KeypointsWithIdStruct& keypointsWithIds)
        {
            if (mapPoint._lastMatchedIndex != UNMATCHED_POINT_INDEX)
            {
                const vector2& screenCoordinates = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMatrix);
                keypointsWithIds._keypoints.push_back(cv::Point2f(screenCoordinates.x(), screenCoordinates.y()));
                keypointsWithIds._ids.push_back(mapPoint._id);
            }
        }

        /**
          * LOCAL MAP MEMBERS
          */

        Local_Map::Local_Map()
        {
            // Check constants
            assert(features::keypoints::INVALID_MAP_POINT_ID == INVALID_POINT_UNIQ_ID);
        }

        bool Local_Map::find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypoint, const matrix34& worldToCamMatrix, matches_containers::match_point_container& matchedPoints)
        {
            int matchIndex = detectedKeypoint.get_tracking_match_index(point._id, _isPointMatched);
            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
            {
                const vector2& projectedMapPoint = utils::world_to_screen_coordinates(point._coordinates, worldToCamMatrix);
                matchIndex = detectedKeypoint.get_match_index(projectedMapPoint, point._descriptor, _isPointMatched);
            }

            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX) {
                //unmatched point
                point._lastMatchedIndex = UNMATCHED_POINT_INDEX;
            }
            else if (detectedKeypoint.get_depth(matchIndex) <= MIN_DEPTH_DISTANCE) {
                // 2D point, still matched
                _isPointMatched[matchIndex] = true;
                point._lastMatchedIndex = matchIndex;
                return true;
            }
            else {
                _isPointMatched[matchIndex] = true;
                point._lastMatchedIndex = matchIndex;
                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                const vector3 screen3DPoint(screenPoint.x(), screenPoint.y(), detectedKeypoint.get_depth(matchIndex));

                matchedPoints.emplace(matchedPoints.end(), screen3DPoint, point._coordinates);
                return true;
            }
            return false;
        }

        matches_containers::match_point_container Local_Map::find_keypoint_matches(const utils::Pose& currentPose, const features::keypoints::Keypoint_Handler& detectedKeypoint)
        {
            // will be used to detect new keypoints for the stagged map
            _isPointMatched.clear();
            _isPointMatched = std::vector<bool>(detectedKeypoint.get_keypoint_count(), false);
            matches_containers::match_point_container matchedPoints; 

            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            // Try to find matches in local map
            for (Map_Point& mapPoint : _localPointMap) 
            {
                find_match(mapPoint, detectedKeypoint, worldToCamMatrix, matchedPoints);
            }

            // Try to find matches in staged points
            for(Staged_Point& stagedPoint : _stagedPoints)
            {
                find_match(stagedPoint, detectedKeypoint, worldToCamMatrix, matchedPoints);
            }

            return matchedPoints;
        }

        matches_containers::match_primitive_container Local_Map::find_primitive_matches(const utils::Pose& currentPose, const features::primitives::primitive_container& detectedPrimitives)
        {
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            matches_containers::match_primitive_container matchedPrimitiveContainer;
            for(const features::primitives::primitive_uniq_ptr& mapPrimitive : _localPrimitiveMap)
            {
                for(const features::primitives::primitive_uniq_ptr& shapePrimitive : detectedPrimitives) 
                {
                    if(mapPrimitive->is_similar(shapePrimitive)) 
                    {
                        matchedPrimitiveContainer.emplace(matchedPrimitiveContainer.end(), shapePrimitive->_normal, mapPrimitive->_normal);
                        break;
                    }
                }
            }

            return matchedPrimitiveContainer;
        }


        void Local_Map::update(const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            const matrix34& camToWorldMatrix = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            update_local_keypoint_map(camToWorldMatrix, keypointObject);

            // add staged points to local map
            update_staged_keypoints_map(camToWorldMatrix, keypointObject);

            // add local map points to global map
            update_local_to_global();

            //std::cout << "local map: " << _localPointMap.size() << " | staged points: " << _stagedPoints.size() << std::endl;
        }

        void Local_Map::update_point_match_status(IMap_Point_With_Tracking& mapPoint, const features::keypoints::Keypoint_Handler& keypointObject, const matrix34& camToWorldMatrix)
        {
            const int matchedPointIndex = mapPoint._lastMatchedIndex;
            if (matchedPointIndex != UNMATCHED_POINT_INDEX)
            {
                const int keypointsSize = static_cast<int>(keypointObject.get_keypoint_count());
                if (matchedPointIndex >= 0 and matchedPointIndex < keypointsSize)
                {
                    // get match coordinates, transform them to world coordinates
                    const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                    const double matchedPointDepth = keypointObject.get_depth(matchedPointIndex);

                    if(matchedPointDepth > MIN_DEPTH_DISTANCE)
                    {
                        // TODO: close points cannot be reliably maintained
                        const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), matchedPointDepth, camToWorldMatrix);

                        const matrix33& worldPointCovariance = utils::get_world_point_covariance(matchedPointCoordinates, matchedPointDepth, get_screen_point_covariance(matchedPointDepth));
                        // update this map point errors & position
                        mapPoint.update_matched(newCoordinates, worldPointCovariance);

                        return;
                    }
                }
                else
                {
                    utils::log_error("Match point index is out of bounds of keypointObject");
                }
            }

            // point is unmatched
            mapPoint.update_unmatched();
        }

        void Local_Map::update_local_keypoint_map(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Remove old map points
            point_map_container::iterator pointMapIterator = _localPointMap.begin();
            while(pointMapIterator != _localPointMap.end())
            {
                // Update the matched/unmatched status
                update_point_match_status(*pointMapIterator, keypointObject, camToWorldMatrix);

                if (pointMapIterator->is_lost()) {
                    // Remove useless point
                    pointMapIterator = _localPointMap.erase(pointMapIterator);
                }
                else
                {
                    ++pointMapIterator;
                }
            }


        }

        void Local_Map::update_staged_keypoints_map(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                // Update the matched/unmatched status
                update_point_match_status(*stagedPointIterator, keypointObject, camToWorldMatrix);

                if (stagedPointIterator->should_add_to_local_map())
                {
                    // Add to local map, remove from staged points, with a copy of the id affected to the local map
                    _localPointMap.emplace(_localPointMap.end(), stagedPointIterator->_coordinates, stagedPointIterator->get_covariance_matrix(), stagedPointIterator->_descriptor, stagedPointIterator->_id);
                    _localPointMap.back()._lastMatchedIndex = stagedPointIterator->_lastMatchedIndex;
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else if (stagedPointIterator->should_remove_from_staged())
                {
                    // Remove from staged points
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else
                {
                    // Increment
                    ++stagedPointIterator;
                }
            }

            // Add all unmatched points to staged point container 
            const size_t keypointVectorSize = _isPointMatched.size();
            for(unsigned int i = 0; i < keypointVectorSize; ++i)
            {
                if (not _isPointMatched[i]) {
                    const double depth = keypointObject.get_depth(i);
                    if (depth <= MIN_DEPTH_DISTANCE)
                    {
                        // TODO handle 2D features
                        continue;
                    }

                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint.x(), screenPoint.y(), depth, camToWorldMatrix);

                    const matrix33& worldPointCovariance = utils::get_world_point_covariance(screenPoint, depth, get_screen_point_covariance(depth));
                    _stagedPoints.emplace(_stagedPoints.end(), worldPoint, worldPointCovariance, keypointObject.get_descriptor(i));
                }
            }
        }


        const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features(const utils::Pose& pose) const
        {
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(pose.get_orientation_quaternion(), pose.get_position());

            const size_t numberOfNewKeypoints = _localPointMap.size() + _stagedPoints.size();

            // initialize output structure
            features::keypoints::KeypointsWithIdStruct keypointsWithIds; 
            keypointsWithIds._ids.reserve(numberOfNewKeypoints);
            keypointsWithIds._keypoints.reserve(numberOfNewKeypoints);

            // add map points with valid retroprojected coordinates
            for (const Map_Point& point : _localPointMap)
            {
                add_point_to_tracked_features(point, worldToCamMatrix, keypointsWithIds);
            }
            // add staged points with valid retroprojected coordinates
            for (const Staged_Point& point : _stagedPoints)
            {
                add_point_to_tracked_features(point, worldToCamMatrix, keypointsWithIds);
            }
            return keypointsWithIds;
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

        void Local_Map::reset()
        {
            _localPointMap.clear();
            _stagedPoints.clear();
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, cv::Mat& debugImage)  const
        {
            const matrix34& worldToCamMtrx = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            for (const Map_Point& mapPoint : _localPointMap) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMtrx);

                //Map Point are green 
                if (mapPoint._lastMatchedIndex == UNMATCHED_POINT_INDEX)
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 128, 0), 1);
                }
                else
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 255, 0), 1);
                }
            }
            for (const Staged_Point& stagedPoint : _stagedPoints) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(stagedPoint._coordinates, worldToCamMtrx);

                //Staged point are yellow 
                if (stagedPoint._lastMatchedIndex == UNMATCHED_POINT_INDEX)
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 100, 200), 1);
                }
                else
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 200, 200), 1);
                }
            }
        }



    }   /* map_management */
}   /* rgbd_slam */
