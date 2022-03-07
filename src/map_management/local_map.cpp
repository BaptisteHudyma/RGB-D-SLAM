#include "local_map.hpp"

#include "parameters.hpp"
#include "triangulation.hpp"
#include "utils.hpp"

namespace rgbd_slam {
    namespace map_management {

        const double MIN_DEPTH_DISTANCE = 40;   // M millimeters is the depth camera minimum reliable distance
        const double MAX_DEPTH_DISTANCE = 6000; // N meters is the depth camera maximum reliable distance

        /**
         * LOCAL UTILS FUNCTIONS
         */

        /**
         * \brief checks the depth validity of a measurement
         */
        bool is_depth_valid(const double depth)
        {
            return (depth > MIN_DEPTH_DISTANCE and depth <= MAX_DEPTH_DISTANCE);
        }

        /**
         * \brief compute a covariance matrix for a point associated with a depth measurement
         */
        matrix33 get_screen_point_covariance(const double depth) 
        {
            // Quadratic error model (uses depth as meters)
            const double depthMeters = depth / 1000.0;
            // If depth is less than the min distance, covariance is set to a high value
            const double depthVariance = std::max(0.0001, is_depth_valid(depth) ? (-0.58 + 0.74 * depthMeters + 2.73 * pow(depthMeters, 2.0)) : 1000.0);
            // a zero variance will break the kalman gain
            assert(depthVariance > 0);

            // TODO xy variance should also depend on the placement of the pixel in x and y
            const double xyVariance = pow(0.1, 2.0);

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
        void add_point_to_tracked_features(const IMap_Point_With_Tracking& mapPoint, features::keypoints::KeypointsWithIdStruct& keypointsWithIds)
        {
            const vector3& coordinates = mapPoint._coordinates;
            assert(not isnan(coordinates.x()) and not isnan(coordinates.y()) and not isnan(coordinates.z()));
            if (mapPoint._lastMatchedIndex != UNMATCHED_POINT_INDEX)
            {
                // use previously known screen coordinates
                keypointsWithIds._keypoints.push_back(cv::Point2f(mapPoint._screenCoordinates.x(), mapPoint._screenCoordinates.y()));
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

        bool Local_Map::find_match(IMap_Point_With_Tracking& point, const features::keypoints::Keypoint_Handler& detectedKeypoint, const matrix44& worldToCamMatrix, matches_containers::match_point_container& matchedPoints)
        {
            int matchIndex = detectedKeypoint.get_tracking_match_index(point._id, _isPointMatched);
            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX)
            {
                vector2 projectedMapPoint;
                const bool isScreenCoordinatesValid = utils::world_to_screen_coordinates(point._coordinates, worldToCamMatrix, projectedMapPoint);
                if (isScreenCoordinatesValid)
                    matchIndex = detectedKeypoint.get_match_index(projectedMapPoint, point._descriptor, _isPointMatched);
            }

            if (matchIndex == features::keypoints::INVALID_MATCH_INDEX) {
                //unmatched point
                point._lastMatchedIndex = UNMATCHED_POINT_INDEX;
                return false;
            }

            const double screenPointDepth = detectedKeypoint.get_depth(matchIndex);
            if (is_depth_valid(screenPointDepth) ) {
                // points with depth measurement
                _isPointMatched[matchIndex] = true;

                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);

                // update index and screen coordinates 
                point._lastMatchedIndex = matchIndex;
                point._screenCoordinates << screenPoint, screenPointDepth;

                const vector3 screen3DPoint(screenPoint.x(), screenPoint.y(), screenPointDepth);
                matchedPoints.emplace(matchedPoints.end(), screen3DPoint, point._coordinates);
                return true;
            }
            else {
                // 2D point
                _isPointMatched[matchIndex] = true;

                // update index and screen coordinates 
                point._lastMatchedIndex = matchIndex;
                point._screenCoordinates << detectedKeypoint.get_keypoint(matchIndex), 0;
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

            const matrix44& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

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
            const matrix44& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

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


        void Local_Map::update(const utils::Pose& previousPose, const utils::Pose& optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            const matrix44& previousCameraToWorldMatrix = utils::compute_camera_to_world_transform(previousPose.get_orientation_quaternion(), previousPose.get_position());
            const matrix44& cameraToWorldMatrix = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            update_local_keypoint_map(previousCameraToWorldMatrix, cameraToWorldMatrix, keypointObject);

            // add staged points to local map
            update_staged_keypoints_map(previousCameraToWorldMatrix, cameraToWorldMatrix, keypointObject);

            // Add unmatched poins to the staged map, to unsure tracking of new features
            add_umatched_keypoints_to_staged_map(cameraToWorldMatrix, keypointObject);

            // add primitives to local map
            //update_local_primitive_map(cameraToWorldMatrix, );

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::update_point_match_status(IMap_Point_With_Tracking& mapPoint, const features::keypoints::Keypoint_Handler& keypointObject, const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix)
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

                    if(is_depth_valid(matchedPointDepth))
                    {
                        // transform screen point to world point
                        const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), matchedPointDepth, cameraToWorldMatrix);
                        // get a measure of the estimated variance of the new world point
                        const matrix33& worldPointCovariance = utils::get_world_point_covariance(matchedPointCoordinates, matchedPointDepth, get_screen_point_covariance(matchedPointDepth));
                        // TODO update the variances with the pose variance

                        // update this map point errors & position
                        mapPoint.update_matched(newCoordinates, worldPointCovariance);

                        // If a new descriptor is available, update it
                        if (keypointObject.is_descriptor_computed(matchedPointIndex))
                            mapPoint._descriptor = keypointObject.get_descriptor(matchedPointIndex);

                        // End of the function
                        return;
                    }
                    else
                    {
#if 0
                        // inefficient...
                        const matrix44& worldToCameraMatrix = utils::compute_world_to_camera_transform(previousCameraToWorldMatrix);
                        vector2 previousPointScreenCoordinates;
                        const bool isTransformationValid = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCameraMatrix, previousPointScreenCoordinates);
                        if (isTransformationValid)
                        {
                            vector3 triangulatedPoint;
                            const bool isTriangulationValid = utils::Triangulation::triangulate(previousCameraToWorldMatrix, cameraToWorldMatrix, previousPointScreenCoordinates, matchedPointCoordinates, triangulatedPoint);
                            // update the match
                            if (isTriangulationValid)
                            {
                                //std::cout << "udpate with triangulation " << triangulatedPoint.transpose() << " " << mapPoint._coordinates.transpose() << std::endl;
                                // get a measure of the estimated variance of the new world point
                                //const matrix33& worldPointCovariance = utils::get_triangulated_point_covariance(triangulatedPoint, get_screen_point_covariance(triangulatedPoint.z())));
                                const matrix33& worldPointCovariance = utils::get_world_point_covariance(vector2(triangulatedPoint.x(), triangulatedPoint.y()), triangulatedPoint.z(), get_screen_point_covariance(triangulatedPoint.z()));
                                // TODO update the variances with the pose variance

                                // update this map point errors & position
                                mapPoint.update_matched(triangulatedPoint, worldPointCovariance);

                                // If a new descriptor is available, update it
                                if (keypointObject.is_descriptor_computed(matchedPointIndex))
                                    mapPoint._descriptor = keypointObject.get_descriptor(matchedPointIndex);
                                return;
                            }
                        }
#endif
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

        void Local_Map::update_local_keypoint_map(const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            point_map_container::iterator pointMapIterator = _localPointMap.begin();
            while(pointMapIterator != _localPointMap.end())
            {
                // Update the matched/unmatched status
                update_point_match_status(*pointMapIterator, keypointObject, previousCameraToWorldMatrix, cameraToWorldMatrix);

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

        void Local_Map::update_staged_keypoints_map(const matrix44& previousCameraToWorldMatrix, const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                // Update the matched/unmatched status
                update_point_match_status(*stagedPointIterator, keypointObject, previousCameraToWorldMatrix, cameraToWorldMatrix);

                if (stagedPointIterator->should_add_to_local_map())
                {
                    const vector3& stagedPointCoordinates = stagedPointIterator->_coordinates;
                    assert(not isnan(stagedPointCoordinates.x()) and not isnan(stagedPointCoordinates.y()) and not isnan(stagedPointCoordinates.z()));
                    // Add to local map, remove from staged points, with a copy of the id affected to the local map
                    _localPointMap.emplace(_localPointMap.end(), stagedPointCoordinates, stagedPointIterator->get_covariance_matrix(), stagedPointIterator->_descriptor, stagedPointIterator->_id);
                    _localPointMap.back()._lastMatchedIndex = stagedPointIterator->_lastMatchedIndex;
                    _localPointMap.back()._screenCoordinates = stagedPointIterator->_screenCoordinates;
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
        }

        void Local_Map::add_umatched_keypoints_to_staged_map(const matrix44& cameraToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add all unmatched points to staged point container 
            const size_t keypointVectorSize = _isPointMatched.size();
            for(unsigned int i = 0; i < keypointVectorSize; ++i)
            {
                if (not _isPointMatched[i]) {
                    const double depth = keypointObject.get_depth(i);
                    if (not is_depth_valid(depth))
                    {
                        continue;
                    }

                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint.x(), screenPoint.y(), depth, cameraToWorldMatrix);
                    assert(not isnan(worldPoint.x()) and not isnan(worldPoint.y()) and not isnan(worldPoint.z()));

                    const matrix33& worldPointCovariance = utils::get_world_point_covariance(screenPoint, depth, get_screen_point_covariance(depth));
                    _stagedPoints.emplace(_stagedPoints.end(), worldPoint, worldPointCovariance, keypointObject.get_descriptor(i));
                    _stagedPoints.back()._screenCoordinates << screenPoint, depth;
                    // This id is to unsure the tracking of this staged point for it's first detection
                    _stagedPoints.back()._lastMatchedIndex = 0;
                }
            }

        }


        const features::keypoints::KeypointsWithIdStruct Local_Map::get_tracked_keypoints_features() const
        {
            const size_t numberOfNewKeypoints = _localPointMap.size() + _stagedPoints.size();

            // initialize output structure
            features::keypoints::KeypointsWithIdStruct keypointsWithIds; 
            keypointsWithIds._ids.reserve(numberOfNewKeypoints);
            keypointsWithIds._keypoints.reserve(numberOfNewKeypoints);

            // add map points with valid retroprojected coordinates
            for (const Map_Point& point : _localPointMap)
            {
                add_point_to_tracked_features(point, keypointsWithIds);
            }
            // add staged points with valid retroprojected coordinates
            for (const Staged_Point& point : _stagedPoints)
            {
                add_point_to_tracked_features(point, keypointsWithIds);
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

        void Local_Map::draw_point_on_image(const IMap_Point_With_Tracking& mapPoint, const matrix44& worldToCameraMatrix, const cv::Scalar& pointColor, cv::Mat& debugImage) const
        {
            if (mapPoint._lastMatchedIndex != UNMATCHED_POINT_INDEX)
            {
                vector2 screenPoint; 
                const bool isCoordinatesValid = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCameraMatrix, screenPoint);

                //Map Point are green 
                if (isCoordinatesValid)
                {
                    cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, pointColor, 1);
                }
            }
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, const bool shouldDisplayStaged, cv::Mat& debugImage)  const
        {
            const matrix44& worldToCamMatrix = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            for (const IMap_Point_With_Tracking& mapPoint : _localPointMap) {
                draw_point_on_image(mapPoint, worldToCamMatrix, cv::Scalar(0, 255, 0), debugImage);
            }
            if (shouldDisplayStaged)
            {
                for (const IMap_Point_With_Tracking& stagedPoint : _stagedPoints) {
                    draw_point_on_image(stagedPoint, worldToCamMatrix, cv::Scalar(0, 200, 200), debugImage);
                }
            }
        }



    }   /* map_management */
}   /* rgbd_slam */
