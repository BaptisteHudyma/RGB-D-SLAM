#include "local_map.hpp"

#include "parameters.hpp"
#include "utils.hpp"

namespace rgbd_slam {
    namespace map_management {

        Local_Map::Local_Map()
        {
        }

        match_point_container Local_Map::find_matches(const utils::Pose currentPose, const features::keypoints::Keypoint_Handler& detectedKeypoint)
        {
            // will be used to detect new keypoints for the stagged map
            _isPointMatched.clear();
            _isPointMatched = std::vector<bool>(detectedKeypoint.get_keypoint_count(), false);
            match_point_container matchedPoints; 

            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(currentPose.get_orientation_quaternion(), currentPose.get_position());

            // Try to find matches in local map
            for (Map_Point& mapPoint : _localMap) 
            {
                const vector2& projectedMapPoint = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMatrix);
                int matchIndex = detectedKeypoint.get_tracking_match_index(mapPoint._id, _isPointMatched);
                if (matchIndex < 0)
                {
                    matchIndex = detectedKeypoint.get_match_index(projectedMapPoint, mapPoint._descriptor, _isPointMatched);
                }

                if (matchIndex < 0) {
                    //unmatched point
                    mapPoint._lastMatchedIndex = -1;
                }
                else if (detectedKeypoint.get_depth(matchIndex) <= 0) {
                    // 2D point, still matched
                    _isPointMatched[matchIndex] = true;
                    mapPoint._lastMatchedIndex = matchIndex;
                }
                else {
                    _isPointMatched[matchIndex] = true;
                    mapPoint._lastMatchedIndex = matchIndex;
                    const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                    assert(screenPoint.x() > 0 and screenPoint.y() > 0);

                    const vector3 screen3DPoint(screenPoint.x(), screenPoint.y(), detectedKeypoint.get_depth(matchIndex));

                    matchedPoints.emplace(matchedPoints.end(), screen3DPoint, mapPoint._coordinates);
                }
            }

            // Try to find matches in staged points
            for(Staged_Point& stagedPoint : _stagedPoints)
            {
                const vector2& projectedStagedPoint = utils::world_to_screen_coordinates(stagedPoint._coordinates, worldToCamMatrix);

                int matchIndex = detectedKeypoint.get_tracking_match_index(stagedPoint._id, _isPointMatched);
                if (matchIndex < 0)
                {
                    matchIndex = detectedKeypoint.get_match_index(projectedStagedPoint, stagedPoint._descriptor, _isPointMatched);
                }

                if (matchIndex < 0) {
                    //unmatched point
                    stagedPoint._lastMatchedIndex = -1;
                }
                else if (detectedKeypoint.get_depth(matchIndex) <= 0) {
                    // 2D point, still matched
                    _isPointMatched[matchIndex] = true;
                    stagedPoint._lastMatchedIndex = matchIndex;
                }
                else {
                    _isPointMatched[matchIndex] = true;
                    stagedPoint._lastMatchedIndex = matchIndex;
                    const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                    assert(screenPoint.x() > 0 and screenPoint.y() > 0);

                    const vector3 screen3DPoint(screenPoint(0), screenPoint(1), detectedKeypoint.get_depth(matchIndex));

                    matchedPoints.emplace(matchedPoints.end(), screen3DPoint, stagedPoint._coordinates);
                }
            }

            return matchedPoints;
        }


        void Local_Map::update(const utils::Pose optimizedPose, const features::keypoints::Keypoint_Handler& keypointObject, features::keypoints::KeypointsWithIdStruct& keypointsWithIds)
        {
            const matrix34& camToWorldMatrix = utils::compute_camera_to_world_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            // add local map points
            update_local_map(camToWorldMatrix, keypointObject);

            // add staged points to local map
            update_staged(camToWorldMatrix, keypointObject);

            // Update tracked keypoints object
            keypointsWithIds._ids.clear();
            keypointsWithIds._keypoints.clear();
            const size_t numberOfNewKeypoints = _localMap.size() + _stagedPoints.size();
            keypointsWithIds._ids.reserve(numberOfNewKeypoints);
            keypointsWithIds._keypoints.reserve(numberOfNewKeypoints);

            for (const Map_Point& point : _localMap)
            {
                const vector2& projectedPoint = utils::world_to_screen_coordinates(point._coordinates, worldToCamMatrix);
                const cv::Point2f projectedPointCV(projectedPoint.x(), projectedPoint.y());
                keypointsWithIds._keypoints.push_back(projectedPointCV);
                keypointsWithIds._ids.push_back(point._id);
            }
            for (const Staged_Point& point : _stagedPoints)
            {
                const vector2& projectedPoint = utils::world_to_screen_coordinates(point._coordinates, worldToCamMatrix);
                const cv::Point2f projectedPointCV(projectedPoint.x(), projectedPoint.y());
                keypointsWithIds._keypoints.push_back(projectedPointCV);
                keypointsWithIds._ids.push_back(point._id);
            }

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::update_local_map(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Must update [2][2] with depth error
            matrix33 screenPointError {
                {1.0/12.0, 0,        0},
                    {0,        1.0/12.0, 0},
                    {0,        0,        0},
            };

            // Remove old map points
            point_map_container::iterator pointMapIterator = _localMap.begin();
            while(pointMapIterator != _localMap.end())
            {
                bool shouldRemovePoint = false;
                const int matchedPointIndex = pointMapIterator->_lastMatchedIndex;
                const int keypointsSize = static_cast<int>(keypointObject.get_keypoint_count());
                if (matchedPointIndex >= 0 and matchedPointIndex < keypointsSize)
                {
                    // get match coordinates, transform them to world coordinates
                    const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                    const double matchedPointDepth = keypointObject.get_depth(matchedPointIndex);

                    const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), matchedPointDepth, camToWorldMatrix);

                    const double pointDepthError = 0.00313 + 0.00116 * matchedPointDepth + 0.00052 * pow(matchedPointDepth, 2.0);
                    screenPointError(2, 2) = pointDepthError;
                    //std::cout << utils::get_world_point_covariance(matchedPointCoordinates, matchedPointDepth, screenPointError) << std::endl;
                    //std::cout << std::endl;

                    // update this map point errors & position
                    const double retroprojectionError = pointMapIterator->update_matched(newCoordinates);
                    shouldRemovePoint = (retroprojectionError > Parameters::get_maximum_map_retroprojection_error());
                }
                else if (matchedPointIndex > 0 and matchedPointIndex >= keypointsSize)
                {
                    utils::log_error("Match point index is out of bounds of keypointObject");
                }
                else
                {
                    pointMapIterator->update_unmatched();
                }

                if (shouldRemovePoint or pointMapIterator->is_lost()) {
                    // Remove useless point
                    pointMapIterator = _localMap.erase(pointMapIterator);
                }
                else
                {
                    ++pointMapIterator;
                }
            }


        }

        void Local_Map::update_staged(const matrix34& camToWorldMatrix, const features::keypoints::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                bool shouldRemovePoint = false;
                const int matchedPointIndex = stagedPointIterator->_lastMatchedIndex;
                const int keypointsSize = static_cast<int>(keypointObject.get_keypoint_count());
                if (matchedPointIndex >= 0 and matchedPointIndex < keypointsSize)
                {
                    // get match coordinates, transform them to world coordinates
                    const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                    const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), keypointObject.get_depth(matchedPointIndex), camToWorldMatrix);

                    // update this map point errors & position
                    const double retroprojectionError = stagedPointIterator->update_matched(newCoordinates);
                    shouldRemovePoint = (retroprojectionError > Parameters::get_maximum_map_retroprojection_error());
                }
                else if (matchedPointIndex > 0 and matchedPointIndex >= keypointsSize)
                {
                    utils::log_error("Match point index is out of bounds of keypointObject");
                }
                else
                {
                    stagedPointIterator->update_unmatched();
                }

                if (stagedPointIterator->should_add_to_local_map())
                {
                    // Add to local map, remove from staged points, with a copy of the id affected to the local map
                    _localMap.emplace(_localMap.end(), stagedPointIterator->_coordinates, stagedPointIterator->_descriptor, stagedPointIterator->_id);
                    stagedPointIterator = _stagedPoints.erase(stagedPointIterator);
                }
                else if (shouldRemovePoint or stagedPointIterator->should_remove_from_staged())
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
                    if (depth <= 0)
                    {
                        // TODO handle 2D features
                        continue;
                    }

                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint.x(), screenPoint.y(), depth, camToWorldMatrix);
                    _stagedPoints.emplace(_stagedPoints.end(), worldPoint, keypointObject.get_descriptor(i));
                }
            }
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

        void Local_Map::reset()
        {
            _localMap.clear();
            _stagedPoints.clear();
        }

        void Local_Map::get_debug_image(const utils::Pose& camPose, cv::Mat& debugImage) const 
        {
            const matrix34& worldToCamMtrx = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            for (const Map_Point& mapPoint : _localMap) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMtrx);

                if (screenPoint.x() > 0 and screenPoint.y() > 0) {
                    //Map Point are green 
                    if (mapPoint._lastMatchedIndex < 0)
                    {
                        cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 128, 0), 1);
                    }
                    else
                    {
                        cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 255, 0), 1);
                    }
                }
            }
            for (const Staged_Point& stagedPoint : _stagedPoints) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(stagedPoint._coordinates, worldToCamMtrx);

                if (screenPoint.x() > 0 and screenPoint.x() > 0) {
                    //Staged point are yellow 
                    if (stagedPoint._lastMatchedIndex < 0)
                    {
                        cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 100, 200), 1);
                    }
                    else
                    {
                        cv::circle(debugImage, cv::Point(screenPoint.x(), screenPoint.y()), 4, cv::Scalar(0, 200, 200), 1);
                    }
                }
            }
        }



    }   /* map_management */
}   /* rgbd_slam */
