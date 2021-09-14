#include "local_map.hpp"

#include "utils.hpp"

namespace rgbd_slam {
    namespace map_management {

        Local_Map::Local_Map()
        {
            _currentIndex = 0;
        }

        match_point_container Local_Map::find_matches(const utils::Keypoint_Handler& detectedKeypoint)
        {
            _isPointMatched = std::vector<bool>(detectedKeypoint.get_keypoint_count(), false);
            match_point_container matchedPoints; 

            unsigned int alreadyMatchedPoint = 0;
            for (utils::Map_Point& mapPoint : _localMap) 
            {
                int matchIndex = detectedKeypoint.get_match_index(mapPoint);
                if (matchIndex < 0 or detectedKeypoint.get_depth(matchIndex) <= 0) {
                    //unmatched point
                    mapPoint._lastMatchedIndex = -2;
                    mapPoint.update_unmatched();
                    continue;
                }
                else {
                    if (_isPointMatched[matchIndex]) {
                        alreadyMatchedPoint += 1;
                        mapPoint._lastMatchedIndex = -2;
                        mapPoint.update_unmatched();
                        continue;
                    }
                    else
                    {
                        _isPointMatched[matchIndex] = true;
                        mapPoint._lastMatchedIndex = matchIndex;
                    } 
                }
                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                assert(screenPoint[0] > 0 and screenPoint[1] > 0);

                const vector3 screen3DPoint(screenPoint(0), screenPoint(1), detectedKeypoint.get_depth(matchIndex));

                matchedPoints.emplace(matchedPoints.end(), screen3DPoint, mapPoint._coordinates);
            }

            // Try to find matches in staged points
            for(utils::Staged_Point& stagedPoint : _stagedPoints)
            {
                int matchIndex = detectedKeypoint.get_match_index(stagedPoint);
                if (matchIndex < 0 or detectedKeypoint.get_depth(matchIndex) <= 0) {
                    //unmatched point
                    stagedPoint._lastMatchedIndex = -2;
                    stagedPoint.update_unmatched();
                    continue;
                }
                else {
                    if (_isPointMatched[matchIndex]) {
                        // Already matched to another point
                        stagedPoint._lastMatchedIndex = -2;
                        stagedPoint.update_unmatched(2);
                        continue;
                    }
                    else
                    {
                        _isPointMatched[matchIndex] = true;
                        stagedPoint._lastMatchedIndex = matchIndex;
                    }
                }
                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                assert(screenPoint[0] > 0 and screenPoint[1] > 0);

                const vector3 screen3DPoint(screenPoint(0), screenPoint(1), detectedKeypoint.get_depth(matchIndex));

                matchedPoints.emplace(matchedPoints.end(), screen3DPoint, stagedPoint._coordinates);
            }

            if (alreadyMatchedPoint > 0)
                std::cerr << "Error: Some points were matched twice: " << alreadyMatchedPoint << std::endl;
            _currentIndex += 1;
            return matchedPoints;
        }


        void Local_Map::update(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject)
        {
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            double meanPointMapError = 0.0;
            double matchedPointCount = 0.0;
            // Remove old map points
            point_map_container::iterator pointMapIterator = _localMap.begin();
            while(pointMapIterator != _localMap.end())
            {
                const int matchedPointIndex = pointMapIterator->_lastMatchedIndex;
                if (matchedPointIndex >= 0)
                {
                    // get match coordinates, transform them to world coordinates
                    const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                    const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), keypointObject.get_depth(matchedPointIndex), worldToCamMatrix);

                    // update this map point errors & position
                    meanPointMapError += pointMapIterator->update_matched(newCoordinates, keypointObject.get_descriptor(matchedPointIndex));
                    matchedPointCount += 1.0;
                }

                if (pointMapIterator->is_lost()) {
                    // Remove useless point
                    pointMapIterator = _localMap.erase(pointMapIterator);
                }
                else
                {
                    ++pointMapIterator;
                }
            }

            // add staged points to local map
            update_staged(worldToCamMatrix, keypointObject);

            // add local map points to global map
            update_local_to_global();

            std::cout << "Mean map point to detected point error is " << meanPointMapError / matchedPointCount << std::endl;
        }

        void Local_Map::update_staged(const matrix34& worldToCamMatrix, const utils::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::iterator stagedPointIterator = _stagedPoints.begin();
            while(stagedPointIterator != _stagedPoints.end())
            {
                const int matchedPointIndex = stagedPointIterator->_lastMatchedIndex;
                if (matchedPointIndex >= 0)
                {
                    // get match coordinates, transform them to world coordinates
                    const vector2& matchedPointCoordinates = keypointObject.get_keypoint(matchedPointIndex);
                    const vector3& newCoordinates = utils::screen_to_world_coordinates(matchedPointCoordinates.x(), matchedPointCoordinates.y(), keypointObject.get_depth(matchedPointIndex), worldToCamMatrix);

                    // update this map point errors & position
                    stagedPointIterator->update_matched(newCoordinates, keypointObject.get_descriptor(matchedPointIndex));
                }
                
                if (stagedPointIterator->should_add_to_local_map())
                {
                    // Add to local map, remove from staged points
                    _localMap.emplace(_localMap.end(), stagedPointIterator->_coordinates, stagedPointIterator->_descriptor);
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
            for(unsigned int i = 0; i < _isPointMatched.size(); ++i)
            {
                if (not _isPointMatched[i]) {
                    const double depth = keypointObject.get_depth(i);
                    if (depth <= 0)
                        continue;
                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint[0], screenPoint[1], depth, worldToCamMatrix);
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

        void Local_Map::get_debug_image(const poseEstimation::Pose& camPose, cv::Mat& debugImage) const 
        {
            const matrix34& worldToCamMtrx = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            for (const utils::Map_Point& mapPoint : _localMap) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMtrx);

                if (screenPoint[0] > 0 and screenPoint[1] > 0) {
                    //Map Point are green 
                    if (mapPoint._lastMatchedIndex < 0)
                    {
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 128, 0), 1);
                    }
                    else
                    {
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 255, 0), 1);
                    }
                }
            }
            for (const utils::Staged_Point& stagedPoint : _stagedPoints) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(stagedPoint._coordinates, worldToCamMtrx);

                if (screenPoint[0] > 0 and screenPoint[1] > 0) {
                    //Staged point are yellow 
                    if (stagedPoint._lastMatchedIndex < 0)
                    {
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 100, 200), 1);
                    }
                    else
                    {
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 200, 200), 1);
                    }
                }
            }

        }



    }   /* map_management */
}   /* rgbd_slam */
