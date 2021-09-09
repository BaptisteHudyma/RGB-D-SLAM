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
            _unmatched = std::vector<bool>(detectedKeypoint.get_keypoint_count(), false);
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
                    if (_unmatched[matchIndex]) {
                        alreadyMatchedPoint += 1;
                        mapPoint._lastMatchedIndex = -2;
                        mapPoint.update_unmatched();
                        continue;
                    }
                    else
                    {
                        _unmatched[matchIndex] = true;
                        mapPoint.update(mapPoint._coordinates, detectedKeypoint.get_descriptor(matchIndex));
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
                    if (_unmatched[matchIndex]) {
                        stagedPoint._lastMatchedIndex = -2;
                        stagedPoint.update_unmatched();
                        continue;
                    }
                    else
                    {
                        _unmatched[matchIndex] = true;
                        stagedPoint.update_matched(stagedPoint._coordinates, detectedKeypoint.get_descriptor(matchIndex));
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
            // Remove old map points
            point_map_container::const_iterator pointMapIterator = _localMap.cbegin();
            while(pointMapIterator != _localMap.cend())
            {
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
            update_staged(optimizedPose, keypointObject);

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::update_staged(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject)
        {
            // Add correct staged points to local map
            staged_point_container::const_iterator stagedPointIterator = _stagedPoints.cbegin();
            while(stagedPointIterator != _stagedPoints.cend())
            {
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
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());
            for(unsigned int i = 0; i < _unmatched.size(); ++i)
            {
                if (not _unmatched[i]) {
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
