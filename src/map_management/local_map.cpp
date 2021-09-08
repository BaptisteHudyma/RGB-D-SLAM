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
                    mapPoint.update_unmacthed();
                    continue;
                }
                else {
                    if (_unmatched[matchIndex]) {
                        alreadyMatchedPoint += 1;
                        mapPoint._lastMatchedIndex = -2;
                        mapPoint.update_unmacthed();
                        continue;
                    }
                    _unmatched[matchIndex] = true;
                    mapPoint.update(_currentIndex, mapPoint._coordinates, mapPoint._descriptor);
                    mapPoint._lastMatchedIndex = matchIndex;
                }
                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                assert(screenPoint[0] > 0 and screenPoint[1] > 0);

                const vector3 screen3DPoint(screenPoint(0), screenPoint(1), detectedKeypoint.get_depth(matchIndex));

                matchedPoints.emplace(matchedPoints.end(), screen3DPoint, mapPoint._coordinates);
            }
            if (alreadyMatchedPoint > 0)
                std::cerr << "Error: Some points were matched twice: " << alreadyMatchedPoint << std::endl;
            _currentIndex += 1;
            return matchedPoints;
        }


        void Local_Map::update(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject)
        {
            // add staged points to local map
            update_staged(optimizedPose, keypointObject);

            // add local map points to global map
            update_local_to_global();
        }

        void Local_Map::reset()
        {
            _localMap.clear();
        }

        void Local_Map::get_debug_image(const poseEstimation::Pose& camPose, cv::Mat& debugImage) const 
        {
            const matrix34& worldToCamMtrx = utils::compute_world_to_camera_transform(camPose.get_orientation_quaternion(), camPose.get_position());

            for (const utils::Map_Point& mapPoint : _localMap) {
                const vector2& screenPoint = utils::world_to_screen_coordinates(mapPoint._coordinates, worldToCamMtrx);

                if (screenPoint[0] > 0 and screenPoint[1] > 0) {
                    if (mapPoint._lastMatchedIndex < 0)
                    {
                        // unmatched points
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 0, 255), 1);
                    }
                    else if (mapPoint.get_age() <= 1)
                    {
                        // new points
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 255, 255), 1);
                    }
                    else {
                        // old matched map points
                        cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(255, 255, 0), 1);
                    }
                }
            }

        }

        void Local_Map::update_staged(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject)
        {
            point_map_container::iterator pointMapIterator;
            unsigned int removePointCount = 0;
            for (pointMapIterator = _localMap.begin(); pointMapIterator != _localMap.end(); ) {
                //utils::Map_Point&
                if (pointMapIterator->_lastMatchedIndex < 0 and pointMapIterator->is_lost(0)) {
                    // Remove useless point
                    _localMap.erase(pointMapIterator++);
                    removePointCount += 1;
                }
                else {
                    pointMapIterator++;
                }
            }

            unsigned int addedPointCount = 0;
            // Add all new points (unmatched points) 
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());
            for(unsigned int i = 0; i < _unmatched.size(); ++i)
            {
                if (not _unmatched[i]) {
                    const double depth = keypointObject.get_depth(i);
                    if (depth <= 0)
                        continue;
                    const vector2& screenPoint = keypointObject.get_keypoint(i);
                    const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint[0], screenPoint[1], depth, worldToCamMatrix);
                    _localMap.emplace(_localMap.end(), worldPoint, keypointObject.get_descriptor(i));
                    addedPointCount += 1;

                }
            }

            std::cout << "Map now contains " << _localMap.size() << " points, with " << addedPointCount << " new points and " << removePointCount << " points removed" << std::endl;
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

    }
}
