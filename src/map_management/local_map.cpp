#include "local_map.hpp"

#include "utils.hpp"

namespace rgbd_slam {
    namespace map_management {

        Local_Map::Local_Map()
        {

        }

        match_point_container Local_Map::find_matches(const utils::Keypoint_Handler& detectedKeypoint)
        {
            match_point_container matchedPoints; 
            for (const utils::Map_Point& mapPoint : _localMap) 
            {
                int matchIndex = detectedKeypoint.get_match_index(mapPoint);
                if (matchIndex < 0) {
                    //unmatched point
                    continue;
                }
                const vector2& screenPoint = detectedKeypoint.get_keypoint(matchIndex);
                assert(screenPoint[0] > 0 and screenPoint[1] > 0);
                matchedPoints.emplace(matchedPoints.end(), screenPoint, mapPoint._coordinates);
            }
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
                    cv::circle(debugImage, cv::Point(screenPoint[0], screenPoint[1]), 4, cv::Scalar(0, 255, 255), 1);
                }
            }
        }

        void Local_Map::update_staged(const poseEstimation::Pose optimizedPose, const utils::Keypoint_Handler& keypointObject)
        {
            //TEMPORARY: prevent too much point by keeping them only one frame
            _localMap.clear();
            const matrix34& worldToCamMatrix = utils::compute_world_to_camera_transform(optimizedPose.get_orientation_quaternion(), optimizedPose.get_position());

            const unsigned int keypointCount = keypointObject.get_keypoint_count();
            for (unsigned int i = 0; i < keypointCount; ++i) {
                double depth = keypointObject.get_depth(i);

                if(depth <= 0)
                    continue;

                const vector2& screenPoint = keypointObject.get_keypoint(i);
                const vector3& worldPoint = utils::screen_to_world_coordinates(screenPoint[0], screenPoint[1], depth, worldToCamMatrix);
                _localMap.emplace(_localMap.end(), worldPoint, keypointObject.get_descriptor(i));
            }
        }

        void Local_Map::update_local_to_global() 
        {
            // TODO when we have a global map

        }

    }
}
