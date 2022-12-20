#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include <opencv2/opencv.hpp>

#include "../parameters.hpp"
#include "../features/primitives/shape_primitives.hpp"

#include "../utils/random.hpp"
#include "../utils/coordinates.hpp"

namespace rgbd_slam {
    namespace map_management {

        const features::primitives::planeId UNMATCHED_PRIMITIVE_ID = 0;

        struct MatchedPrimitive 
        {
            MatchedPrimitive():
                _matchId(UNMATCHED_PRIMITIVE_ID),
                _unmatchedCount(0)
            {};

            bool is_matched() const
            {
                return _matchId != UNMATCHED_PRIMITIVE_ID;
            }

            void mark_matched(const features::primitives::planeId matchId)
            {
                _matchId = matchId;
                _unmatchedCount = 0;
            }

            void mark_unmatched()
            {
                _matchId = UNMATCHED_PRIMITIVE_ID;
                ++_unmatchedCount;
            }

            bool is_lost() const
            {
                return _unmatchedCount >= Parameters::get_maximum_unmatched_before_removal();
            }

            features::primitives::planeId _matchId; // Id of the last match
            size_t _unmatchedCount; // count of unmatched iterations
        };

        struct MapPlane 
        {
            MapPlane() : _id(_currentPlaneId++)
            {
                cv::Vec3b color;
                color[0] = utils::Random::get_random_uint(255);
                color[1] = utils::Random::get_random_uint(255);
                color[2] = utils::Random::get_random_uint(255);
                _color = color;
            };

            // Unique identifier of this primitive in map
            const size_t _id;

            utils::PlaneWorldCoordinates _parametrization;
            MatchedPrimitive _matchedPlane;
            cv::Mat _shapeMask;

            cv::Scalar _color;  // display color of this primitive


            private:
            inline static size_t _currentPlaneId = 1;   // 0 is invalid
        };



    }   // map_management
}       // rgbd_slam



#endif
