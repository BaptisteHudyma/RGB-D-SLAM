#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include "../features/primitives/shape_primitives.hpp"
#include "parameters.hpp"

namespace rgbd_slam {
    namespace map_management {

        const uchar UNMATCHED_PRIMITIVE_ID = 0;

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

            void mark_matched(size_t matchId)
            {
                _matchId = matchId;
                _unmatchedCount = 0;
            }

            void mark_unmatched()
            {
                _matchId = UNMATCHED_PRIMITIVE_ID;
                ++_unmatchedCount;
            }

            bool is_lost()
            {
                return _unmatchedCount >= Parameters::get_maximum_unmatched_before_removal();
            }

            size_t _matchId; // Id of the last match
            size_t _unmatchedCount; // count of unmatched iterations
        };

        struct Primitive 
        {
            explicit Primitive(features::primitives::primitive_uniq_ptr primitive) : 
                _id(_currentPrimitiveId++),
                _primitive(std::move(primitive))
            {
                cv::Vec3b color;
                color[0] = rand() % 255;
                color[1] = rand() % 255;
                color[2] = rand() % 255;
                _color = color;
            };

            // Unique identifier of this primitive in map
            const size_t _id;

            const features::primitives::primitive_uniq_ptr _primitive;
            MatchedPrimitive _matchedPrimitive;

            cv::Scalar _color;  // display color of this primitive


            private:
            inline static size_t _currentPrimitiveId = 1;   // 0 is invalid
        };



    }   // map_management
}       // rgbd_slam



#endif
