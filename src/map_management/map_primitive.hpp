#ifndef RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPRIMITIVE_HPP

#include <opencv2/opencv.hpp>

#include "../parameters.hpp"
#include "../features/primitives/shape_primitives.hpp"

#include "../utils/random.hpp"
#include "../utils/coordinates.hpp"

namespace rgbd_slam {
    namespace map_management {

        const int UNMATCHED_PRIMITIVE_ID = -1;

        struct MatchedPrimitive 
        {
            MatchedPrimitive():
                _matchIndex(UNMATCHED_PRIMITIVE_ID),
                _unmatchedCount(0)
            {};

            bool is_matched() const
            {
                return _matchIndex != UNMATCHED_PRIMITIVE_ID;
            }

            void mark_matched(const uint matchIndex)
            {
                _matchIndex = static_cast<int>(matchIndex);
            }

            void mark_unmatched()
            {
                _matchIndex = UNMATCHED_PRIMITIVE_ID;
            }

            bool is_lost() const
            {
                const static size_t maximumUnmatchBeforeremoval = Parameters::get_maximum_unmatched_before_removal();
                return _unmatchedCount >= maximumUnmatchBeforeremoval;
            }

            int _matchIndex; // Id of the last match
            size_t _unmatchedCount; // count of unmatched iterations
        };

        struct MapPlane 
        {
            MapPlane(const utils::PlaneWorldCoordinates& parametrization, const utils::WorldCoordinate& centroid, const cv::Mat& shapeMask) : _id(_currentPlaneId++),
                _parametrization(parametrization), _centroid(centroid), _shapeMask(shapeMask)
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
            utils::WorldCoordinate _centroid;

            MatchedPrimitive _matchedPlane;
            cv::Mat _shapeMask;

            cv::Scalar _color;  // display color of this primitive

            /**
             * \brief Return the number of pixels in this plane mask
             */
            uint get_contained_pixels() const
            {
                const static uint cellSize = Parameters::get_depth_map_patch_size();
                const static uint pixelPerCell = cellSize * cellSize;
                return cv::countNonZero(_shapeMask) * pixelPerCell;
            }

            void update(const features::primitives::Plane& detectedPlane, const planeCameraToWorldMatrix& planeCameraToWorld, const cameraToWorldMatrix& cameraToWorld)
            {
                // TODO update plane
                _parametrization = detectedPlane.get_parametrization().to_world_coordinates(planeCameraToWorld);
                _centroid = detectedPlane.get_centroid().to_world_coordinates(cameraToWorld);
                _shapeMask = detectedPlane.get_shape_mask();

                _matchedPlane._unmatchedCount = 0;
            }

            void update_unmatched()
            {
                _matchedPlane._unmatchedCount += 1;
            }

            private:
            inline static size_t _currentPlaneId = 1;   // 0 is invalid
        };



    }   // map_management
}       // rgbd_slam



#endif
