#include "map_primitive.hpp"
#include "../parameters.hpp"

namespace rgbd_slam {
    namespace map_management {

        MatchedPrimitive::MatchedPrimitive():
            _matchIndex(UNMATCHED_PRIMITIVE_ID)
        {
        }

        bool MatchedPrimitive::is_matched() const
        {
            return _matchIndex != UNMATCHED_PRIMITIVE_ID;
        }

        void MatchedPrimitive::mark_matched(const uint matchIndex)
        {
            _matchIndex = static_cast<int>(matchIndex);
        }

        void MatchedPrimitive::mark_unmatched()
        {
            _matchIndex = UNMATCHED_PRIMITIVE_ID;
        }



        MapPlane::MapPlane(const utils::PlaneWorldCoordinates& parametrization, const utils::WorldCoordinate& centroid, const cv::Mat& shapeMask) : _id(_currentPlaneId++),
            _parametrization(parametrization), _centroid(centroid), _shapeMask(shapeMask), _unmatchedCount(0)
        {
            cv::Vec3b color;
            color[0] = utils::Random::get_random_uint(255);
            color[1] = utils::Random::get_random_uint(255);
            color[2] = utils::Random::get_random_uint(255);
            _color = color;
        }

        uint MapPlane::get_contained_pixels() const
        {
            const static uint cellSize = Parameters::get_depth_map_patch_size();
            const static uint pixelPerCell = cellSize * cellSize;
            return cv::countNonZero(_shapeMask) * pixelPerCell;
        }

        void MapPlane::update(const features::primitives::Plane& detectedPlane, const planeCameraToWorldMatrix& planeCameraToWorld, const cameraToWorldMatrix& cameraToWorld)
        {
            // TODO update plane
            _parametrization = detectedPlane.get_parametrization().to_world_coordinates(planeCameraToWorld);
            _centroid = detectedPlane.get_centroid().to_world_coordinates(cameraToWorld);
            _shapeMask = detectedPlane.get_shape_mask();

            _unmatchedCount = 0;
        }

        void MapPlane::update_unmatched()
        {
            _unmatchedCount += 1;
        }

        bool MapPlane::is_lost() const
        {
            const static size_t maximumUnmatchBeforeremoval = Parameters::get_maximum_unmatched_before_removal();
            return _unmatchedCount >= maximumUnmatchBeforeremoval;
        }


    }
}