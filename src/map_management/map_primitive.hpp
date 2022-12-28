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

        /**
         * \brief Represent a matched primitive, in the detected planes
         */
        struct MatchedPrimitive 
        {
            MatchedPrimitive();

            bool is_matched() const;

            void mark_matched(const uint matchIndex);

            void mark_unmatched();

            int get_match_index() const { return _matchIndex; };

            private:
            int _matchIndex; // Id of the last match
        };

        /**
         * \brief Represent a plane in the local map
         */
        struct MapPlane 
        {
            MapPlane(const utils::PlaneWorldCoordinates& parametrization, const utils::WorldCoordinate& centroid, const cv::Mat& shapeMask);

            /**
             * \brief Return the number of pixels in this plane mask
             */
            uint get_contained_pixels() const;
            
            /**
             * \brief update a map plane with the given detected plane 
             * \param[in] detectedPlane The detected plane, associated with this plane, in camera space
             * \param[in] planeCameraToWorld A matrix to convert from a camera plane to a world plane
             * \param[in] cameraToWorld A matrix to convert from camera point to a world point
             */
            void update(const features::primitives::Plane& detectedPlane, const planeCameraToWorldMatrix& planeCameraToWorld, const cameraToWorldMatrix& cameraToWorld);

            /**
             * \brief Update a plane with no matches
             */
            void update_unmatched();

            bool is_lost() const;

            // Unique identifier of this primitive in map
            const size_t _id;
            // matched detected plane
            MatchedPrimitive _matchedPlane;

            utils::PlaneWorldCoordinates get_parametrization() const { return _parametrization; };
            utils::WorldCoordinate get_centroid() const { return _centroid; };
            cv::Mat get_mask() const { return _shapeMask; };
            cv::Scalar get_color() const { return _color; };


            private:
            inline static size_t _currentPlaneId = 1;   // 0 is invalid

            utils::PlaneWorldCoordinates _parametrization;
            utils::WorldCoordinate _centroid;

            cv::Mat _shapeMask;

            cv::Scalar _color;  // display color of this primitive
            size_t _unmatchedCount; // count of unmatched iterations
        };



    }   // map_management
}       // rgbd_slam



#endif
