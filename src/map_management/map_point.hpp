#ifndef RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP
#define RGBDSLAM_MAPMANAGEMENT_MAPPOINT_HPP

#include "../types.hpp"
#include <opencv2/opencv.hpp>

#include "tracking/kalman_filter.hpp"

namespace rgbd_slam {
    namespace map_management {

        const size_t INVALID_POINT_UNIQ_ID = 0; // This id indicates an invalid unique id for a map point
        const int UNMATCHED_POINT_INDEX = -1;      // Id of a unmatched point

        struct MatchedScreenPoint
        {
            MatchedScreenPoint():
                _matchIndex(UNMATCHED_POINT_INDEX)
            {
                _screenCoordinates.setZero();
            };

            explicit MatchedScreenPoint(const screenCoordinates& screenPoint, const int matchIndex = UNMATCHED_POINT_INDEX):
                _screenCoordinates(screenPoint),
                _matchIndex(matchIndex)
            {};

            bool is_matched() const
            { 
                return _matchIndex != UNMATCHED_POINT_INDEX;
            }

            void mark_unmatched()
            {
                _matchIndex = UNMATCHED_POINT_INDEX;
                _screenCoordinates.setZero();
            }

            // matched point coordinates in screen space
            screenCoordinates _screenCoordinates;

            // Match index in the detected point object (can be UNMATCHED_POINT_INDEX);
            int _matchIndex;
        };

        /**
         * \brief Basic keypoint class 
         */
        struct Point 
        {
            // world coordinates
            worldCoordinates _coordinates;

            // 3D descriptor (SURF)
            cv::Mat _descriptor;

            // unique identifier, to match this point without using descriptors
            const size_t _id;

            protected:
            Point (const worldCoordinates& coordinates, const cv::Mat& descriptor);
            // copy constructor
            Point (const worldCoordinates& coordinates, const cv::Mat& descriptor, const size_t id);

            inline static size_t _currentPointId = 1;   // 0 is invalid
        };

        /**
         * \brief Prepare the basic functions for a staged/map point
         */
        struct IMap_Point_With_Tracking
            : public Point
        {
            IMap_Point_With_Tracking(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor);
            IMap_Point_With_Tracking(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id);
            /**
             * \brief Compute a confidence in this point (-1, 1)
             */
            virtual double get_confidence() const =  0;

            /**
             * \brief Call when this point was matched to another point
             */
            virtual double update_matched(const worldCoordinates& newPointCoordinates, const matrix33& covariance) = 0;

            /**
             * \brief Call when this point was not matched to anything
             */
            virtual void update_unmatched(int removeNMatches = 1) = 0;

            const Eigen::MatrixXd get_covariance_matrix() const { 
                return _kalmanFilter->get_state_covariance();
            };

            // an object referencing the last match for this point
            MatchedScreenPoint _matchedScreenPoint;

            protected:

            /**
             * \brief update the current point by tracking with a kalman filter. Will update the point position & covariance
             * \return The distance between the new position ans the previous one
             */
            double track_point(const worldCoordinates& newPointCoordinates, const matrix33& newPointCovariance);

            /**
             * \brief Build the inputs caracteristics of the kalman filter
             */
            void build_kalman_filter();

            private:
                tracking::KalmanFilter* _kalmanFilter;
        };

        /**
         * \brief Concurrent for a map point
         */
        class Staged_Point
            : public IMap_Point_With_Tracking
        {
            public:
                Staged_Point(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor);
                Staged_Point(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id);

                // Count the number of times his points was matched
                int _matchesCount;

                /**
                 * \brief Should add this staged point to the local map
                 */
                bool should_add_to_local_map() const;

                /**
                 * \brief True if this staged point should not be added to local map
                 */
                bool should_remove_from_staged() const;

                /**
                 * \brief Call when this point was not matched to anything
                 */
                void update_unmatched(int removeNMatches = 1) override;

                /**
                 * \brief Call when this point was matched to another point
                 */
                double update_matched(const worldCoordinates& newPointCoordinates, const matrix33& covariance) override;

            private:
                /**
                 * \brief Compute a confidence in this point (-1, 1)
                 */
                double get_confidence() const override;
        };


        /**
         * \brief A map point structure, containing all the necessary informations to identify a map point
         */
        class Map_Point 
            : public IMap_Point_With_Tracking 
        {

            public:
                Map_Point(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor);
                Map_Point(const worldCoordinates& coordinates, const matrix33& covariance, const cv::Mat& descriptor, const size_t id);


                /**
                 * \brief True is this point is lost : should be removed from local map. Should be used only for map points
                 */
                bool is_lost() const; 

                /**
                 * \brief Update this point without it being detected/matched
                 */
                void update_unmatched(int removeNMatches = 1) override;

                /**
                 * \brief Update this map point with the given informations: it is matched with another point
                 */
                double update_matched(const worldCoordinates& newPointCoordinates, const matrix33& covariance) override;

                int get_age() const {
                    return _age;
                }

                cv::Scalar _color;  // display color of this primitive

            protected:
                void set_random_color();

                /**
                 * \brief Compute a confidence score (-1, 1)
                 */
                double get_confidence() const override; 

            private:
                // The number of times this point failed tracking.
                unsigned int _failTrackingCount;

                // Successful matches count
                int _age;
        };

    } /* map_management */
} /* rgbd_slam */

#endif
