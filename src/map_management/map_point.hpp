#ifndef LOCAL_MAP_POINT_HPP
#define LOCAL_MAP_POINT_HPP

#include "types.hpp"

#include <opencv2/opencv.hpp>



namespace rgbd_slam {
    namespace utils {


        /**
         * \brief Basic keypoint class 
         */
        struct Point {
            // world coordinates
            vector3 _coordinates;

            // Gaussian uncertainty in 3D
            vector3 _coordinateUncertainty;

            // 3D descriptor (SURF)
            cv::Mat _descriptor;

            Point (const vector3& coordinates, const cv::Mat& descriptor);
        };

        class Staged_Point
            : public Point
        {
            public:
                Staged_Point(const vector3& coordinates, const cv::Mat& descriptor);

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
                void update_unmatched(int removeNMatches = 1);

                /**
                  * \brief Call when this point was matched to another point
                  */
                void update_matched(const vector3& newPointCoordinates, const cv::Mat& newDescriptor);

                int _lastMatchedIndex;
            private:
                /**
                 * \brief Compute a confidence in this point (-1, 1)
                 */
                virtual double get_confidence() const;

                // count of unique matches
                int _matchedCount;
        };


        /**
         * \brief A map point structure, containing all the necessary informations to identify a map point
         */
        class Map_Point 
            : public Point
        {

            public:
                Map_Point(const vector3& coordinates, const cv::Mat& descriptor);


                /**
                 * \brief True is this point is lost : should be removed from local map. Should be used only for map points
                 */
                bool is_lost() const; 

                /**
                 * \brief Update this point without it being detected/matched
                 */
                void update_unmatched();

                /**
                 * \brief Update this map point with the given informations: it is matched with another point
                 */
                double update_matched(const vector3& newPointCoordinates, const cv::Mat& newDescriptor);

                int get_age() const {
                    return _age;
                }

                int _lastMatchedIndex;

            protected:
                /**
                  * \brief Compute a confidence score (-1, 1)
                  */
                double get_confidence() const; 

            private:
                // The number of times this point failed tracking.
                unsigned int _failTrackingCount;

                // Successful matches count
                int _age;   

                //0 - 1: keypoint confidence (consecutive matches, ...) 
                float _confidence;
        };

    }

    typedef std::list<utils::Point> point_container;
    typedef std::pair<utils::Point, utils::Point> map_point_pair;
    typedef std::list<map_point_pair> matched_point_container;
    typedef std::map<unsigned int, utils::Point> point_map;
}

#endif
