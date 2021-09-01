#ifndef LOCAL_MAP_POINT_HPP
#define LOCAL_MAP_POINT_HPP

#include "types.hpp"

#include <opencv2/opencv.hpp>

//Observe a point for N frames to gain max liability
#define MAP_POINT_AGE_LIABILITY 5
// Maximum time before we consider this point too old
#define MAX_LOST_TEMPORISATION 1
// Minimum point liability for the local map
#define MINIMUM_LIABILITY_FOR_LOCAL_MAP 0.7
// Max unmatched points to consider this map point as lost
#define MAX_UNMTACHED_FOR_TRACKING 2


namespace rgbd_slam {
    namespace utils {


        /**
         * \brief Basic keypoint class 
         */
        struct Point {
            // world coordinates
            vector3 _coordinates;

            // 3D descriptor (SURF)
            cv::Mat _descriptor;

            Point (const vector3& coordinates, const cv::Mat& descriptor);
        };


        /**
         * \brief A map point structure, containing all the necessary informations to identify a map point
         */
        class Map_Point 
            : public Point
        {

            public:
                Map_Point(const vector3& coordinates, const cv::Mat& descriptor, double observationTimeStamp = 0);

                double get_liability(); 

                /**
                 * \brief True is this point is lost : should be removed from local map. Should be used only for map points
                 */
                bool is_lost(double currentTimeStamp); 

                /**
                 * \brief Should add this staged point to the local map
                 */
                bool should_add_to_map();

                /**
                 * \brief Update this point without it being detected/matched
                 */
                void update_unmacthed();

                /**
                 * \brief Update this map point with the given informations: it is matched with another point
                 */
                void update(double observationTimeStamp, const vector3& newPointCoordinates, const cv::Mat& newDescriptor);
                
                int _lastMatchedIndex;

            private:
                // depending on context, if the point is staged then this is the number of frames it has been tracked while staged. If it is a map point, then this is the number of times it failed tracking.
                unsigned int _counter;

                // Successful matches count
                unsigned int _age;   

                //0 - 1: keypoint fiability (consecutive matches, ...) 
                float _liability;

                //timestamp of the last update
                double _lastUpdated;


        };

    }

    typedef std::list<utils::Point> point_container;
    typedef std::pair<utils::Point, utils::Point> map_point_pair;
    typedef std::list<map_point_pair> matched_point_container;
    typedef std::map<unsigned int, utils::Point> point_map;
}

#endif
