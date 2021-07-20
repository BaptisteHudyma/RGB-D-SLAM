#ifndef SLAM_LOCAL_MAP_HPP
#define SLAM_LOCAL_MAP_HPP 

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

#include "types.hpp"


namespace map_management {

    //Observe a point for N frames to gain max liability
#define MAP_POINT_AGE_LIABILITY 5
    // Maximum time before we consider this point too old
#define MAX_LOST_TEMPORISATION 1
    // Minimum point liability for the local map
#define MINIMUM_LIABILITY_FOR_LOCAL_MAP 0.7
    // Max unmatched points to consider this map point as lost
#define MAX_UNMTACHED_FOR_TRACKING 2

    /**
     * \brief A map point structure, containing all the necessary informations to identify a map point
     */
    struct Map_Point {
        // world coordinates
        vector3 _coordinates;

        // 3D descriptor (SURF)
        cv::Mat _descriptor;

        // depending on context, if the point is staged then this is the number of frames it has been tracked while staged. If it is a map point, then this is the number of times it failed tracking.
        unsigned int _counter;

        // Successful matches count
        unsigned int _age;   

        //0 - 1: keypoint fiability (consecutive matches, ...) 
        float _liability;

        //timestamp of the last update
        double _lastUpdated;

        Map_Point(double observationTimeStamp = 0) {
            _age = 0;
            _lastUpdated =  observationTimeStamp;
            _counter = 0;
        }

        double get_liability() {
            double liability = _age /  MAP_POINT_AGE_LIABILITY;
            if (liability > 1.0)
                return 1.0;
            return liability;
        }

        /**
         * \brief True is this point is lost : should be removed from local map. Should be used only for map points
         */
        bool is_lost(double currentTimeStamp) {
            return (_counter > MAX_UNMTACHED_FOR_TRACKING) and ((currentTimeStamp - _lastUpdated) > MAX_LOST_TEMPORISATION);
        }

        /**
         * \brief Should add this staged point to the local map
         */
        bool should_add_to_map() {
            return (get_liability() > MINIMUM_LIABILITY_FOR_LOCAL_MAP);
        }

        /**
         * \brief Update this point without it being detected/matched
         */
        void update_unmacthed()
        {
            // decrease successful matches
            if (_age > 0)
                _age -= 1;
            _counter += 1;
        }

        /**
         * \brief Update this map point with the given informations: it is matched with another point
         */
        void update_map_point(double observationTimeStamp, const vector3& newPointCoordinates, const cv::Mat& newDescriptor) 
        {
            _counter = 0;
            _age += 1;
            _coordinates = newPointCoordinates;
            _descriptor = newDescriptor;
            _lastUpdated = observationTimeStamp;
        }

        void update_staged_point(double observationTimeStamp, const vector3& newPointCoordinates, const cv::Mat& newDescriptor) 
        {
            _counter += 1;
            _age += 1;
            _coordinates = newPointCoordinates;
            _descriptor = newDescriptor;
            _lastUpdated = observationTimeStamp;
        }
    };



    /**
     * \brief Maintain a local map around the camera. Can return matched features, and update the global map when features are estimated to be reliable. For now we dont have a global map
     */
    class Local_Map {
        public:
            Local_Map();

            /**
             * \brief Compute the point feature matches between the local map and a given set of points. Update the staged point list matched points
             */
            const matched_point_container find_matches(const keypoint_container& detectedPoints);

            /**
             * \brief Update the local and global map 
             */
            void update();


            /**
             * \brief Hard clean the local map
             */
            void reset();

        protected:
            /**
             * \brief Add previously uncertain features to the local map
             */
            void update_staged();

            /**
             * \brief Clean the local map so it stays local, and update the global map with the good features
             */
            void update_local_to_global();

        private:
            //local point map
            std::list<Map_Point> _localMap;
            keypoint_container _stagedPoints;

            //local primitive map

    };

}

#endif
