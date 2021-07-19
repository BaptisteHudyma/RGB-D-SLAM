#ifndef SLAM_LOCAL_MAP_HPP
#define SLAM_LOCAL_MAP_HPP 

#include <list>
#include <vector>

#include "types.hpp"
#include "KeyPointDetection.hpp"

typedef std::list<point_pair> matched_point_container;

namespace map_management {

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
            std::vector<vector3> _localMap;
            std::vector<vector3> _stagedPoints;

            //local primitive map

    };

}

#endif
