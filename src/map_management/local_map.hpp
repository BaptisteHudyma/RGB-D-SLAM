#ifndef SLAM_LOCAL_MAP_HPP
#define SLAM_LOCAL_MAP_HPP 

#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "types.hpp"
#include "map_point.hpp"


namespace rgbd_slam {
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
            std::list<Map_Point> _localMap;
            keypoint_container _stagedPoints;

            cv::Ptr<cv::DescriptorMatcher> _featuresMatcher;

            //local primitive map

    };

}
}

#endif
