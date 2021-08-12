#include "local_map.hpp"

namespace rgbd_slam {
namespace map_management {

    Local_Map::Local_Map()
    {
        _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));
    }

    const matched_point_container Local_Map::find_matches(const keypoint_container& detectedPoints) 
    {
        // KNN search
        // Outliers ?
        // add map matches (world coordinates) to matched list
        // add raw keypoints (frame coordinates) to staged points

        //pose opti
        //update map with stagged points, using optimized pose to update staged point positions
        

        /*
        std::vector< std::vector<cv::DMatch> > knnMatches;
        _featuresMatcher->knnMatch(_lastFrameDescriptors, thisFrameDescriptors, knnMatches, 2);

        matched_point_container matchedPoints;
        for (size_t i = 0; i < knnMatches.size(); i++)
        {
            const std::vector<cv::DMatch>& match = knnMatches[i];
            //check if point is a good match by checking it's distance to the second best matched point
            if (match[0].distance < _maxMatchDistance * match[1].distance)
            {
                int trainIdx = match[0].trainIdx;   //last frame key point
                int queryIdx = match[0].queryIdx;   //this frame key point

                if (_localMap.contains(trainIdx) and thisFrameKeypoints.contains(queryIdx)) {
                    point_pair newMatch = std::make_pair(
                            _localMap[trainIdx]._coordinates,
                            thisFrameKeypoints[queryIdx]
                            );
                    matchedPoints.push_back(newMatch);
                }
            }
        }
        return matchedPoints;
        */
    }


    void Local_Map::update()
    {
        // add staged points to local map
        update_staged();

        // add local map points to global map
        update_local_to_global();
    }

    void Local_Map::reset()
    {
        _localMap.clear();
        _stagedPoints.clear();
    }

    void Local_Map::update_staged() 
    {

    }

    void Local_Map::update_local_to_global() 
    {
        // TODO when we have a global map

    }

}
}
