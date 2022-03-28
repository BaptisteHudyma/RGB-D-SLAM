#include "keypoint_handler.hpp"

#include "parameters.hpp"
#include "logger.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {


            bool is_in_border(const cv::Point2f &pt, const cv::Mat &im) 
            {
                return 
                    BORDER_SIZE <= pt.x and
                    BORDER_SIZE <= pt.y and
                    pt.x < im.cols - BORDER_SIZE and
                    pt.y < im.rows - BORDER_SIZE;
            } 

            float get_depth_approximation(const cv::Mat& depthImage, const cv::Point2f& depthCoordinates)
            {
                if (is_in_border(depthCoordinates, depthImage)) 
                {
                    const float border = 2;
                    const cv::Mat roi(depthImage(cv::Rect(depthCoordinates.x - border, depthCoordinates.y - border, border * 2, border * 2)));
                    double min, max;
                    cv::minMaxLoc(roi, &min, &max);
                    return min;
                }
                return 0;
            }


            Keypoint_Handler::Keypoint_Handler(std::vector<cv::Point2f>& inKeypoints, cv::Mat& inDescriptors, const KeypointsWithIdStruct& lastKeypointsWithIds, const cv::Mat& depthImage, const double maxMatchDistance) :
                _maxMatchDistance(maxMatchDistance)
            {
                if (_maxMatchDistance <= 0) {
                    utils::log_error("Maximum matching distance must be > 0");
                    exit(-1);
                }
                // knn matcher
                _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));

                _descriptors = inDescriptors;

                const float cellSize = static_cast<float>(Parameters::get_search_matches_cell_size());
                _searchSpaceCellRadius = std::ceil(Parameters::get_search_matches_distance() / cellSize);

                _cellCountX = std::ceil(depthImage.cols / cellSize);
                _cellCountY = std::ceil(depthImage.rows / cellSize);
                assert(_cellCountX > 0 and _cellCountY > 0);

                _searchSpaceIndexContainer.resize(_cellCountY * _cellCountX);

                // Fill depth values, add points to image boxes
                const uint allKeypointSize = inKeypoints.size() + lastKeypointsWithIds._keypoints.size();
                _depths = std::vector<double>(allKeypointSize, 0.0);
                _keypoints = std::vector<vector2>(allKeypointSize);

                // Add detected keypoints first
                const size_t keypointIndexOffset = inKeypoints.size();
                for(size_t pointIndex = 0; pointIndex < keypointIndexOffset; ++pointIndex) {
                    const cv::Point2f& pt = inKeypoints[pointIndex];;
                    const vector2 vectorKeypoint(pt.x, pt.y); 

                    _keypoints[pointIndex] = vectorKeypoint; 

                    const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint));
                    assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                    _searchSpaceIndexContainer[searchSpaceIndex].push_back(pointIndex);

                    // Depths are in millimeters, will be 0 if coordinates are invalid
                    _depths[pointIndex] = get_depth_approximation(depthImage, pt);
                }


                // Add optical flow keypoints then 
                const size_t opticalPointSize = lastKeypointsWithIds._keypoints.size();
                for(size_t pointIndex = 0; pointIndex < opticalPointSize; ++pointIndex) {
                    const size_t newKeypointIndex = pointIndex + keypointIndexOffset;

                    // fill in unique point index
                    const size_t uniqueIndex = lastKeypointsWithIds._ids[pointIndex];
                    if (uniqueIndex > 0) {
                        _uniqueIdsToKeypointIndex[uniqueIndex] = newKeypointIndex;
                    }
                    else {
                        utils::log_error("A keypoint detected by optical flow does nothave a valid keypoint id");
                    }

                    const cv::Point2f& pt = lastKeypointsWithIds._keypoints[pointIndex];;
                    const vector2 vectorKeypoint(pt.x, pt.y); 

#if 0
                    // add to matcher (not activated = never matched with descriptors)
                    const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint));
                    assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                    _searchSpaceIndexContainer[searchSpaceIndex].push_back(newKeypointIndex);
#endif

                    _keypoints[newKeypointIndex] = vectorKeypoint; 

                    // Depths are in millimeters, will be 0 if coordinates are invalid
                    _depths[newKeypointIndex] = get_depth_approximation(depthImage, pt);
                }
            }


            uint Keypoint_Handler::get_search_space_index(const int_pair& searchSpaceIndex) const
            {
                return get_search_space_index(searchSpaceIndex.second, searchSpaceIndex.first);
            }
            uint Keypoint_Handler::get_search_space_index(const uint x, const uint y) const 
            {
                return y * _cellCountY + x;
            }


            const Keypoint_Handler::int_pair Keypoint_Handler::get_search_space_coordinates(const vector2& pointToPlace) const
            {
                const double cellSize = static_cast<double>(Parameters::get_search_matches_cell_size());
                const int_pair cellCoordinates(
                        std::clamp(floor(pointToPlace.y() / cellSize), 0.0, _cellCountY - 1.0),
                        std::clamp(floor(pointToPlace.x() / cellSize), 0.0, _cellCountX - 1.0)
                        );
                return cellCoordinates;
            }

            const cv::Mat Keypoint_Handler::compute_key_point_mask(const vector2& pointToSearch, const std::vector<bool>& isKeyPointMatchedContainer) const
            {
                const int_pair& searchSpaceCoordinates = get_search_space_coordinates(pointToSearch);

                const uint startY = std::max(0, searchSpaceCoordinates.first - _searchSpaceCellRadius);
                const uint startX = std::max(0, searchSpaceCoordinates.second - _searchSpaceCellRadius);

                const uint endY = std::min(_cellCountY, searchSpaceCoordinates.first + _searchSpaceCellRadius + 1);
                const uint endX = std::min(_cellCountX, searchSpaceCoordinates.second + _searchSpaceCellRadius + 1);

                // Squared search diameter, to compare distance without sqrt
                const float squaredSearchDiameter = pow(Parameters::get_search_matches_distance(), 2);

                cv::Mat keyPointMask(cv::Mat::zeros(1, _descriptors.rows, CV_8UC1));
                for (uint i = startY; i < endY; ++i)
                {
                    for (uint j = startX; j < endX; ++j)
                    {
                        const size_t searchSpaceIndex = get_search_space_index(j, i);
                        assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                        const index_container& keypointIndexContainer = _searchSpaceIndexContainer[searchSpaceIndex]; 
                        for(const int keypointIndex : keypointIndexContainer)
                        {
                            if (not isKeyPointMatchedContainer[keypointIndex])
                            {
                                const vector2& keypoint = get_keypoint(keypointIndex);
                                const double squarredDistance = 
                                    pow(keypoint.x() - pointToSearch.x(), 2.0) + 
                                    pow(keypoint.y() - pointToSearch.y(), 2.0);

                                if (squarredDistance <= squaredSearchDiameter)
                                    keyPointMask.at<uint8_t>(0, keypointIndex) = 1;
                            }
                        }
                    }
                }

                return keyPointMask;
            }

            int Keypoint_Handler::get_tracking_match_index(const size_t mapPointId) const
            {
                if (_keypoints.empty())
                    return INVALID_MATCH_INDEX;

                // search if the keypoint id is in the detected points
                if (mapPointId != INVALID_MAP_POINT_ID)
                {
                    // return the match if it's the case
                    uintToUintContainer::const_iterator uniqueIndexIterator = _uniqueIdsToKeypointIndex.find(mapPointId);
                    if (uniqueIndexIterator != _uniqueIdsToKeypointIndex.cend()) {
                        return static_cast<int>(uniqueIndexIterator->second);
                    }
                }
                return INVALID_MATCH_INDEX;
            }

            int Keypoint_Handler::get_tracking_match_index(const size_t mapPointId, const std::vector<bool>& isKeyPointMatchedContainer) const
            {
                assert(isKeyPointMatchedContainer.size() == _keypoints.size());

                if (mapPointId != INVALID_MAP_POINT_ID) {
                    const int trackingIndex = get_tracking_match_index(mapPointId);
                    if (trackingIndex != INVALID_MATCH_INDEX)
                    {
                        if (!isKeyPointMatchedContainer[trackingIndex]) {
                            return trackingIndex;
                        }
                        else {
                            // Somehow, this unique index is already associated with another keypoint
                            utils::log_error("The requested point unique index is already matched");
                        }
                    }
                }
                return INVALID_MATCH_INDEX;
            }

            int Keypoint_Handler::get_match_index(const vector2& projectedMapPoint, const cv::Mat& mapPointDescriptor, const std::vector<bool>& isKeyPointMatchedContainer) const
            {
                assert(isKeyPointMatchedContainer.size() == _keypoints.size());
                // cannot compute matches without a match or descriptors
                if (_keypoints.empty() or _descriptors.rows <= 0)
                    return INVALID_MATCH_INDEX;

                // check descriptor dimensions
                assert(mapPointDescriptor.cols == _descriptors.cols);

                const cv::Mat& keyPointMask = compute_key_point_mask(projectedMapPoint, isKeyPointMatchedContainer);

                std::vector< std::vector<cv::DMatch> > knnMatches;
                _featuresMatcher->knnMatch(mapPointDescriptor, _descriptors, knnMatches, 2, keyPointMask);

                //check the farthest neighbors
                if (knnMatches[0].size() > 1) {
                    const std::vector<cv::DMatch>& match = knnMatches[0];
                    //check if point is a good match by checking it's distance to the second best matched point
                    if (match[0].distance < _maxMatchDistance * match[1].distance) {
                        int id = match[0].trainIdx;
                        return id;   //this frame key point
                    }
                    return INVALID_MATCH_INDEX;
                }
                else if (knnMatches[0].size() == 1) {
                    int id = knnMatches[0][0].trainIdx;
                    return id;   //this frame key point
                }
                return INVALID_MATCH_INDEX;
            }

        }
    }
}
