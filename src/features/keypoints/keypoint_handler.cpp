#include "keypoint_handler.hpp"

#include "../../parameters.hpp"
#include "../../outputs/logger.hpp"
#include "../../types.hpp"

namespace rgbd_slam {
    namespace features {
        namespace keypoints {


            bool is_in_border(const cv::Point2f &pt, const cv::Mat &im, const double borderSize) 
            {
                assert(borderSize >= 0);
                return 
                    borderSize <= pt.x and
                    borderSize <= pt.y and
                    pt.x < static_cast<double>(im.cols) - borderSize and
                    pt.y < static_cast<double>(im.rows) - borderSize;
            }

            double get_depth_approximation(const cv::Mat& depthImage, const cv::Point2f& depthCoordinates)
            {
                const double border = BORDER_SIZE;
                assert(border > 0);
                if (is_in_border(depthCoordinates, depthImage, border)) 
                {
                    const cv::Mat roi(depthImage(
                                cv::Rect(
                                    std::max(int(depthCoordinates.x - border), 0),
                                    std::max(int(depthCoordinates.y - border), 0),
                                    int(border * 2.0), 
                                    int(border * 2.0))
                                ));
                    double min, max;
                    cv::minMaxLoc(roi, &min, &max);
                    return min;
                }
                return 0.0;
            }


            Keypoint_Handler::Keypoint_Handler(std::vector<cv::Point2f>& inKeypoints, cv::Mat& inDescriptors, const KeypointsWithIdStruct& lastKeypointsWithIds, const cv::Mat& depthImage, const double maxMatchDistance) :
                _maxMatchDistance(maxMatchDistance)
            {
                if (_maxMatchDistance <= 0) {
                    outputs::log_error("Maximum matching distance must be > 0");
                    exit(-1);
                }
                // knn matcher
                _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));

                _descriptors = inDescriptors;

                const static double cellSize = static_cast<double>(Parameters::get_search_matches_cell_size());
                assert(cellSize > 0);
                _searchSpaceCellRadius = static_cast<uint>(std::ceil(Parameters::get_search_matches_distance() / cellSize));
                assert(_searchSpaceCellRadius > 0);

                _cellCountX = static_cast<uint>(std::ceil(depthImage.cols / cellSize));
                _cellCountY = static_cast<uint>(std::ceil(depthImage.rows / cellSize));
                assert(_cellCountX > 0 and _cellCountY > 0);

                _searchSpaceIndexContainer.resize(_cellCountY * _cellCountX);

                // Fill depth values, add points to image boxes
                const size_t allKeypointSize = inKeypoints.size() + lastKeypointsWithIds._keypoints.size();
                _keypoints = std::vector<utils::ScreenCoordinate>(allKeypointSize);

                // Add detected keypoints first
                const uint keypointIndexOffset = static_cast<uint>(inKeypoints.size());
                for(uint pointIndex = 0; pointIndex < keypointIndexOffset; ++pointIndex) {
                    const cv::Point2f& pt = inKeypoints[pointIndex];
                    // Depths are in millimeters, will be 0 if coordinates are invalid
                    const double associatedDepth = get_depth_approximation(depthImage, pt);
                    const utils::ScreenCoordinate vectorKeypoint(pt.x, pt.y, associatedDepth); 

                    _keypoints[pointIndex] = vectorKeypoint; 

                    const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint));
                    assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                    _searchSpaceIndexContainer[searchSpaceIndex].push_back(pointIndex);
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
                        outputs::log_error("A keypoint detected by optical flow does nothave a valid keypoint id");
                    }

                    const cv::Point2f& pt = lastKeypointsWithIds._keypoints[pointIndex];
                    // Depths are in millimeters, will be 0 if coordinates are invalid
                    const double depthApproximation = get_depth_approximation(depthImage, pt);
                    const utils::ScreenCoordinate vectorKeypoint(pt.x, pt.y, depthApproximation); 

#if 0
                    // add to matcher (not activated = never matched with descriptors)
                    const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint));
                    assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                    _searchSpaceIndexContainer[searchSpaceIndex].push_back(newKeypointIndex);
#endif

                    _keypoints[newKeypointIndex] = vectorKeypoint;
                }
            }


            uint Keypoint_Handler::get_search_space_index(const uint_pair& searchSpaceIndex) const
            {
                return get_search_space_index(searchSpaceIndex.second, searchSpaceIndex.first);
            }
            uint Keypoint_Handler::get_search_space_index(const uint x, const uint y) const 
            {
                return y * _cellCountY + x;
            }


            const Keypoint_Handler::uint_pair Keypoint_Handler::get_search_space_coordinates(const utils::ScreenCoordinate2D& pointToPlace) const
            {
                const static double cellSize = static_cast<double>(Parameters::get_search_matches_cell_size());
                const uint_pair cellCoordinates(
                        std::clamp(floor(pointToPlace.y() / cellSize), 0.0, _cellCountY - 1.0),
                        std::clamp(floor(pointToPlace.x() / cellSize), 0.0, _cellCountX - 1.0)
                        );
                return cellCoordinates;
            }

            const cv::Mat Keypoint_Handler::compute_key_point_mask(const utils::ScreenCoordinate2D& pointToSearch, const std::vector<bool>& isKeyPointMatchedContainer) const
            {
                const uint_pair& searchSpaceCoordinates = get_search_space_coordinates(pointToSearch);
                // compute a search zone for the potential matches of this point
                const uint startY = std::max(0U, searchSpaceCoordinates.first - _searchSpaceCellRadius);
                const uint startX = std::max(0U, searchSpaceCoordinates.second - _searchSpaceCellRadius);

                const uint endY = std::min(_cellCountY, searchSpaceCoordinates.first + _searchSpaceCellRadius + 1);
                const uint endX = std::min(_cellCountX, searchSpaceCoordinates.second + _searchSpaceCellRadius + 1);

                // Squared search diameter, to compare distance without sqrt
                const static float squaredSearchDiameter = static_cast<float>(pow(Parameters::get_search_matches_distance(), 2.0));

                // set a mask of the size of the keypoints, with everything at zero (nothing can be matched)
                cv::Mat keyPointMask(cv::Mat::zeros(1, _descriptors.rows, CV_8UC1));
                for (uint i = startY; i < endY; ++i)
                {
                    for (uint j = startX; j < endX; ++j)
                    {
                        const size_t searchSpaceIndex = get_search_space_index(j, i);
                        assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

                        // get all keypoints in this area
                        const index_container& keypointIndexContainer = _searchSpaceIndexContainer[searchSpaceIndex]; 
                        for(const uint keypointIndex : keypointIndexContainer)
                        {
                            // ignore this point if it is already matched (prevent multiple matches of one point)
                            if (not isKeyPointMatchedContainer[keypointIndex])
                            {
                                const utils::ScreenCoordinate2D& keypoint = get_keypoint(keypointIndex);
                                const double squarredDistance = (keypoint - pointToSearch).squaredNorm();

                                // keypoint is in a circle around the target keypoints, allow a potential match
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
                        assert(trackingIndex >= 0 and static_cast<size_t>(trackingIndex) < isKeyPointMatchedContainer.size());
                        if (not isKeyPointMatchedContainer[static_cast<size_t>(trackingIndex)]) {
                            return trackingIndex;
                        }
                        else {
                            // Somehow, this unique index is already associated with another keypoint
                            outputs::log_error("The requested point unique index is already matched");
                        }
                    }
                }
                return INVALID_MATCH_INDEX;
            }

            int Keypoint_Handler::get_match_index(const utils::ScreenCoordinate2D& projectedMapPoint, const cv::Mat& mapPointDescriptor, const std::vector<bool>& isKeyPointMatchedContainer) const
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

                assert(knnMatches.size() > 0);
                const std::vector<cv::DMatch>& firstMatch = knnMatches[0];

                //check the farthest neighbors
                if (firstMatch.size() > 1) {
                    //check if point is a good match by checking it's distance to the second best matched point
                    if (firstMatch[0].distance < _maxMatchDistance * firstMatch[1].distance) {
                        int id = firstMatch[0].trainIdx;
                        return id;   //this frame key point
                    }
                    return INVALID_MATCH_INDEX;
                }
                else if (firstMatch.size() == 1) {
                    int id = firstMatch[0].trainIdx;
                    return id;   //this frame key point
                }
                return INVALID_MATCH_INDEX;
            }

        }
    }
}
