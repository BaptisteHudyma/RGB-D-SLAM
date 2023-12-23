#include "keypoint_handler.hpp"
#include "../../outputs/logger.hpp"
#include "../../parameters.hpp"
#include "../../types.hpp"
#include <cstddef>

namespace rgbd_slam::features::keypoints {

bool is_in_border(const cv::Point2f& pt, const cv::Mat& im, const double borderSize) noexcept
{
    assert(borderSize >= 0);
    return borderSize <= pt.x and borderSize <= pt.y and pt.x < static_cast<double>(im.cols) - borderSize and
           pt.y < static_cast<double>(im.rows) - borderSize;
}

double get_depth_approximation(const cv::Mat_<float>& depthImage, const cv::Point2f& depthCoordinates) noexcept
{
    const double border = BORDER_SIZE;
    assert(border > 0);
    if (is_in_border(depthCoordinates, depthImage, border))
    {
        const cv::Mat_<float> roi(depthImage(cv::Rect(std::max(int(depthCoordinates.x - border), 0),
                                                      std::max(int(depthCoordinates.y - border), 0),
                                                      int(border * 2.0),
                                                      int(border * 2.0))));
        double min;
        double max;
        cv::minMaxLoc(roi, &min, &max);
        if (min <= 0)
            return max;
        else
            return min;
    }
    return 0.0;
}

Keypoint_Handler::Keypoint_Handler(const uint depthImageCols,
                                   const uint depthImageRows,
                                   const double maxMatchDistance) :
    _maxMatchDistance(maxMatchDistance)
{
    if (_maxMatchDistance <= 0)
    {
        outputs::log_error("Maximum matching distance must be > 0");
        exit(-1);
    }
    // knn matcher
    _featuresMatcher = cv::Ptr<cv::BFMatcher>(new cv::BFMatcher(cv::NORM_HAMMING, false));

    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    static_assert(cellSize > 0);

    _cellCountX = static_cast<uint>(std::ceil(depthImageCols / cellSize));
    _cellCountY = static_cast<uint>(std::ceil(depthImageRows / cellSize));
    assert(_cellCountX > 0 and _cellCountY > 0);

    _searchSpaceIndexContainer.resize(static_cast<size_t>(_cellCountY) * _cellCountX);
    for (index_container& indexContainer: _searchSpaceIndexContainer)
    {
        indexContainer.reserve(5);
    }
}

void Keypoint_Handler::clear() noexcept
{
    _keypoints.clear();
    _uniqueIdsToKeypointIndex.clear();

    for (index_container& indexContainer: _searchSpaceIndexContainer)
    {
        indexContainer.clear();
    }

    //_descriptors.release();
}

void Keypoint_Handler::set(std::vector<cv::Point2f>& inKeypoints,
                           const cv::Mat& inDescriptors,
                           const KeypointsWithIdStruct& lastKeypointsWithIds,
                           const cv::Mat_<float>& depthImage) noexcept
{
    // clear last state
    clear();

    _descriptors = inDescriptors;

    // Fill depth values, add points to image boxes
    const size_t allKeypointSize = inKeypoints.size() + lastKeypointsWithIds.size();
    _keypoints = std::vector<ScreenCoordinate>(allKeypointSize);

    // Add detected keypoints first
    const uint keypointIndexOffset = static_cast<uint>(inKeypoints.size());
    for (uint pointIndex = 0; pointIndex < keypointIndexOffset; ++pointIndex)
    {
        const cv::Point2f& pt = inKeypoints[pointIndex];
        // Depths are in millimeters, will be 0 if coordinates are invalid
        const double associatedDepth = get_depth_approximation(depthImage, pt);
        const ScreenCoordinate vectorKeypoint(pt.x, pt.y, associatedDepth);

        _keypoints[pointIndex] = vectorKeypoint;

        const uint searchSpaceIndex = get_search_space_index(get_search_space_coordinates(vectorKeypoint.get_2D()));
        assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

        _searchSpaceIndexContainer[searchSpaceIndex].push_back(pointIndex);
    }

    // Add optical flow keypoints then
    const size_t opticalPointSize = lastKeypointsWithIds.size();
    for (size_t pointIndex = 0; pointIndex < opticalPointSize; ++pointIndex)
    {
        const size_t newKeypointIndex = pointIndex + keypointIndexOffset;

        // fill in unique point index
        const KeypointsWithIdStruct::keypointWithId& kp = lastKeypointsWithIds.at(pointIndex);
        const size_t uniqueIndex = kp._id;
        const cv::Point2f& pt = kp._point;

        if (uniqueIndex > 0)
        {
            _uniqueIdsToKeypointIndex[uniqueIndex] = newKeypointIndex;
        }
        else
        {
            outputs::log_error("A keypoint detected by optical flow does not have a valid keypoint id");
        }

        // Depths are in millimeters, will be 0 if coordinates are invalid
        const double depthApproximation = get_depth_approximation(depthImage, pt);
        const ScreenCoordinate vectorKeypoint(pt.x, pt.y, depthApproximation);

        // no search space: already matched
        _keypoints[newKeypointIndex] = vectorKeypoint;
    }
}

uint Keypoint_Handler::get_search_space_index(const uint_pair& searchSpaceIndex) const noexcept
{
    return get_search_space_index(searchSpaceIndex.second, searchSpaceIndex.first);
}
uint Keypoint_Handler::get_search_space_index(const uint x, const uint y) const noexcept { return y * _cellCountY + x; }

Keypoint_Handler::uint_pair Keypoint_Handler::get_search_space_coordinates(
        const ScreenCoordinate2D& pointToPlace) const noexcept
{
    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    const uint_pair cellCoordinates(std::clamp(floor(pointToPlace.y() / cellSize), 0.0, _cellCountY - 1.0),
                                    std::clamp(floor(pointToPlace.x() / cellSize), 0.0, _cellCountX - 1.0));
    return cellCoordinates;
}

cv::Mat_<uchar> Keypoint_Handler::compute_key_point_mask(const ScreenCoordinate2D& pointToSearch,
                                                         const vectorb& isKeyPointMatchedContainer,
                                                         const uint searchSpaceCellRadius) const noexcept
{
    const auto [searchSpaceCoordinatesY, searchSpaceCoordinatesX] = get_search_space_coordinates(pointToSearch);
    // compute a search zone for the potential matches of this point
    const uint startY = std::max(0U, searchSpaceCoordinatesY - searchSpaceCellRadius);
    const uint startX = std::max(0U, searchSpaceCoordinatesX - searchSpaceCellRadius);

    const uint endY = std::min(_cellCountY, searchSpaceCoordinatesY + searchSpaceCellRadius + 1);
    const uint endX = std::min(_cellCountX, searchSpaceCoordinatesX + searchSpaceCellRadius + 1);

    // set a mask of the size of the keypoints, with everything at zero (nothing can be matched)
    cv::Mat_<uchar> keyPointMask = cv::Mat_<float>::zeros(1, _descriptors.rows);
    for (uint i = startY; i < endY; ++i)
    {
        for (uint j = startX; j < endX; ++j)
        {
            const size_t searchSpaceIndex = get_search_space_index(j, i);
            assert(searchSpaceIndex < _searchSpaceIndexContainer.size());

            // get all keypoints in this area
            const index_container& keypointIndexContainer = _searchSpaceIndexContainer[searchSpaceIndex];
            fill_keypoint_mask(pointToSearch, keypointIndexContainer, isKeyPointMatchedContainer, keyPointMask);
        }
    }

    return keyPointMask;
}

void Keypoint_Handler::fill_keypoint_mask(const ScreenCoordinate2D& pointToSearch,
                                          const index_container& keypointIndexContainer,
                                          const vectorb& isKeyPointMatchedContainer,
                                          cv::Mat_<uchar>& keyPointMask) const noexcept
{
    // Squared search diameter, to compare distance without sqrt
    constexpr float squaredSearchDiameter = static_cast<float>(SQR(parameters::matching::matchSearchRadius_px));
    for (const uint keypointIndex: keypointIndexContainer)
    {
        // ignore this point if it is already matched (prevent multiple matches of one point)
        if (not isKeyPointMatchedContainer[keypointIndex])
        {
            const ScreenCoordinate2D& keypoint = get_keypoint(keypointIndex).get_2D();
            const double squarredDistance = (keypoint - pointToSearch).squaredNorm();

            // keypoint is in a circle around the target keypoints, allow a potential match
            if (squarredDistance <= squaredSearchDiameter)
                keyPointMask(0, static_cast<int>(keypointIndex)) = 1;
        }
    }
}

int Keypoint_Handler::get_tracking_match_index(const size_t mapPointId) const noexcept
{
    if (_keypoints.empty())
        return INVALID_MATCH_INDEX;

    // search if the keypoint id is in the detected points
    if (mapPointId != INVALID_MAP_POINT_ID)
    {
        // return the match if it's the case
        uintToUintContainer::const_iterator uniqueIndexIterator = _uniqueIdsToKeypointIndex.find(mapPointId);
        if (uniqueIndexIterator != _uniqueIdsToKeypointIndex.cend())
        {
            return static_cast<int>(uniqueIndexIterator->second);
        }
    }
    return INVALID_MATCH_INDEX;
}

int Keypoint_Handler::get_tracking_match_index(const size_t mapPointId,
                                               const vectorb& isKeyPointMatchedContainer) const noexcept
{
    assert(static_cast<size_t>(isKeyPointMatchedContainer.size()) == _keypoints.size());

    if (mapPointId != INVALID_MAP_POINT_ID)
    {
        const int trackingIndex = get_tracking_match_index(mapPointId);
        if (trackingIndex != INVALID_MATCH_INDEX)
        {
            assert(trackingIndex >= 0 and static_cast<Eigen::Index>(trackingIndex) < isKeyPointMatchedContainer.size());
            if (not isKeyPointMatchedContainer[static_cast<Eigen::Index>(trackingIndex)])
            {
                return trackingIndex;
            }
            else
            {
                // Somehow, this unique index is already associated with another keypoint
                outputs::log_error("The requested point unique index is already matched");
            }
        }
    }
    return INVALID_MATCH_INDEX;
}

int Keypoint_Handler::get_match_index(const ScreenCoordinate2D& projectedMapPoint,
                                      const cv::Mat& mapPointDescriptor,
                                      const vectorb& isKeyPointMatchedContainer,
                                      const double searchSpaceRadius) const noexcept
{
    assert(_featuresMatcher != nullptr);
    assert(static_cast<size_t>(isKeyPointMatchedContainer.size()) == _keypoints.size());

    // cannot compute matches without a match or descriptors
    if (_keypoints.empty() or _descriptors.rows <= 0)
        return INVALID_MATCH_INDEX;

    constexpr double cellSize = parameters::matching::matchSearchRadius_px + 1.0;
    static_assert(cellSize > 0);
    const uint searchSpaceCellRadius = static_cast<uint>(std::ceil(searchSpaceRadius / cellSize));
    assert(searchSpaceCellRadius > 0);

    // check descriptor dimensions
    assert(!mapPointDescriptor.empty());
    assert(mapPointDescriptor.cols == _descriptors.cols);

    const cv::Mat_<uchar>& keyPointMask =
            compute_key_point_mask(projectedMapPoint, isKeyPointMatchedContainer, searchSpaceCellRadius);

    std::vector<std::vector<cv::DMatch>> knnMatches;
    _featuresMatcher->knnMatch(mapPointDescriptor, _descriptors, knnMatches, 2, keyPointMask);

    assert(knnMatches.size() > 0);

    // check the farthest neighbors
    const std::vector<cv::DMatch>& firstMatch = knnMatches[0];
    if (firstMatch.size() > 1)
    {
        // check if point is a good match by checking it's distance to the second best matched point
        if (firstMatch[0].distance < _maxMatchDistance * firstMatch[1].distance)
        {
            int id = firstMatch[0].trainIdx;
            return id; // this frame key point
        }
        return INVALID_MATCH_INDEX;
    }
    else if (firstMatch.size() == 1)
    {
        int id = firstMatch[0].trainIdx;
        return id; // this frame key point
    }
    return INVALID_MATCH_INDEX;
}

} // namespace rgbd_slam::features::keypoints
